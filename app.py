import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import os
import json
import uuid
import streamlit.components.v1 as components
import re
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
from tabs.employee_management import render_employee_management_tab
from tabs.group_rules import render_group_rules_tab
from tabs.availability_editor import render_availability_tab
from tabs.schedule_output import render_schedule_tab
from tabs.import_export import render_import_export_tab
from utils import availability_utils
from scheduling_logic import (
    load_employees,
    load_data,
    save_employees,
    save_data,
    init_availability,
    generate_schedule,
    import_from_google_form,
    import_employees_from_main_excel,
    add_employee,
    edit_employee,
    delete_employee,
    restore_employees_from_storage,
    save_employees_to_storage_only,
    export_availability_to_excel,
    clear_availability,
    sync_availability,
    get_last_generated_schedule,
    load_role_rules,
    EMPLOYEES,
    ROLE_RULES,
)

# --- Optional: group rules (backward compatible with older deployments) ---
try:
    from scheduling_logic import load_group_rules, save_group_rules, GROUP_RULES  # type: ignore
    GROUP_RULES_ENABLED = True
except Exception:
    GROUP_RULES_ENABLED = False

    def load_group_rules():  # type: ignore
        return {"version": 1, "updated_at": None, "groups": []}

    def save_group_rules(_group_rules=None, also_write_local=True):  # type: ignore
        return None

    GROUP_RULES = {"version": 1, "updated_at": None, "groups": []}

import firebase_manager as fm
fm.initialize_firebase()

# Optional: Altair for clickable heatmap (fallback if not available)
try:
    import altair as alt  # type: ignore
except Exception:
    alt = None  # type: ignore

# --- Main App ---
def _safe_get_secret(key: str) -> Optional[str]:
    """Best-effort read from Streamlit secrets without hard-failing in environments w/o secrets."""
    try:
        # st.secrets behaves like a dict; .get may not exist on older versions
        if hasattr(st, "secrets") and st.secrets is not None:
            try:
                if key in st.secrets:
                    return str(st.secrets[key]).strip()
            except Exception:
                pass
            try:
                v = st.secrets.get(key)  # type: ignore[attr-defined]
                if v is not None:
                    return str(v).strip()
            except Exception:
                pass
    except Exception:
        pass
    return None


def _detect_branch_name() -> Optional[str]:
    """Detect current branch name via env vars or .git/HEAD (works in many deploy setups)."""
    env_candidates = [
        "STREAMLIT_GIT_BRANCH",
        "STREAMLIT_BRANCH",
        "GIT_BRANCH",
        "BRANCH_NAME",
        "GITHUB_REF_NAME",
        "GITHUB_REF",
        "CI_COMMIT_REF_NAME",
    ]
    for k in env_candidates:
        v = os.getenv(k)
        if not v:
            continue
        v = v.strip()
        # e.g. refs/heads/test
        if v.startswith("refs/heads/"):
            return v.split("/")[-1]
        # e.g. origin/test
        if "/" in v and not v.startswith("http"):
            return v.split("/")[-1]
        return v

    try:
        head_path = Path(__file__).resolve().parent / ".git" / "HEAD"
        if head_path.exists():
            head = head_path.read_text(encoding="utf-8", errors="ignore").strip()
            if head.startswith("ref:"):
                ref = head.split(":", 1)[1].strip()
                return ref.split("/")[-1]
    except Exception:
        pass

    return None


def _resolve_app_version() -> str:
    """
    Resolve app version marker for UI labeling.
    Priority: Streamlit secrets -> env -> branch name -> stable.
    """
    # 1) Streamlit secrets (supports legacy key with space)
    for key in ("APP_VERSION", "APP VERSION", "APPVERSION"):
        v = _safe_get_secret(key)
        if v:
            return v

    # 2) Environment variables (works for many CI/deploy setups)
    for key in ("APP_VERSION", "APPVERSION"):
        v = os.getenv(key)
        if v:
            return v.strip()

    # 3) Branch-based inference
    branch = _detect_branch_name()
    if branch:
        b = branch.lower()
        if b == "test" or "test" in b:
            return "beta"

    return "stable"


def get_app_title(base_title: str) -> str:
    """Get the appropriate app title based on environment/version."""
    version = _resolve_app_version().lower()
    if version in ("beta", "test", "testing", "dev"):
        return f"{base_title} (Beta)"
    return base_title


st.set_page_config(page_title=get_app_title("BP Schedule Maker"), layout="wide")
st.title(get_app_title("Auto-Schedule Maker"))



# NEW: Add a button for one-time data upload (optional but useful)
if st.button("Upload Initial Data to Firebase"):
    fm.upload_initial_data()


# --- State Management ---
def refresh_master_data():
    """Refresh employees/role rules/group rules from Firebase and sync availability."""
    load_role_rules()
    st.session_state.employees = load_employees()
    st.session_state.group_rules = load_group_rules()
    # Sync employees with availability data
    sync_availability()
    st.session_state.availability = load_data()


def initialize_session_state():
    """Load initial data into the session state."""
    if not st.session_state.get('initialized'):
        st.session_state.start_date = datetime(2025, 3, 17)
        refresh_master_data()

        if not st.session_state.availability:
            st.session_state.availability = init_availability(
                st.session_state.start_date, st.session_state.employees
            )
        st.session_state.initialized = True
        st.session_state.warnings = []
        st.session_state.generated_schedule = None

# --- Helper Functions ---

# --- Compatibility helpers ---
def _df_elementwise(df: pd.DataFrame, func):
    """
    Pandas deprecated DataFrame.applymap in favor of DataFrame.map (elementwise).
    Keep a small shim for compatibility across pandas versions.
    """
    if hasattr(df, "map"):
        return df.map(func)  # pandas >= 2.1
    return df.applymap(func)  # pragma: no cover

def _excel_rgb_to_hex6(rgb):
    """
    Convert openpyxl rgb/argb like 'FFD9D9D9' to 'D9D9D9' (hex6).
    Returns '' if invalid/empty.
    """
    if rgb is None:
        return ""
    s = str(rgb).strip()
    if not s or s.lower() in {"none", "nan"}:
        return ""
    if len(s) == 8:  # ARGB -> RGB
        s = s[2:]
    if len(s) != 6:
        return ""
    # Validate hex
    try:
        int(s, 16)
    except Exception:
        return ""
    return s.upper()

def _auto_text_hex_from_bg(bg_hex6):
    """
    Pick black/white text for readability based on background brightness.
    bg_hex6: 'RRGGBB'
    """
    if not bg_hex6 or len(bg_hex6) != 6:
        return ""
    r = int(bg_hex6[0:2], 16)
    g = int(bg_hex6[2:4], 16)
    b = int(bg_hex6[4:6], 16)
    # Perceived luminance (sRGB)
    luminance = (0.299 * r) + (0.587 * g) + (0.114 * b)
    return "FFFFFF" if luminance < 140 else "000000"

def _excel_rgb_to_css(rgb):
    """
    openpyxl fill.fgColor.rgb is often ARGB like 'FFD9D9D9'. Convert to CSS.
    Returns '' when no color should be applied.
    """
    hex6 = _excel_rgb_to_hex6(rgb)
    # Treat "no fill" / default as empty (openpyxl sometimes gives 00000000)
    if not hex6 or hex6 in {"000000"}:
        return ""
    return f"background-color: #{hex6};"

def _availability_cell_value(cell):
    """Extract display/edit value from availability cell."""
    v = cell
    if isinstance(cell, dict) and "value" in cell:
        v = cell.get("value")
    if v is None:
        return ""
    # If upstream stored an Excel-auto-parsed date for a shift like "10-19", show as "M-D".
    try:
        if isinstance(v, pd.Timestamp):
            if not pd.isna(v):
                dt = v.to_pydatetime()
                if dt.hour == 0 and dt.minute == 0 and dt.second == 0:
                    return f"{dt.month}-{dt.day}"
        if isinstance(v, datetime):
            if v.hour == 0 and v.minute == 0 and v.second == 0:
                return f"{v.month}-{v.day}"
    except Exception:
        pass
    if isinstance(v, list):
        return ", ".join(map(str, v))
    return str(v)

def _parse_time_to_minutes(t: str) -> Optional[int]:
    t = (t or "").strip()
    if not t:
        return None
    # HH:MM
    if ":" in t:
        hh, mm = t.split(":", 1)
        try:
            return int(hh) * 60 + int(mm)
        except Exception:
            return None
    # 4-digit HHMM
    if len(t) == 4 and t.isdigit():
        return int(t[:2]) * 60 + int(t[2:])
    # hour
    try:
        return int(float(t)) * 60
    except Exception:
        return None

def _normalize_cell_str(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, dict) and "value" in v:
        v = v.get("value")
    if v is None:
        return ""
    return str(v).strip()

_LEAVE_CODES = {"AL", "CL", "PH", "ON"}

def _is_leave_like(s: str) -> bool:
    if not s:
        return True
    up = s.upper()
    # explicit off/out/leave markers
    if any(x in up for x in ["OFF", "OUT", "自由調配", "HALF OFF", "PH HALF"]):
        return True
    # token-level leave codes
    tokens = re.split(r"[^A-Z0-9]+", up)
    return any(tok in _LEAVE_CODES for tok in tokens if tok)

def _intervals_from_cell(cell_value: Any) -> List[Tuple[int, int]]:
    """
    Parse a schedule cell into working intervals (minutes).
    Supported:
    - "10-19", "0930-1830"
    - "10 OBS 19" / "7 OBS" (treated as 9h shift: 7-16)
    - numeric hours like "10" / "15" / "7" (treated as 9h shift)
    - multiple segments separated by commas: "8-17,15-24"
    Leave codes (AL/CL/PH/ON etc) are treated as NOT working.
    """
    s = _normalize_cell_str(cell_value)
    if not s:
        return []

    # If clearly leave-like AND no time patterns, treat as off
    if _is_leave_like(s):
        # But allow cells that contain actual time ranges despite annotations.
        has_time_hint = bool(re.search(r"\d", s))
        has_range = bool(re.search(r"\d\s*[-–]\s*\d", s)) or bool(re.search(r"\d{4}\s*-\s*\d{4}", s))
        has_obs = "OBS" in s.upper()
        if not (has_range or has_obs):
            return []

    up = s.upper()

    intervals: List[Tuple[int, int]] = []

    # 1) Explicit time ranges (supports 10-19, 0930-1830, 9:30-18:30)
    range_re = re.compile(r"(\d{1,2}(?::\d{2})?|\d{4})\s*[-–]\s*(\d{1,2}(?::\d{2})?|\d{4})")
    for a, b in range_re.findall(up):
        sm = _parse_time_to_minutes(a)
        em = _parse_time_to_minutes(b)
        if sm is None or em is None:
            continue
        if em <= sm:
            # e.g. 15-24 ok, but 24-0 not expected here; clamp later
            continue
        intervals.append((max(0, sm), min(24 * 60, em)))

    # 2) OBS tagged ranges like "10 OBS 19"
    obs_range_re = re.compile(r"\b(\d{1,2})\s*OBS\s*(\d{1,2})\b")
    for a, b in obs_range_re.findall(up):
        sm = _parse_time_to_minutes(a)
        em = _parse_time_to_minutes(b)
        if sm is None or em is None:
            continue
        if em <= sm:
            continue
        intervals.append((max(0, sm), min(24 * 60, em)))

    # 3) OBS start only like "7 OBS" -> assume 9h
    obs_start_re = re.compile(r"\b(\d{1,2})\s*OBS\b")
    for a in obs_start_re.findall(up):
        sm = _parse_time_to_minutes(a)
        if sm is None:
            continue
        em = sm + 9 * 60
        intervals.append((max(0, sm), min(24 * 60, em)))

    # 4) pure numeric like "10" / "15.0" -> assume 9h, with a small special-case for 9 meaning 09:30
    if not intervals:
        m = re.fullmatch(r"\d+(?:\.\d+)?", s)
        if m:
            hour = int(float(s))
            sm = hour * 60
            # common convention: 9 means 09:30 (matches "0930-1830" seen in your sheets)
            if hour == 9:
                sm = 9 * 60 + 30
            em = sm + 9 * 60
            intervals.append((max(0, sm), min(24 * 60, em)))

    # Merge overlaps
    if not intervals:
        return []
    intervals.sort()
    merged: List[Tuple[int, int]] = [intervals[0]]
    for s0, e0 in intervals[1:]:
        s1, e1 = merged[-1]
        if s0 <= e1:
            merged[-1] = (s1, max(e1, e0))
        else:
            merged.append((s0, e0))
    return merged

def _time_window_to_minutes(start: str, end: str) -> Tuple[int, int]:
    s = _parse_time_to_minutes(start) or 0
    e = _parse_time_to_minutes(end) or 24 * 60
    if end.strip() == "24:00":
        e = 24 * 60
    return max(0, s), min(24 * 60, e)

def validate_group_coverage_from_availability(
    availability: Dict[str, Dict[str, Any]],
    group: Dict[str, Any],
    group_rules: Optional[Dict[str, Any]] = None,
    step_minutes: int = 60,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Return (summary_df, deficits_df, all_checked_df).

    - summary_df: per date shortage summary (units = checked slots, slot size inferred by `time` step)
    - deficits_df: only rows with shortage > 0
    - all_checked_df: all checked rows (including shortage == 0), per date/time slot
    """
    members = [m for m in (group.get("members") or []) if m]
    windows = group.get("requirements_windows") or []
    current_rule_type = str(group.get("rule_type") or "routine").strip().lower()
    if current_rule_type not in {"routine", "task"}:
        current_rule_type = "routine"

    # Pre-parse availability intervals per date/member
    parsed: Dict[str, Dict[str, List[Tuple[int, int]]]] = {}
    for date_key, emps in (availability or {}).items():
        if not isinstance(emps, dict):
            continue
        parsed[date_key] = {}
        for m in members:
            cell = emps.get(m)
            parsed[date_key][m] = _intervals_from_cell(cell)

    # Default to 60-minute granularity for UI visualization.
    try:
        step_minutes = int(step_minutes)
    except Exception:
        step_minutes = 60
    if step_minutes <= 0:
        step_minutes = 60
    all_rows: List[Dict[str, Any]] = []
    for date_key in sorted(parsed.keys()):
        try:
            d = pd.to_datetime(date_key).to_pydatetime()
        except Exception:
            continue

        for w in windows:
            day_type = str(w.get("day_type") or "all").strip().lower()
            if not _day_type_applies_ui(day_type, d):
                continue
            start_s = str(w.get("start") or "00:00")
            end_s = str(w.get("end") or "24:00")
            min_staff = int(w.get("min_staff") or 0)
            ws, we = _time_window_to_minutes(start_s, end_s)
            if we <= ws:
                continue

            # iterate by step_minutes, but only validate slots that overlap the window
            slot_start = (ws // step_minutes) * step_minutes
            while slot_start < we:
                slot_s = slot_start
                slot_e = min(slot_s + step_minutes, 24 * 60)
                slot_start += step_minutes
                if slot_e <= ws or slot_s >= we:
                    continue
                on = []
                for m in members:
                    # Routine rule: members can only take one routine at a time.
                    # If this member is already occupied by a higher-priority routine group,
                    # they cannot be counted for this routine group.
                    if current_rule_type == "routine":
                        conflict_group = _member_conflict_group_name(
                            member=m,
                            date_obj=d,
                            slot_s=slot_s,
                            slot_e=slot_e,
                            current_group=group,
                            group_rules=group_rules or {},
                        )
                        if conflict_group:
                            continue
                    for is0, ie0 in parsed[date_key].get(m, []):
                        # any overlap counts
                        if max(is0, slot_s) < min(ie0, slot_e):
                            on.append(m)
                            break
                staffed = len(on)
                shortage = max(0, min_staff - staffed)
                all_rows.append(
                    {
                        "date": date_key,
                        "time": _format_minutes_to_hhmm(slot_s),
                        "required": min_staff,
                        "staffed": staffed,
                        "shortage": shortage,
                        "on_duty": "、".join(on),
                        "window": f"{day_type} {start_s}-{end_s}",
                    }
                )

    all_checked_df = pd.DataFrame(all_rows)
    if all_checked_df.empty:
        summary_df = pd.DataFrame(columns=["date", "shortage_units", "total_shortage"])
        deficits_df = pd.DataFrame(columns=["date", "time", "required", "staffed", "shortage", "on_duty", "window"])
        return summary_df, deficits_df, all_checked_df

    # De-duplicate overlaps: per date/time keep max(required); staffed is identical; recompute shortage
    all_checked_df["required"] = pd.to_numeric(all_checked_df["required"], errors="coerce").fillna(0).astype(int)
    all_checked_df["staffed"] = pd.to_numeric(all_checked_df["staffed"], errors="coerce").fillna(0).astype(int)
    all_checked_df = (
        all_checked_df.groupby(["date", "time"], as_index=False)
        .agg(
            required=("required", "max"),
            staffed=("staffed", "max"),
            shortage=("shortage", "max"),
            on_duty=("on_duty", lambda x: next((s for s in x if str(s).strip()), "")),
            window=("window", lambda x: " | ".join(sorted(set([str(s) for s in x if str(s).strip()])))),
        )
    )
    all_checked_df["shortage"] = (all_checked_df["required"] - all_checked_df["staffed"]).clip(lower=0)

    deficits_df = all_checked_df[all_checked_df["shortage"] > 0].copy()
    if deficits_df.empty:
        summary_df = pd.DataFrame(columns=["date", "shortage_units", "total_shortage"])
        return summary_df, deficits_df, all_checked_df

    summary_df = (
        deficits_df.groupby("date")
        .agg(shortage_units=("shortage", "count"), total_shortage=("shortage", "sum"))
        .reset_index()
        .sort_values(["total_shortage", "shortage_units"], ascending=False)
    )
    return summary_df, deficits_df, all_checked_df


def _build_group_coverage_heatmap_df(all_checked_df: pd.DataFrame, step_minutes: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build a Mon-Sun x time-slots table with deficit rate (0..1) and checked counts.
    Cells outside validated windows are NaN.
    Returns (rate_df, count_df).
    """
    if all_checked_df is None or all_checked_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    try:
        step_minutes = int(step_minutes)
    except Exception:
        step_minutes = 60
    if step_minutes <= 0:
        step_minutes = 60

    df = all_checked_df.copy()
    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date_dt"])
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    df["weekday"] = df["date_dt"].dt.dayofweek  # Mon=0..Sun=6
    df["is_deficit"] = (pd.to_numeric(df["shortage"], errors="coerce").fillna(0) > 0).astype(int)

    agg = (
        df.groupby(["weekday", "time"], as_index=False)
        .agg(checked=("is_deficit", "count"), deficit=("is_deficit", "sum"))
    )
    agg["rate"] = agg["deficit"] / agg["checked"]

    time_slots = [t for t in _time_options(step_minutes) if t != "24:00"]
    col_names = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    rate_df = pd.DataFrame(index=time_slots, columns=col_names, data=float("nan"))
    count_df = pd.DataFrame(index=time_slots, columns=col_names, data=float("nan"))

    wd_to_col = {0: "周一", 1: "周二", 2: "周三", 3: "周四", 4: "周五", 5: "周六", 6: "周日"}
    for _, r in agg.iterrows():
        col = wd_to_col.get(int(r["weekday"]))
        t = str(r["time"])
        if col in rate_df.columns and t in rate_df.index:
            rate_df.loc[t, col] = float(r["rate"])
            count_df.loc[t, col] = int(r["checked"])

    return rate_df, count_df


def _heatmap_style_from_rate(v: Any) -> str:
    """
    Rate heatmap cell style:
    - NaN: gray (not applicable)
    - 0: green-ish (OK)
    - 1: red-ish (deficit)
    """
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "background-color: #f3f4f6; color: #9ca3af;"
        x = float(v)
    except Exception:
        return "background-color: #f3f4f6; color: #9ca3af;"

    x = max(0.0, min(1.0, x))
    # interpolate between light green and light red
    g0 = (217, 242, 217)  # ok
    r1 = (248, 215, 218)  # deficit
    rr = int(g0[0] + (r1[0] - g0[0]) * x)
    gg = int(g0[1] + (r1[1] - g0[1]) * x)
    bb = int(g0[2] + (r1[2] - g0[2]) * x)
    return f"background-color: rgb({rr},{gg},{bb}); color: #111827;"


def _build_week_bins_from_dates(date_keys: List[str]) -> List[Dict[str, Any]]:
    """
    Build calendar 7-day bins from min(date_keys) to max(date_keys).
    Only include bins that contain at least 1 imported date.
    Each bin is: {start_date, end_date, label, dates_in_bin}
    """
    dts = sorted([pd.to_datetime(x, errors="coerce") for x in (date_keys or []) if str(x).strip()])
    dts = [x for x in dts if pd.notna(x)]
    if not dts:
        return []
    min_d = dts[0].date()
    max_d = dts[-1].date()
    have = set([x.date() for x in dts])
    bins: List[Dict[str, Any]] = []
    # Align bins to calendar week (Mon..Sun) to avoid date order "jumping" in the heatmap.
    cur = min_d - timedelta(days=min_d.weekday())
    while cur <= max_d:
        end = min(cur + timedelta(days=6), max_d)
        dates_in = [cur + timedelta(days=i) for i in range((end - cur).days + 1) if (cur + timedelta(days=i)) in have]
        if dates_in:
            label = f"{cur.isoformat()} ~ {end.isoformat()}（{(end - cur).days + 1}天，含{len(dates_in)}天数据）"
            bins.append(
                {
                    "start_date": cur,
                    "end_date": end,
                    "label": label,
                    "dates_in_bin": [d.isoformat() for d in dates_in],
                }
            )
        cur = cur + timedelta(days=7)
    return bins


def _build_week_grid_df(
    *,
    all_checked_df: pd.DataFrame,
    week_start: datetime.date,
    week_end: datetime.date,
    step_minutes: int = 30,
) -> pd.DataFrame:
    """
    Build a 7-day (calendar) grid for Altair:
    rows: time slots; cols: weekday; each cell maps to a specific date in [week_start, week_end].
    """
    try:
        step_minutes = int(step_minutes)
    except Exception:
        step_minutes = 60
    if step_minutes <= 0:
        step_minutes = 60

    if all_checked_df is None or all_checked_df.empty:
        return pd.DataFrame()

    df = all_checked_df.copy()
    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date_dt"])
    if df.empty:
        return pd.DataFrame()
    df["date_only"] = df["date_dt"].dt.date

    # filter to this week range
    df = df[(df["date_only"] >= week_start) & (df["date_only"] <= week_end)]

    # quick lookup
    lookup: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for _, r in df.iterrows():
        dk = r.get("date")
        tm = r.get("time")
        if dk is None or tm is None:
            continue
        lookup[(str(dk), str(tm))] = {
            "required": int(r.get("required") or 0),
            "staffed": int(r.get("staffed") or 0),
            "shortage": int(r.get("shortage") or 0),
        }

    day_labels = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    wd_to_label = {0: "周一", 1: "周二", 2: "周三", 3: "周四", 4: "周五", 5: "周六", 6: "周日"}
    time_slots = [t for t in _time_options(step_minutes) if t != "24:00"]

    rows: List[Dict[str, Any]] = []
    cur = week_start
    while cur <= week_end:
        dk = cur.isoformat()
        wd = cur.weekday()
        wlabel = wd_to_label.get(wd, "")
        for t in time_slots:
            rec = lookup.get((dk, t))
            if rec is None:
                rows.append(
                    {
                        "date": dk,
                        "weekday": wlabel,
                        "time": t,
                        "status": "na",
                        "required": None,
                        "staffed": None,
                        "shortage": None,
                    }
                )
            else:
                shortage = int(rec.get("shortage") or 0)
                rows.append(
                    {
                        "date": dk,
                        "weekday": wlabel,
                        "time": t,
                        "status": "deficit" if shortage > 0 else "ok",
                        "required": int(rec.get("required") or 0),
                        "staffed": int(rec.get("staffed") or 0),
                        "shortage": shortage,
                    }
                )
        cur = cur + timedelta(days=1)

    out = pd.DataFrame(rows)
    # ensure all weekdays appear in axis
    out["weekday"] = pd.Categorical(out["weekday"], categories=day_labels, ordered=True)
    return out


def _extract_date_time_from_obj(obj: Any) -> Optional[Tuple[str, str]]:
    """
    Best-effort extraction of (date, time) from Streamlit chart selection payloads.
    """
    if obj is None:
        return None
    if isinstance(obj, dict):
        if "date" in obj and "time" in obj:
            return (str(obj["date"]), str(obj["time"]))
        for v in obj.values():
            got = _extract_date_time_from_obj(v)
            if got:
                return got
    if isinstance(obj, list):
        for it in obj:
            got = _extract_date_time_from_obj(it)
            if got:
                return got
    # some objects expose .selection
    try:
        sel = getattr(obj, "selection", None)
        got = _extract_date_time_from_obj(sel)
        if got:
            return got
    except Exception:
        pass
    return None


def _is_leave_like_raw(v: Any) -> bool:
    """
    Heuristic: treat non-shift, short alpha codes as leave (AL/SL/CL/...
    Keeps this intentionally permissive; raw value will always be displayed verbatim.
    """
    if v is None:
        return False
    s = str(v).strip()
    if not s:
        return False
    if len(s) > 6:
        return False
    return bool(re.fullmatch(r"[A-Za-z]{1,6}", s))


def _member_conflict_group_name(
    *,
    member: str,
    date_obj: datetime,
    slot_s: int,
    slot_e: int,
    current_group: Dict[str, Any],
    group_rules: Dict[str, Any],
) -> Optional[str]:
    """
    Routine-only conflict rule:
    - A member can only be counted for ONE routine at the same time.
    - For now, "priority" is inferred by group order in group_rules (earlier = higher).
    - Task rules never conflict with routine or task.
    """
    try:
        groups = (group_rules or {}).get("groups", []) or []
    except Exception:
        groups = []

    current_rule_type = str(current_group.get("rule_type") or "routine").strip().lower()
    if current_rule_type not in {"routine", "task"}:
        current_rule_type = "routine"
    if current_rule_type != "routine":
        return None

    def _gkey(g: Dict[str, Any]) -> str:
        return str(g.get("id") or g.get("name") or "").strip()

    cur_key = _gkey(current_group)
    cur_idx = None
    for i, g in enumerate(groups):
        if _gkey(g) == cur_key:
            cur_idx = i
            break

    # If we can't find the current group in the list, skip conflicts to avoid false negatives.
    if cur_idx is None:
        return None

    for i, g in enumerate(groups):
        try:
            if i >= cur_idx:
                continue
            gname = str(g.get("name") or "").strip()
            if not gname:
                continue
            g_rule_type = str(g.get("rule_type") or "routine").strip().lower()
            if g_rule_type not in {"routine", "task"}:
                g_rule_type = "routine"
            if g_rule_type != "routine":
                continue
            if not bool(g.get("active", True)):
                continue
            members = g.get("members") or []
            if member not in members:
                continue
            windows = g.get("requirements_windows") or []
            for w in windows:
                try:
                    if int(w.get("min_staff") or 0) <= 0:
                        continue
                except Exception:
                    pass
                day_type = str(w.get("day_type") or "all").strip().lower()
                if not _day_type_applies_ui(day_type, date_obj):
                    continue
                ws, we = _time_window_to_minutes(str(w.get("start") or "00:00"), str(w.get("end") or "24:00"))
                if we <= ws:
                    continue
                # overlap between this slot and that group's window
                if max(ws, slot_s) < min(we, slot_e):
                    return gname
        except Exception:
            continue
    return None


def _build_cell_member_detail_df(
    *,
    availability: Dict[str, Dict[str, Any]],
    group: Dict[str, Any],
    group_rules: Dict[str, Any],
    date_key: str,
    time_hhmm: str,
    step_minutes: int = 60,
) -> pd.DataFrame:
    """
    Build per-member status table for a specific (date, time slot).
    Shows raw imported cell value verbatim in parentheses.
    """
    members = [m for m in (group.get("members") or []) if m]
    current_rule_type = str(group.get("rule_type") or "routine").strip().lower()
    if current_rule_type not in {"routine", "task"}:
        current_rule_type = "routine"

    try:
        date_obj = pd.to_datetime(date_key).to_pydatetime()
    except Exception:
        date_obj = datetime.now()

    slot_s = _parse_time_to_minutes(time_hhmm) or 0
    slot_e = min(slot_s + int(step_minutes), 24 * 60)

    emps = (availability or {}).get(date_key, {}) or {}
    rows: List[Dict[str, Any]] = []
    for m in members:
        raw = emps.get(m)
        raw_s = _normalize_cell_str(raw)
        intervals = _intervals_from_cell(raw)
        on_duty = any(max(is0, slot_s) < min(ie0, slot_e) for is0, ie0 in intervals)

        conflict_group = _member_conflict_group_name(
            member=m,
            date_obj=date_obj,
            slot_s=slot_s,
            slot_e=slot_e,
            current_group=group,
            group_rules=group_rules,
        )

        # Leave-like raw values first (AL/SL/...)
        if _is_leave_like_raw(raw_s) and not intervals:
            status = "请假"
            detail = f"{raw_s}" if raw_s else ""
        # Then conflict (higher priority group) overrides "到岗"
        elif conflict_group:
            status = "被例行占用"
            if raw_s:
                detail = f"{raw_s}（{conflict_group}）"
            else:
                detail = f"（空，{conflict_group}）"
        elif on_duty:
            status = "到岗"
            detail = raw_s if raw_s else ""
        else:
            status = "未到岗"
            detail = raw_s if raw_s else ""

        rows.append({"成员": m, "状态": status, "明细": detail})

    return pd.DataFrame(rows)


# -----------------------------
# Group rules editor helpers
# -----------------------------
_DOW_KEYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
# UI focuses on Mon-Sun (+ all). Backend remains compatible with weekday/weekend.
_DAY_TYPE_OPTIONS_BASE = ["all"] + _DOW_KEYS + ["weekday", "weekend"]


def _time_options(step_minutes: int = 30) -> List[str]:
    opts: List[str] = []
    m = 0
    while m < 24 * 60:
        h = m // 60
        mm = m % 60
        opts.append(f"{h:02d}:{mm:02d}")
        m += step_minutes
    opts.append("24:00")
    return opts


_TIME_OPTIONS_BASE = _time_options(30)


def _format_minutes_to_hhmm(m: int) -> str:
    if m == 1440:
        return "24:00"
    h = m // 60
    mm = m % 60
    return f"{h:02d}:{mm:02d}"


def _normalize_time_str_ui(v: Any) -> str:
    """
    Normalize time strings to canonical "HH:MM" (or "24:00").
    Accepts legacy formats like "0700", "7", "24", "2400", "09:30".
    Returns "" if empty/None-like/invalid.
    """
    if v is None:
        return ""
    s = str(v).strip()
    if not s:
        return ""
    if s.lower() in {"none", "nan", "null"}:
        return ""
    m = _parse_time_to_minutes(s)
    if m is None:
        return ""
    return _format_minutes_to_hhmm(int(m))


def _normalize_windows_df_for_editor(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Prepare windows df for UI editor:
    - normalize day_type to canonical keys
    - normalize start/end to "HH:MM" / "24:00"
    - drop rows with missing start/end (common garbage rows like "None")
    Returns (new_df, dropped_rows_count)
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df, 0
    out = df.copy()
    dropped = 0

    out["day_type"] = out.get("day_type", "all").apply(_normalize_day_type_ui)
    out["start"] = out.get("start", "00:00").apply(_normalize_time_str_ui)
    out["end"] = out.get("end", "24:00").apply(_normalize_time_str_ui)

    # Drop rows where start/end missing after normalization
    mask_keep = (out["start"].astype(str).str.strip() != "") & (out["end"].astype(str).str.strip() != "")
    dropped = int((~mask_keep).sum())
    out = out[mask_keep].reset_index(drop=True)

    # Ensure min_staff is int-ish for display; keep as-is if conversion fails
    if "min_staff" in out.columns:
        def _to_int_or_keep(x):
            try:
                return int(x)
            except Exception:
                return x
        out["min_staff"] = out["min_staff"].apply(_to_int_or_keep)

    return out, dropped


def _normalize_day_type_ui(v: Any) -> str:
    s = str(v or "all").strip().lower()
    if not s:
        return "all"
    if s in {"all", "weekday", "weekend"}:
        return s
    # allow Mon/Tue/Wed... or Monday/Tues...
    s3 = s[:3]
    if s3 in set(_DOW_KEYS):
        return s3
    # allow Chinese (best-effort)
    zh = {
        "周一": "mon",
        "星期一": "mon",
        "周二": "tue",
        "星期二": "tue",
        "周三": "wed",
        "星期三": "wed",
        "周四": "thu",
        "星期四": "thu",
        "周五": "fri",
        "星期五": "fri",
        "周六": "sat",
        "星期六": "sat",
        "周日": "sun",
        "周天": "sun",
        "星期日": "sun",
        "星期天": "sun",
        "每天": "all",
        "全部": "all",
        "工作日": "weekday",
        "周末": "weekend",
    }
    return zh.get(str(v or "").strip(), "all")


def _day_type_applies_ui(window_day_type: Any, date_obj: datetime) -> bool:
    wd = _normalize_day_type_ui(window_day_type)
    if wd == "all":
        return True
    dow = _DOW_KEYS[date_obj.weekday()]
    if wd == dow:
        return True
    is_weekend = date_obj.weekday() >= 5
    if wd == "weekday":
        return not is_weekend
    if wd == "weekend":
        return is_weekend
    return False


def _validate_and_build_windows_df(win_df: pd.DataFrame) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Validate edited windows df and return (windows_list, errors_list).
    windows_list is normalized to canonical keys (all/mon..sun/weekday/weekend).
    """
    errors: List[str] = []
    windows: List[Dict[str, Any]] = []
    if win_df is None or not isinstance(win_df, pd.DataFrame) or win_df.empty:
        return windows, errors

    for idx, r in win_df.iterrows():
        raw_day = r.get("day_type", "all")
        day_type = _normalize_day_type_ui(raw_day)
        start = _normalize_time_str_ui(r.get("start", "00:00"))
        end = _normalize_time_str_ui(r.get("end", "24:00"))

        # Skip fully empty rows (Streamlit may keep a blank row for dynamic editor)
        if (not str(raw_day).strip()) and (not start) and (not end) and (str(r.get("min_staff", "")).strip() == ""):
            continue

        if day_type not in set(_DAY_TYPE_OPTIONS_BASE):
            errors.append(f"第 {idx + 1} 行：day_type 必须从下拉选项中选择。")

        if not start or not end:
            errors.append(f"第 {idx + 1} 行：start/end 不能为空（会自动忽略 None/空值行）。")
            continue

        sm = _parse_time_to_minutes(start)
        em = _parse_time_to_minutes(end)
        if sm is None or em is None:
            errors.append(f"第 {idx + 1} 行：start/end 时间格式不合法。")
            continue
        sm = int(sm)
        em = int(em)
        if sm >= 1440:
            errors.append(f"第 {idx + 1} 行：start 不能为 24:00。")
        if (sm % 30) != 0:
            errors.append(f"第 {idx + 1} 行：start 需为 30 分钟刻度（如 07:00 / 07:30）。")
        if em != 1440 and (em % 30) != 0:
            errors.append(f"第 {idx + 1} 行：end 需为 30 分钟刻度（如 16:00 / 16:30 / 24:00）。")
        if em <= sm:
            errors.append(f"第 {idx + 1} 行：end 必须大于 start。")

        try:
            min_staff = int(r.get("min_staff", 1))
        except Exception:
            min_staff = None
        if min_staff is None or min_staff < 0:
            errors.append(f"第 {idx + 1} 行：min_staff 必须是 ≥ 0 的整数。")
            min_staff = 0

        windows.append({"day_type": day_type, "start": start, "end": end, "min_staff": int(min_staff)})

    return windows, errors

def _availability_cell_css(cell):
    """Extract CSS style string from availability cell (background + text color)."""
    if isinstance(cell, dict):
        bg_hex6 = _excel_rgb_to_hex6(cell.get("color"))
        font_hex6 = _excel_rgb_to_hex6(cell.get("font_color"))

        styles = []

        bg_css = _excel_rgb_to_css(cell.get("color"))
        if bg_css:
            styles.append(bg_css)

        # Prefer explicit font color; if missing, auto based on bg.
        chosen_font = font_hex6
        if bg_hex6 and (not chosen_font):
            chosen_font = _auto_text_hex_from_bg(bg_hex6)
        # If explicit font equals bg (unreadable), fall back to auto.
        if bg_hex6 and chosen_font and chosen_font == bg_hex6:
            chosen_font = _auto_text_hex_from_bg(bg_hex6)

        if chosen_font:
            styles.append(f"color: #{chosen_font};")

        return " ".join(styles)
    return ""

# Converts availability dict (with nested value/color) to a simple df for display
def flatten_availability_for_display(availability):
    records = []
    for date, emps in availability.items():
        row = {'Date': date}
        for emp, cell in emps.items():
            val = cell['value']
            row[emp] = val
        records.append(row)
    return pd.DataFrame(records)

# If you want to visualize color in Streamlit, apply background formatting
def get_cell_styles(availability):
    import pandas as pd
    # Returns a DataFrame of styles (you could use .style.apply in display, but Streamlit might need custom rendering)
    styles = []
    for date, emps in availability.items():
        row = {}
        for emp, cell in emps.items():
            color = cell.get('color')
            row[emp] = f'background-color: #{color[2:]}' if color else ''
        styles.append(row)
    return pd.DataFrame(styles)


def merge_edited_df_with_color(edited_df, orig_availability):
    """
    Given a DataFrame of "Date", Emp1, Emp2... (all cells are pure values, no color shown),
    and the original nested-availability dict with color+value,
    return a new nested dict with new values but colors preserved/retained.
    """
    if edited_df is None or edited_df.empty:
        return orig_availability or {}

    result = {}
    if "Date" in edited_df.columns:
        edited_df_nodate = edited_df.set_index("Date")
    else:
        edited_df_nodate = edited_df.copy()

    for date, row in edited_df_nodate.iterrows():
        date_key = str(date)
        result[date_key] = {}
        for emp, new_val in row.items():
            if isinstance(new_val, float) and pd.isna(new_val):
                new_val = ""
            if new_val is None:
                new_val = ""
            if not isinstance(new_val, str):
                new_val = str(new_val)
            new_val = new_val.strip()
            orig_cell = orig_availability.get(date_key, {}).get(emp, {})
            cell_color = orig_cell.get("color")
            cell_font = orig_cell.get("font_color")
            result[date_key][emp] = {"value": new_val, "color": cell_color, "font_color": cell_font}
    return result


def availability_to_dataframe():
    """Converts the availability dictionary to a Pandas DataFrame for editing."""
    availability_dict = st.session_state.availability
    if not availability_dict:
        return pd.DataFrame()

    df = pd.DataFrame(availability_dict).T

    # --- NEW: Use imported order as the employee column order if available ---
    if "imported_col_order" in st.session_state:
        col_order = [col for col in st.session_state.imported_col_order if col]  # remove any blank/None
    else:
        col_order = [emp.name for emp in st.session_state.employees]
    # Ensure all imported columns are present in df, even if empty
    for emp in col_order:
        if emp not in df.columns:
            df[emp] = ["" for _ in range(len(df))]
    df = df[col_order]  # This sets the display order

    # Convert availability cells to editable display values (value-only)
    return _df_elementwise(df, _availability_cell_value)

def availability_to_color_css_dataframe():
    """Build a DataFrame of CSS strings (background colors) aligned to availability_to_dataframe()."""
    availability_dict = st.session_state.availability
    if not availability_dict:
        return pd.DataFrame()
    df = pd.DataFrame(availability_dict).T
    # Match the same column order logic
    if "imported_col_order" in st.session_state:
        col_order = [col for col in st.session_state.imported_col_order if col]
    else:
        col_order = [emp.name for emp in st.session_state.employees]
    for emp in col_order:
        if emp not in df.columns:
            df[emp] = ["" for _ in range(len(df))]
    df = df[col_order]
    return _df_elementwise(df, _availability_cell_css)

def convert_availability_dates_to_str(availability):
    """
    Recursively convert all datetime-like keys/values in the availability dict to strings.
    This prevents `TypeError: Object of type datetime is not JSON serializable` when exporting.
    """
    import datetime as dt

    def conv(obj):
        if isinstance(obj, dict):
            return {conv_key(k): conv(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [conv(v) for v in obj]
        if isinstance(obj, tuple):
            return [conv(v) for v in obj]
        if isinstance(obj, set):
            return [conv(v) for v in obj]

        # pandas Timestamp / NaT
        try:
            if isinstance(obj, pd.Timestamp):
                if pd.isna(obj):
                    return None
                return obj.to_pydatetime().isoformat()
        except Exception:
            pass

        # python datetime/date/time
        if isinstance(obj, dt.datetime):
            return obj.isoformat()
        if isinstance(obj, dt.date):
            return obj.strftime("%Y-%m-%d")
        if isinstance(obj, dt.time):
            return obj.isoformat()

        # numpy scalar support (optional)
        try:
            import numpy as np  # type: ignore
            if isinstance(obj, (np.integer, np.floating, np.bool_)):
                return obj.item()
            if isinstance(obj, np.datetime64):
                # Convert to ISO string
                return str(obj)
        except Exception:
            pass

        return obj

    def conv_key(k):
        # Most keys represent a date row; normalize to YYYY-MM-DD.
        try:
            if isinstance(k, pd.Timestamp):
                if pd.isna(k):
                    return "NaT"
                return k.to_pydatetime().strftime("%Y-%m-%d")
        except Exception:
            pass

        if isinstance(k, dt.datetime) or isinstance(k, dt.date):
            return k.strftime("%Y-%m-%d")
        return str(k)
    
    return conv(availability)


def dataframe_to_availability(edited_df):
    """Converts the edited DataFrame back to the availability dictionary format."""
    # Keep original structure (value+color) if present; update value only.
    orig = st.session_state.get("availability") or {}
    new_avail = {}
    for date_key, row in edited_df.iterrows():
        d = str(date_key)
        new_avail[d] = {}
        for emp in edited_df.columns:
            new_val = row[emp]
            if isinstance(new_val, float) and pd.isna(new_val):
                new_val = ""
            if new_val is None:
                new_val = ""
            if not isinstance(new_val, str):
                new_val = str(new_val)
            new_val = new_val.strip()

            orig_cell = orig.get(d, {}).get(emp)
            color = orig_cell.get("color") if isinstance(orig_cell, dict) else None
            font_color = orig_cell.get("font_color") if isinstance(orig_cell, dict) else None
            new_avail[d][emp] = {"value": new_val, "color": color, "font_color": font_color}

    st.session_state.availability = new_avail


# --- Initialization ---
initialize_session_state()

availability_df = availability_utils.availability_to_dataframe()
availability_color_css_df = availability_utils.availability_to_color_css_dataframe()

# --- Sidebar UI ---
st.sidebar.title("🗓️ Schedule Maker")
st.sidebar.write("Manage employee availability and generate schedules.")

st.sidebar.header("Main Shift Employee Import")
main_shift_file = st.sidebar.file_uploader("Upload Main Shift Excel", type=["xlsx"])

if main_shift_file:
    # Pull current employee names for live comparison
    current_employee_names = [e.name for e in st.session_state.employees]
    names_detected, names_missing, imported_availability = import_employees_from_main_excel(
        main_shift_file,
        current_employee_names,
        None
    )
    st.session_state.availability = imported_availability


    df = pd.read_excel(main_shift_file, header=None)
    imported_col_order = []
    for name in df.iloc[0]:
        if pd.notna(name):
            str_name = str(name).strip()
            if str_name and str_name not in imported_col_order:
                imported_col_order.append(str_name)
    st.session_state.imported_col_order = imported_col_order
    
    sheet_dates = [str(x).strip() for x in df.iloc[1:, 0] if pd.notna(x) and str(x).strip()]
    st.session_state.imported_sheet_dates = sheet_dates

    if sheet_dates:
        st.sidebar.info(f"First date detected: {sheet_dates[0]}")
        st.sidebar.info(f"Last date detected: {sheet_dates[-1]}")
    else:
        st.sidebar.warning("No dates detected in imported sheet.")

    # Identify employees present in the system but NOT in the imported sheet
    extra_employees = [e.name for e in st.session_state.employees if e.name not in st.session_state.imported_col_order]
    st.session_state.extra_employees = extra_employees

    if "extra_employees" in st.session_state and st.session_state.extra_employees:
        st.sidebar.subheader("Employees not in imported sheet")
        for extra_name in st.session_state.extra_employees:
            with st.sidebar.form(key=f"remove_{extra_name}_form"):
                st.write(f"Employee '{extra_name}' found in system but NOT in imported main sheet.")
                remove = st.form_submit_button(f"Remove '{extra_name}'")
                if remove:
                    delete_employee(extra_name)
                    st.session_state.extra_employees.remove(extra_name)
                    st.toast(f"🗑️ Employee '{extra_name}' removed from system (not in latest main sheet import).")
                    st.session_state.initialized = False
                    st.rerun()


    st.sidebar.write("Detected Employees from Sheet:")
    st.sidebar.write(", ".join(names_detected))
    if names_missing:
        st.sidebar.warning(f"Missing employees in system: {', '.join(names_missing)}")
        # Prompt user to input all details for each new employee
        for name in names_missing:
            with st.sidebar.form(key=f"add_{name}_form"):
                st.write(f"Add employee: {name}")
                role = st.selectbox(f"Role for {name}", list(ROLE_RULES.keys()))
                shift = st.text_input(f"Shift for {name} (e.g., 10-19)")
                start, end = None, None
                if "-" in shift:
                    start, end = shift.split("-", 1)
                submit = st.form_submit_button("Add Employee")
                if submit:
                    add_employee(name, role, start, end)
                    st.success(f"Employee {name} added.")

st.sidebar.header("Actions")
if st.sidebar.button("Generate Schedule"):
    with st.spinner("Generating schedule..."):
        warnings = generate_schedule(st.session_state.availability, st.session_state.start_date, export_to_excel=False)
        st.session_state.warnings = warnings
        st.session_state.generated_schedule = pd.DataFrame(get_last_generated_schedule())
        if not warnings:
            st.toast("✅ Schedule generated successfully!")
        else:
            st.toast(f"⚠️ Schedule generated with {len(warnings)} warnings.")


if st.sidebar.button("Save All Changes", type="primary"):
    with st.spinner("Saving data..."):
        # Get edited DataFrame from your DataEditor variable (replace displaydf/editeddf as needed)
        # For example, edited_df = st.session_state['edited_availability_df'] if you've stored it there
        # If you use a local variable, just reference it directly

        # Step 1: Merge new text values with old color info
        # Replace 'edited_df' with whatever you use for the currently edited table
        merged_availability = availability_utils.merge_edited_df_with_color(
            availability_df, st.session_state.availability
        )

        # Step 2: Save to session state and file
        st.session_state.availability = merged_availability
        save_data(st.session_state.availability)
        save_employees(st.session_state.employees)
        st.toast("💾 All changes saved to server files!")

# Explicit save for imported availability (clarify persistence for users)
if st.sidebar.button("保存当前总表到 Firebase（仅 availability）", type="secondary"):
    try:
        with st.spinner("Saving availability to Firebase..."):
            save_data(st.session_state.availability)
        st.toast("✅ 已保存当前 availability 到 Firebase。")
    except Exception as e:
        st.sidebar.error(f"保存失败：{e}")


if st.sidebar.button("Clear All Availability"):
    st.session_state.availability = clear_availability(st.session_state.start_date, st.session_state.employees)
    st.toast("🗑️ Availability cleared and reset.")
    st.rerun()

# --- Main Page UI ---
st.title("Employee Availability Editor")

employee_tab, group_rules_tab, availability_tab, schedule_tab, import_export_tab = st.tabs(
    ["员工管理", "小组规则", "Availability（未完成）", "生成排班（未完成）", "导入/导出"]
)
with employee_tab:
    render_employee_management_tab(
        refresh_master_data=refresh_master_data,
        save_employees_to_storage_only=save_employees_to_storage_only,
        add_employee=add_employee,
        edit_employee=edit_employee,
        delete_employee=delete_employee,
        save_group_rules=save_group_rules,
        ROLE_RULES=ROLE_RULES,
        GROUP_RULES=GROUP_RULES,
    )
with group_rules_tab:
    render_group_rules_tab(
        group_rules_enabled=GROUP_RULES_ENABLED,
        load_group_rules=load_group_rules,
        save_group_rules=save_group_rules,
        group_rules_default=GROUP_RULES,
        alt=alt,
        fm=fm,
    )
with availability_tab:
    render_availability_tab(
        role_rules=ROLE_RULES,
        components=components,
        import_from_google_form=import_from_google_form,
        export_availability_to_excel=export_availability_to_excel,
        generated_schedule=st.session_state.generated_schedule,
    )


with schedule_tab:
    render_schedule_tab()

with import_export_tab:
    render_import_export_tab(
        role_rules=ROLE_RULES,
        employees=st.session_state.employees,
    )
