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
    export_availability_to_excel,
    clear_availability,
    sync_availability,
    get_last_generated_schedule,
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
def initialize_session_state():
    """Load initial data into the session state."""
    if not st.session_state.get('initialized'):
        st.session_state.start_date = datetime(2025, 3, 17)
        st.session_state.employees = load_employees()
        
        # Sync employees with availability data
        sync_availability() 
        
        st.session_state.availability = load_data()
        st.session_state.group_rules = load_group_rules()
        
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
    cur = min_d
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
    current_group_name: str,
    group_rules: Dict[str, Any],
    current_priority: int,
) -> Optional[str]:
    """
    Reserve interface for 'super employee' / cross-group priorities.

    If a member belongs to another group with higher priority, and that group's
    requirement windows cover this date/time slot, we mark the member as
    conflicted by that group (meaning: their time should be considered unavailable
    to the current group).
    """
    try:
        groups = (group_rules or {}).get("groups", []) or []
    except Exception:
        groups = []
    for g in groups:
        try:
            gname = str(g.get("name") or "").strip()
            if not gname or gname == current_group_name:
                continue
            gprio = int(g.get("priority") or 0)
            if gprio <= int(current_priority or 0):
                continue
            members = g.get("members") or []
            if member not in members:
                continue
            windows = g.get("requirements_windows") or []
            for w in windows:
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
    current_group_name = str(group.get("name") or "").strip()
    current_priority = int(group.get("priority") or 0)

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
            current_group_name=current_group_name,
            group_rules=group_rules,
            current_priority=current_priority,
        )

        # Leave-like raw values first (AL/SL/...)
        if _is_leave_like_raw(raw_s) and not intervals:
            status = "请假"
            detail = f"{raw_s}" if raw_s else ""
        # Then conflict (higher priority group) overrides "到岗"
        elif conflict_group:
            status = "无优先级"
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
    result = {}
    edited_df_nodate = edited_df.set_index("Date")
    for date, row in edited_df_nodate.iterrows():
        result[date] = {}
        for emp in row.index:
            new_val = row[emp]
            orig_cell = orig_availability.get(date, {}).get(emp, {})
            cell_color = orig_cell.get("color")
            cell_font = orig_cell.get("font_color")
            result[date][emp] = {"value": new_val, "color": cell_color, "font_color": cell_font}
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
availability_df = availability_to_dataframe()
availability_color_css_df = availability_to_color_css_dataframe()

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
        merged_availability = merge_edited_df_with_color(availability_df, st.session_state.availability)

        # Step 2: Save to session state and file
        st.session_state.availability = merged_availability
        save_data(st.session_state.availability)
        save_employees()
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

st.sidebar.header("Data Import")
google_form_upload = st.sidebar.file_uploader("Import from Google Form (Excel)", type=["xlsx"])
if google_form_upload:
    try:
        with st.spinner("Importing..."):
            result = import_from_google_form(google_form_upload)
            st.toast(f"✅ {result}")
            # Reload data after import
            st.session_state.initialized = False
            st.rerun()
    except Exception as e:
        st.sidebar.error(f"Import failed: {e}")


st.sidebar.header("Data Export")
# Export Availability
avail_export_data = export_availability_to_excel(st.session_state.availability)
st.sidebar.download_button(
    label="Export Availability to Excel",
    data=avail_export_data,
    file_name="availability_export.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

# Export generated schedule
if st.session_state.generated_schedule is not None and not st.session_state.generated_schedule.empty:
    schedule_export_data = st.session_state.generated_schedule.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button(
        label="Export Schedule to CSV",
        data=schedule_export_data,
        file_name="generated_schedule.csv",
        mime="text/csv",
    )
    
st.sidebar.header("Data Persistence")
st.sidebar.info("To permanently save changes, download the data files and commit them to your Git repository.")

# Download core data files
employees_json = json.dumps([emp.__dict__ for emp in st.session_state.employees], indent=4)
st.sidebar.download_button("Download employees.json", employees_json, "employees.json")

availability_serializable = convert_availability_dates_to_str(st.session_state.availability)
availability_json = json.dumps(availability_serializable, ensure_ascii=False, indent=4)

st.sidebar.download_button("Download availability.json", availability_json, "availability.json")

# Download group rules
group_rules_json = json.dumps(st.session_state.get("group_rules") or GROUP_RULES, ensure_ascii=False, indent=2)
st.sidebar.download_button("Download group_rules.json", group_rules_json, "group_rules.json")


# --- Main Page UI ---
st.title("Employee Availability Editor")

# --- Employee Management Section ---
with st.expander("Manage Employees"):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Add New Employee")
        with st.form("add_employee_form", clear_on_submit=True):
            add_name = st.text_input("Name")
            add_role = st.selectbox("Role", list(ROLE_RULES.keys()))
            add_start_time = st.text_input("Start Time (for fixed time roles)", "10-19")
            add_end_time = ""
            if '-' in add_start_time:
                add_start_time, add_end_time = add_start_time.split('-')
            
            if st.form_submit_button("Add Employee"):
                add_employee(add_name, add_role, start_time=add_start_time, end_time=add_end_time)
                st.toast(f"✅ Employee '{add_name}' added.")
                st.session_state.initialized = False
                st.rerun()
                
    with col2:
        st.subheader("Edit or Delete Employee")
        employees_list = st.session_state.employees
        selected_employee_name = st.selectbox("Select Employee to Edit/Delete", [e.name for e in employees_list])
        
        if selected_employee_name:
            emp_to_edit = next((e for e in employees_list if e.name == selected_employee_name), None)
            
            with st.form("edit_employee_form"):
                st.write(f"Editing: **{emp_to_edit.name}**")
                new_name = st.text_input("New Name", value=emp_to_edit.name)
                new_role = st.selectbox("New Role", list(ROLE_RULES.keys()), index=list(ROLE_RULES.keys()).index(emp_to_edit.employee_type))
                new_shift = st.text_input("New Shift (e.g., 10-19)", value=f"{emp_to_edit.start_time}-{emp_to_edit.end_time}" if emp_to_edit.start_time else "")
                
                submitted = st.form_submit_button("Update Employee")
                if submitted:
                    start_time, end_time = (new_shift.split('-') if '-' in new_shift else (None, None))
                    edit_employee(emp_to_edit.name, new_name, new_role, new_start_time=start_time, new_end_time=end_time)
                    st.toast(f"✅ Employee '{new_name}' updated.")
                    st.session_state.initialized = False
                    st.rerun()

            if st.button(f"Delete {selected_employee_name}", type="secondary"):
                delete_employee(selected_employee_name)
                st.toast(f"🗑️ Employee '{selected_employee_name}' deleted.")
                st.session_state.initialized = False
                st.rerun()


# --- Custom Group Rules (Team Rules) ---
with st.expander("自定义更表规则（小组）"):
    if not GROUP_RULES_ENABLED:
        st.warning("当前部署环境的 `scheduling_logic.py` 版本不包含小组规则功能（load_group_rules）。请确保已把最新代码部署/推送后再使用此功能。")
        st.stop()

    # --- Import group_rules.json (dry-run preview; does NOT write to Firebase unless you click save) ---
    st.markdown("**导入 group_rules.json（可选）**")
    st.caption("选择文件后只会在本次会话中解析与预览，不会自动写入 Firebase。需要你点击“应用/保存”按钮才会生效。")
    uploaded_group_rules = st.file_uploader(
        "选择一个 group_rules.json（或 Firebase 的备份文件）",
        type=["json"],
        key="group_rules_import_uploader",
    )
    if uploaded_group_rules is not None:
        try:
            raw_text = uploaded_group_rules.getvalue().decode("utf-8", errors="ignore")
            imported_obj = json.loads(raw_text)
            # Best-effort normalize using scheduling_logic internal helper if available
            try:
                from scheduling_logic import _normalize_group_rules  # type: ignore
                imported_obj = _normalize_group_rules(imported_obj)  # type: ignore[misc]
            except Exception:
                pass
            st.session_state["_imported_group_rules_preview"] = imported_obj
        except Exception as e:
            st.session_state.pop("_imported_group_rules_preview", None)
            st.error(f"导入失败：{e}")

    preview_obj = st.session_state.get("_imported_group_rules_preview")
    if isinstance(preview_obj, dict) and isinstance(preview_obj.get("groups", None), list):
        groups_preview = preview_obj.get("groups") or []
        st.success(f"已解析：{len(groups_preview)} 个小组。")
        if groups_preview:
            names = [g.get("name") for g in groups_preview if isinstance(g, dict) and g.get("name")]
            if names:
                st.caption("预览（前 12 个小组名）：" + "、".join([str(x) for x in names[:12]]))

        import_cols = st.columns([1, 1, 2])
        with import_cols[0]:
            if st.button("应用到当前会话", type="secondary", key="apply_imported_group_rules"):
                st.session_state.group_rules = preview_obj
                st.toast("已应用导入的小组规则到当前会话（未写入 Firebase）。")
                st.session_state.initialized = False
                st.rerun()
        with import_cols[1]:
            if st.button("应用并保存到 Firebase", type="primary", key="apply_and_save_imported_group_rules"):
                st.session_state.group_rules = preview_obj
                save_group_rules(st.session_state.group_rules)
                st.toast("✅ 已导入并保存到 Firebase。")
                st.session_state.initialized = False
                st.rerun()
        with import_cols[2]:
            st.caption("说明：保存时会进行 schema 规范化；无效规则段（如 start/end 为 None）不会写回。")

    def _reset_group_edit_widgets():
        """
        When switching the selected group, we must clear the edit widget keys.
        Otherwise Streamlit will reuse previous widget state and the UI appears "not refreshed".
        """
        # Clear any previously created per-group edit widgets
        for k in list(st.session_state.keys()):
            if k.startswith("edit_group_ui__") or k.startswith("confirm_delete_group_ui__"):
                del st.session_state[k]

    # Refresh from Firebase
    cols = st.columns([1, 1, 2])
    with cols[0]:
        if st.button("🔄 从Firebase刷新小组规则"):
            st.session_state.group_rules = load_group_rules()
            st.toast("已刷新小组规则。")
    with cols[1]:
        if st.button("💾 保存小组规则到Firebase", type="primary"):
            save_group_rules(st.session_state.group_rules)
            st.toast("小组规则已保存到 Firebase。")

    # Diagnostics (helps when different deployments / Firebase envs appear inconsistent)
    with st.expander("诊断：Firebase 读取到的小组规则（只读）", expanded=False):
        try:
            proj = None
            try:
                proj = st.secrets.get("firebase", {}).get("service_account", {}).get("project_id")
            except Exception:
                proj = None
            if proj:
                st.caption(f"Firebase project_id: {proj}")

            raw = fm.get_data("group_rules")
            if raw is None:
                st.warning("fm.get_data('group_rules') 返回 None（Firebase 中该路径可能为空/无权限/连接异常）。")
            else:
                st.caption(f"fm.get_data('group_rules') 类型：{type(raw).__name__}")
                if isinstance(raw, dict):
                    st.caption(f"keys: {list(raw.keys())}")
                    st.caption(f"updated_at: {raw.get('updated_at')}")
                    gs = raw.get("groups") or []
                    st.caption(f"groups 数量: {len(gs) if isinstance(gs, list) else 'N/A'}")
                st.json(raw)

            # Storage backup check
            try:
                backup = None
                if hasattr(fm, "get_json_from_storage"):
                    backup = fm.get_json_from_storage("config/group_rules.json")
                if backup is None:
                    st.warning("Storage 备份读取结果：None（可能 bucket 名称不匹配或无权限）。")
                else:
                    st.success("Storage 备份读取成功：config/group_rules.json")
                    if isinstance(backup, dict):
                        st.caption(f"backup keys: {list(backup.keys())}")
                        st.caption(f"backup updated_at: {backup.get('updated_at')}")
                        bg = backup.get('groups') or []
                        st.caption(f"backup groups 数量: {len(bg) if isinstance(bg, list) else 'N/A'}")
                    st.json(backup)
            except Exception as e:
                st.error(f"Storage 备份读取异常：{e}")
        except Exception as e:
            st.error(f"诊断读取失败：{e}")

    group_rules = st.session_state.get("group_rules") or GROUP_RULES
    groups = group_rules.get("groups", [])

    # --- Validate group coverage based on imported "total sheet" (availability) ---
    st.subheader("验证小组需求（基于已导入的总表）")
    if not groups:
        st.info("暂无小组可验证。请先创建并保存小组规则。")
    elif not st.session_state.get("availability"):
        st.warning("当前还没有导入总表（availability）。请先在侧边栏导入主更表。")
    else:
        name_to_group2 = {g.get("name"): g for g in groups if g.get("name")}
        sel_name = st.selectbox("选择要验证的小组", options=list(name_to_group2.keys()), key="validate_group_name")
        # UI simplified: week selector + clickable grid + always-on detail panel
        gsel = name_to_group2.get(sel_name)

        # Persist last validation result in session_state so widget interactions won't wipe the UI.
        if st.button("开始验证", type="primary", key="run_validate_group"):
            if not gsel:
                st.error("未选择有效小组。")
            else:
                with st.spinner("正在按 60 分钟时段校验覆盖..."):
                    summary_df, deficits_df, all_checked_df = validate_group_coverage_from_availability(
                        st.session_state.availability, gsel, step_minutes=60
                    )
                # Build week bins from imported dates
                date_keys = []
                try:
                    date_keys = sorted(list(set([str(x) for x in all_checked_df.get("date", []).tolist()])))
                except Exception:
                    date_keys = sorted(list(set([str(x) for x in (st.session_state.availability or {}).keys()])))
                week_bins = _build_week_bins_from_dates(date_keys)
                # Default to first bin
                if week_bins:
                    st.session_state["validate_week_bin_idx"] = 0
                st.session_state["_validate_group_last_result"] = {
                    "group_name": sel_name,
                    "step_minutes": 60,
                    "summary_df": summary_df,
                    "deficits_df": deficits_df,
                    "all_checked_df": all_checked_df,
                    "week_bins": week_bins,
                    "computed_at": datetime.now().isoformat(timespec="seconds"),
                }

        # Render from last result (if it matches current selected group)
        last = st.session_state.get("_validate_group_last_result")
        if not gsel:
            st.info("请选择一个有效小组，然后点击“开始验证”。")
        elif not isinstance(last, dict) or last.get("group_name") != sel_name:
            st.info("请点击“开始验证”生成结果后，再进行热力网格/单格明细查看。")
        else:
            summary_df = last.get("summary_df")
            deficits_df = last.get("deficits_df")
            all_checked_df = last.get("all_checked_df")
            step_minutes = int(last.get("step_minutes") or 30)
            week_bins = last.get("week_bins") or []

            # Defensive: ensure dataframes exist
            if not isinstance(all_checked_df, pd.DataFrame) or all_checked_df.empty:
                st.info("暂无可展示结果（可能规则段为空或导入日期为空）。")
            else:
                has_deficit = isinstance(deficits_df, pd.DataFrame) and (not deficits_df.empty)
                if has_deficit:
                    approx_hours = len(deficits_df) * (step_minutes / 60.0)
                    st.warning(
                        f"⚠️ 小组「{sel_name}」存在缺口时段（{step_minutes}min/格）：{len(deficits_df)} 条（约 {approx_hours:.1f} 小时）"
                    )
                else:
                    st.success(f"✅ 小组「{sel_name}」在当前总表日期范围内：所有规则段均满足（无缺口）。")

                if not week_bins:
                    st.info("无法生成周分段（日期解析失败或导入日期为空）。")
                else:
                    labels = [b.get("label") for b in week_bins]
                    idx = st.selectbox(
                        "选择时间范围（每 7 天一段）",
                        options=list(range(len(labels))),
                        format_func=lambda i: labels[i],
                        key="validate_week_bin_idx",
                    )
                    wb = week_bins[int(idx)]
                    week_start = datetime.fromisoformat(str(wb["start_date"])).date() if isinstance(wb.get("start_date"), str) else wb.get("start_date")
                    week_end = datetime.fromisoformat(str(wb["end_date"])).date() if isinstance(wb.get("end_date"), str) else wb.get("end_date")

                    grid_df = _build_week_grid_df(
                        all_checked_df=all_checked_df,
                        week_start=week_start,
                        week_end=week_end,
                        step_minutes=step_minutes,
                    )

                    # default selected cell within this week (first deficit -> first ok -> first)
                    def _pick_default_cell() -> Tuple[str, str]:
                        sub = grid_df[grid_df["status"] != "na"].copy()
                        if sub.empty:
                            return (week_start.isoformat(), "00:00")
                        d1 = sub[sub["status"] == "deficit"]
                        if not d1.empty:
                            r = d1.iloc[0]
                            return (str(r["date"]), str(r["time"]))
                        r = sub.iloc[0]
                        return (str(r["date"]), str(r["time"]))

                    cur_sel = st.session_state.get("_validate_selected_cell")
                    if not isinstance(cur_sel, dict):
                        cur_sel = {}
                    sel_date = str(cur_sel.get("date") or "")
                    sel_time = str(cur_sel.get("time") or "")
                    in_week = False
                    try:
                        sd = datetime.fromisoformat(sel_date).date()
                        in_week = (sd >= week_start) and (sd <= week_end)
                    except Exception:
                        in_week = False
                    if (not in_week) or (not sel_time):
                        d0, t0 = _pick_default_cell()
                        st.session_state["_validate_selected_cell"] = {"date": d0, "time": t0}
                        sel_date, sel_time = d0, t0

                    # Render clickable chart if Altair is available; otherwise fallback table.
                    st.caption("点击热力图任意一格，下方明细会自动切换到该格对应的日期+时间。")
                    if alt is not None and not grid_df.empty:
                        sel_param = alt.selection_point(fields=["date", "time"], on="click", empty=False, name="cell")
                        chart = (
                            alt.Chart(grid_df)
                            .mark_rect()
                            .encode(
                                x=alt.X("weekday:N", sort=["周一", "周二", "周三", "周四", "周五", "周六", "周日"], title=None),
                                y=alt.Y("time:N", sort=sorted(grid_df["time"].unique(), reverse=True), title=None),
                                color=alt.Color(
                                    "status:N",
                                    scale=alt.Scale(domain=["na", "ok", "deficit"], range=["#f3f4f6", "#d9f2d9", "#f8d7da"]),
                                    legend=None,
                                ),
                                tooltip=[
                                    alt.Tooltip("date:N", title="日期"),
                                    alt.Tooltip("weekday:N", title="周几"),
                                    alt.Tooltip("time:N", title="时间格"),
                                    alt.Tooltip("required:Q", title="required"),
                                    alt.Tooltip("staffed:Q", title="staffed"),
                                    alt.Tooltip("shortage:Q", title="shortage"),
                                ],
                            )
                            .add_params(sel_param)
                            .properties(height=720)
                        )
                        # attempt to get selection payload from Streamlit (version-dependent)
                        try:
                            evt = st.altair_chart(chart, use_container_width=True, on_select="rerun", key="validate_group_week_heatmap")
                            got = _extract_date_time_from_obj(evt)
                            if got:
                                st.session_state["_validate_selected_cell"] = {"date": got[0], "time": got[1]}
                                sel_date, sel_time = got[0], got[1]
                        except TypeError:
                            # older Streamlit: no on_select support
                            st.altair_chart(chart, use_container_width=True)
                    else:
                        # fallback
                        st.dataframe(
                            grid_df.pivot_table(index="time", columns="weekday", values="status", aggfunc="first"),
                            width="stretch",
                            height=720,
                        )
                        st.caption("提示：当前环境不支持点击热力图取值（Altair 或 on_select 不可用）。如需联动，请升级 Streamlit 或安装 Altair。")

                    # Detail panel (always visible)
                    st.subheader("明细")
                    st.caption(f"当前选择：{sel_date} {sel_time}（{step_minutes}min/格）")
                    row0 = None
                    try:
                        row0 = all_checked_df[(all_checked_df["date"] == sel_date) & (all_checked_df["time"] == sel_time)].head(1)
                    except Exception:
                        row0 = None
                    if isinstance(row0, pd.DataFrame) and (not row0.empty):
                        r0 = row0.iloc[0].to_dict()
                        st.caption(
                            f"该格校验结果：required={int(r0.get('required') or 0)} / staffed={int(r0.get('staffed') or 0)} / shortage={int(r0.get('shortage') or 0)}"
                        )
                    detail_df = _build_cell_member_detail_df(
                        availability=st.session_state.availability,
                        group=gsel,
                        group_rules=st.session_state.get("group_rules") or GROUP_RULES,
                        date_key=sel_date,
                        time_hhmm=sel_time,
                        step_minutes=step_minutes,
                    )
                    # Render detail with availability-style colors (DataFrame / Styler)
                    try:
                        cell_map = (st.session_state.availability or {}).get(sel_date, {}) or {}
                    except Exception:
                        cell_map = {}

                    def _status_css(s: Any) -> str:
                        v = str(s or "")
                        if v == "到岗":
                            return "background-color: #d9f2d9; color: #111827;"
                        if v == "未到岗":
                            return "background-color: #f8d7da; color: #111827;"
                        if v == "请假":
                            return "background-color: #fff3cd; color: #111827;"
                        if v == "无优先级":
                            return "background-color: #e2e8f0; color: #111827;"
                        return ""

                    def _raw_css_for_member(member: Any) -> str:
                        try:
                            cell = cell_map.get(str(member))
                        except Exception:
                            cell = None
                        return _availability_cell_css(cell)

                    if isinstance(detail_df, pd.DataFrame) and (not detail_df.empty) and ("成员" in detail_df.columns):
                        styler = detail_df.style
                        if "状态" in detail_df.columns:
                            styler = styler.applymap(_status_css, subset=["状态"])
                        if "明细" in detail_df.columns:
                            # colorize "明细" using the imported availability cell colors
                            styler = styler.apply(lambda r: [_raw_css_for_member(r.get("成员"))], axis=1, subset=["明细"])
                        st.dataframe(styler, width="stretch", height=320)
                    else:
                        st.dataframe(detail_df, width="stretch", height=320)

                    with st.expander("高级：查看缺口明细/按日期汇总", expanded=False):
                        if isinstance(summary_df, pd.DataFrame):
                            st.markdown("**按日期汇总**")
                            st.dataframe(summary_df, width="stretch", height=220)
                        if isinstance(deficits_df, pd.DataFrame):
                            st.markdown("**缺口明细（仅缺口）**")
                            st.dataframe(deficits_df, width="stretch", height=360)

        # Explicit save hint for imported availability
        st.caption("提示：侧边栏导入总表只会更新本次会话内的数据；如需写入 Firebase，请点击侧边栏的 “Save All Changes”。")

    st.caption("说明：小组规则用于校验排班是否满足“某时段最少需要多少人值更”。此处按“30 分钟时段”进行覆盖校验与可视化。")

    # Overview
    if groups:
        st.markdown("**概览（点击“成员”可展开查看）**")
        header_cols = st.columns([2, 4, 1, 1])
        header_cols[0].markdown("**名称**")
        header_cols[1].markdown("**成员**")
        header_cols[2].markdown("**成员数**")
        header_cols[3].markdown("**规则段数**")

        for g in groups:
            name = g.get("name")
            members = g.get("members", []) or []
            rules = g.get("requirements_windows", []) or []
            member_count = len(members)
            rules_count = len(rules)

            row_cols = st.columns([2, 4, 1, 1], vertical_alignment="center")
            with row_cols[0]:
                st.write(name)
            with row_cols[1]:
                with st.expander(f"成员（{member_count}）", expanded=False):
                    if members:
                        st.write("、".join(members))
                    else:
                        st.caption("（无成员）")
            row_cols[2].write(member_count)
            row_cols[3].write(rules_count)
    else:
        st.info("当前还没有任何小组。你可以在下面创建一个。")

    employee_names = [e.name for e in st.session_state.employees]

    st.subheader("创建新小组")
    with st.form("create_group_form", clear_on_submit=True):
        new_name = st.text_input("小组名称（必填）")
        new_desc = st.text_input("备注/说明（可选）")
        new_active = st.checkbox("启用", value=True)
        new_headcount = st.number_input("规划人数（可选）", min_value=0, value=0, step=1)
        new_members = st.multiselect("成员（从现有员工中选择）", options=employee_names, default=[])

        st.markdown("规则段（可多段）：每一段表示在该时间窗内，每个小时至少需要多少名成员在岗。")
        st.caption("day_type 建议：all=每天；mon..sun=周一..周日。start/end 为 30 分钟刻度，end 可选 24:00。")
        default_windows_df = pd.DataFrame([{"day_type": "all", "start": "00:00", "end": "24:00", "min_staff": 1}])
        # Include any existing values (if rerun keeps state) so editor won't blank them out,
        # but validation will still require selections to be from base options.
        day_opts = list(dict.fromkeys(_DAY_TYPE_OPTIONS_BASE + [str(x).strip().lower() for x in default_windows_df.get("day_type", []) if str(x).strip()]))
        start_opts = list(dict.fromkeys(_TIME_OPTIONS_BASE + [str(x).strip() for x in default_windows_df.get("start", []) if str(x).strip()]))
        end_opts = list(dict.fromkeys(_TIME_OPTIONS_BASE + [str(x).strip() for x in default_windows_df.get("end", []) if str(x).strip()]))
        win_df = st.data_editor(
            default_windows_df,
            num_rows="dynamic",
            width="stretch",
            hide_index=True,
            column_config={
                "day_type": st.column_config.SelectboxColumn("day_type", options=day_opts, required=True, help="all=每天；mon..sun=周一..周日（兼容 weekday/weekend）。"),
                "start": st.column_config.SelectboxColumn("start", options=start_opts, required=True, help="开始时间（30 分钟刻度）。"),
                "end": st.column_config.SelectboxColumn("end", options=end_opts, required=True, help="结束时间（30 分钟刻度；可选 24:00）。"),
                "min_staff": st.column_config.NumberColumn("min_staff", min_value=0, step=1, required=True, help="该时间窗内，每小时最少在岗人数。"),
            },
            key="new_group_windows",
        )

        submitted = st.form_submit_button("创建小组")
        if submitted:
            if not new_name.strip():
                st.error("小组名称不能为空。")
            else:
                # Prevent duplicate names
                if any(g.get("name") == new_name.strip() for g in groups):
                    st.error("已存在同名小组，请换一个名称。")
                else:
                    windows, win_errors = _validate_and_build_windows_df(win_df)
                    if win_errors:
                        st.error("规则段存在问题，请修正后再提交：\n\n- " + "\n- ".join(win_errors))
                        st.stop()

                    new_group = {
                        "id": uuid.uuid4().hex,
                        "name": new_name.strip(),
                        "description": new_desc.strip(),
                        "active": bool(new_active),
                        "headcount_planned": int(new_headcount) if new_headcount else None,
                        "members": list(new_members),
                        "requirements_windows": windows,
                    }
                    group_rules.setdefault("groups", []).append(new_group)
                    st.session_state.group_rules = group_rules
                    save_group_rules(st.session_state.group_rules)
                    st.toast(f"✅ 小组“{new_name.strip()}”已创建并保存。")
                    st.session_state.initialized = False
                    st.rerun()

    st.subheader("编辑/删除现有小组")
    if groups:
        name_to_group = {g.get("name"): g for g in groups if g.get("name")}

        # If we need to update the selected group programmatically (e.g. after rename/delete),
        # do it BEFORE the selectbox is instantiated to avoid StreamlitAPIException.
        pending_key = "_pending_selected_group_name"
        if pending_key in st.session_state:
            st.session_state["selected_group_name"] = st.session_state[pending_key]
            del st.session_state[pending_key]

        selected_group_name = st.selectbox(
            "选择小组",
            options=list(name_to_group.keys()),
            key="selected_group_name",
            on_change=_reset_group_edit_widgets,
        )
        g = name_to_group.get(selected_group_name)

        if g:
            gid = str(g.get("id") or g.get("name") or "unknown")
            key_prefix = f"edit_group_ui__{gid}__"
            edit_cols = st.columns([2, 2])
            with edit_cols[0]:
                edited_name = st.text_input("小组名称", value=g.get("name", ""), key=f"{key_prefix}name")
                edited_desc = st.text_input("备注/说明", value=g.get("description", ""), key=f"{key_prefix}desc")
                edited_active = st.checkbox("启用", value=bool(g.get("active", True)), key=f"{key_prefix}active")
                edited_headcount = st.number_input(
                    "规划人数（可选）",
                    min_value=0,
                    value=int(g.get("headcount_planned") or 0),
                    step=1,
                    key=f"{key_prefix}headcount",
                )
                edited_members = st.multiselect(
                    "成员（从现有员工中选择）",
                    options=employee_names,
                    default=[m for m in (g.get("members") or []) if m in employee_names],
                    key=f"{key_prefix}members",
                )

            with edit_cols[1]:
                windows_df = pd.DataFrame(g.get("requirements_windows") or [])
                if windows_df.empty:
                    windows_df = pd.DataFrame([{"day_type": "all", "start": "00:00", "end": "24:00", "min_staff": 1}])
                windows_df, dropped_bad = _normalize_windows_df_for_editor(windows_df)
                if dropped_bad:
                    st.caption(f"已自动忽略 {dropped_bad} 行无效规则段（start/end 为空或为 None）。保存后这些无效行也不会写回。")
                # Include any existing values so the editor can display legacy data,
                # but validator will still enforce base options on save.
                existing_day = [str(x).strip().lower() for x in windows_df.get("day_type", []) if str(x).strip()]
                existing_start = [str(x).strip() for x in windows_df.get("start", []) if str(x).strip()]
                existing_end = [str(x).strip() for x in windows_df.get("end", []) if str(x).strip()]
                day_opts = list(dict.fromkeys(_DAY_TYPE_OPTIONS_BASE + existing_day))
                start_opts = list(dict.fromkeys(_TIME_OPTIONS_BASE + existing_start))
                end_opts = list(dict.fromkeys(_TIME_OPTIONS_BASE + existing_end))
                edited_windows_df = st.data_editor(
                    windows_df,
                    num_rows="dynamic",
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "day_type": st.column_config.SelectboxColumn("day_type", options=day_opts, required=True, help="all=每天；mon..sun=周一..周日（兼容 weekday/weekend）。"),
                        "start": st.column_config.SelectboxColumn("start", options=start_opts, required=True, help="开始时间（30 分钟刻度）。"),
                        "end": st.column_config.SelectboxColumn("end", options=end_opts, required=True, help="结束时间（30 分钟刻度；可选 24:00）。"),
                        "min_staff": st.column_config.NumberColumn("min_staff", min_value=0, step=1, required=True, help="该时间窗内，每小时最少在岗人数。"),
                    },
                    key=f"{key_prefix}windows",
                )

            action_cols = st.columns([1, 1, 2])
            with action_cols[0]:
                if st.button("保存该小组修改", type="primary"):
                    # Validate rename collisions
                    new_name_norm = edited_name.strip()
                    if not new_name_norm:
                        st.error("小组名称不能为空。")
                    elif new_name_norm != g.get("name") and any(x.get("name") == new_name_norm for x in groups):
                        st.error("已存在同名小组，请换一个名称。")
                    else:
                        new_windows, win_errors = _validate_and_build_windows_df(edited_windows_df)
                        if win_errors:
                            st.error("规则段存在问题，请修正后再保存：\n\n- " + "\n- ".join(win_errors))
                            st.stop()

                        g["name"] = new_name_norm
                        g["description"] = edited_desc.strip()
                        g["active"] = bool(edited_active)
                        g["headcount_planned"] = int(edited_headcount) if edited_headcount else None
                        g["members"] = list(edited_members)
                        g["requirements_windows"] = new_windows

                        st.session_state.group_rules = group_rules
                        save_group_rules(st.session_state.group_rules)
                        st.toast("✅ 已保存小组修改到 Firebase。")
                        # If renamed, keep selection in sync
                        st.session_state["_pending_selected_group_name"] = new_name_norm
                        st.session_state.initialized = False
                        st.rerun()

            with action_cols[1]:
                confirm_delete = st.checkbox(
                    "确认删除",
                    value=False,
                    key=f"confirm_delete_group_ui__{gid}",
                )
                if st.button("删除该小组", type="secondary", disabled=not confirm_delete):
                    group_rules["groups"] = [x for x in group_rules.get("groups", []) if x.get("id") != g.get("id")]
                    st.session_state.group_rules = group_rules
                    save_group_rules(st.session_state.group_rules)
                    st.toast("🗑️ 小组已删除并保存到 Firebase。")
                    # After delete, reset selection to the first group (if any)
                    remaining = [x.get("name") for x in group_rules.get("groups", []) if x.get("name")]
                    if remaining:
                        st.session_state["_pending_selected_group_name"] = remaining[0]
                    elif "selected_group_name" in st.session_state:
                        del st.session_state["selected_group_name"]
                    st.session_state.initialized = False
                    st.rerun()


# --- Availability Editor ---
st.header("Availability Grid")
# Role filter
all_roles = ["All"] + list(ROLE_RULES.keys())
selected_role = st.selectbox("Filter by Role:", options=all_roles)

# Filter employees based on role
if selected_role == "All":
    filtered_employees = [emp.name for emp in st.session_state.employees]
else:
    filtered_employees = [emp.name for emp in st.session_state.employees if emp.employee_type == selected_role]


availability_df = availability_to_dataframe()

if not availability_df.empty:
    # Defensive: use imported_col_order if set, else employee order
    col_order = st.session_state.get("imported_col_order")
    if col_order is None:
        col_order = [emp.name for emp in st.session_state.employees]
    display_columns = [emp for emp in col_order if emp in availability_df.columns]
    display_df = availability_df[display_columns].copy()
    
    sheet_dates = st.session_state.get("imported_sheet_dates")
    
    if sheet_dates is None:
        st.warning("No imported sheet dates found.")
    else:
        # Always make displayed df match number of imported dates
        display_df = availability_df.copy()
        if len(display_df) > len(sheet_dates):
            display_df = display_df.iloc[:len(sheet_dates)]
        elif len(display_df) < len(sheet_dates):
            num_missing = len(sheet_dates) - len(display_df)
            blank = pd.DataFrame([[""] * len(display_df.columns)] * num_missing, columns=display_df.columns)
            display_df = pd.concat([display_df, blank], ignore_index=True)
        # Format sheet_dates to dd/mm/yyyy
        formatted_dates = [
            pd.to_datetime(date).strftime("%d/%m/%Y") for date in sheet_dates
        ]

        display_df.insert(0, "Date", formatted_dates)

        # --- Color preview (read-only) ---
        try:
            preview_values_df = display_df.drop(columns=["Date"]).copy()
            preview_css_df = availability_color_css_df.copy()
            # Align shapes defensively
            preview_css_df = preview_css_df.reindex_like(preview_values_df)
            preview_css_df = preview_css_df.fillna("")

            with st.expander("彩色预览（只读）", expanded=False):
                # Stats: how many cells actually have a color style
                total_cells = int(preview_css_df.size)
                colored_cells = int((preview_css_df.astype(str) != "").sum().sum()) if total_cells else 0
                st.caption(f"检测到有颜色的格子：{colored_cells}/{total_cells}")

                preview_rows = st.number_input("预览行数（避免太大导致卡顿）", min_value=10, max_value=200, value=60, step=10)
                pv = preview_values_df.head(int(preview_rows))
                pc = preview_css_df.head(int(preview_rows))
                styler = pv.style.apply(lambda _df: pc, axis=None)

                mode = st.radio(
                    "渲染方式",
                    options=["HTML（推荐，颜色更稳定）", "DataFrame（有时主题会吞掉颜色）"],
                    horizontal=True,
                )
                if mode.startswith("HTML"):
                    components.html(styler.to_html(), height=460, scrolling=True)
                else:
                    st.dataframe(styler, height=420, width="stretch")
        except Exception:
            # Preview should never break editing
            pass

        st.info("You can directly edit the cells below. Changes are saved when you click 'Save All Changes'.")
        edited_df = st.data_editor(display_df, height=600)

        # WARNING: Remove "Date" column before updating raw data!
        full_edited_df = availability_df.copy()
        edited_df_no_date = edited_df.drop(columns=["Date"])
        full_edited_df.update(edited_df_no_date)
        dataframe_to_availability(full_edited_df)
else:
    st.warning("No availability data found. Initialize or import data.")


# --- Display Generated Schedule ---
if st.session_state.generated_schedule is not None:
    st.header("Generated Schedule")
    
    # Display warnings if any
    if st.session_state.warnings:
        st.warning("The following issues were found during schedule generation:")
        for warning in st.session_state.warnings:
            st.write(f"- {warning}")
            
    st.dataframe(st.session_state.generated_schedule)

