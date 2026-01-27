import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


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
            if not day_type_applies_ui(day_type, d):
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


def build_group_coverage_heatmap_df(all_checked_df: pd.DataFrame, step_minutes: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
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

    time_slots = [t for t in time_options(step_minutes) if t != "24:00"]
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


def heatmap_style_from_rate(v: Any) -> str:
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


def build_week_bins_from_dates(date_keys: List[str]) -> List[Dict[str, Any]]:
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


def build_week_grid_df(
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
    time_slots = [t for t in time_options(step_minutes) if t != "24:00"]

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


def extract_date_time_from_obj(obj: Any) -> Optional[Tuple[str, str]]:
    """
    Best-effort extraction of (date, time) from Streamlit chart selection payloads.
    """
    if obj is None:
        return None
    if isinstance(obj, dict):
        if "date" in obj and "time" in obj:
            return (str(obj["date"]), str(obj["time"]))
        for v in obj.values():
            got = extract_date_time_from_obj(v)
            if got:
                return got
    if isinstance(obj, list):
        for it in obj:
            got = extract_date_time_from_obj(it)
            if got:
                return got
    # some objects expose .selection
    try:
        sel = getattr(obj, "selection", None)
        got = extract_date_time_from_obj(sel)
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
                if not day_type_applies_ui(day_type, date_obj):
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


def build_cell_member_detail_df(
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
DAY_TYPE_OPTIONS_BASE = ["all"] + _DOW_KEYS + ["weekday", "weekend"]


def time_options(step_minutes: int = 30) -> List[str]:
    opts: List[str] = []
    m = 0
    while m < 24 * 60:
        h = m // 60
        mm = m % 60
        opts.append(f"{h:02d}:{mm:02d}")
        m += step_minutes
    opts.append("24:00")
    return opts


TIME_OPTIONS_BASE = time_options(30)


def _format_minutes_to_hhmm(m: int) -> str:
    if m == 1440:
        return "24:00"
    h = m // 60
    mm = m % 60
    return f"{h:02d}:{mm:02d}"


def _normalize_time_str_ui(v: Any) -> str:
    if v is None:
        return ""
    s = str(v).strip()
    if not s:
        return ""
    # Accept "0900" or "900"
    if s.isdigit():
        if len(s) in (3, 4):
            # "900" -> "09:00"
            if len(s) == 3:
                s = "0" + s
            return f"{s[:2]}:{s[2:]}"
        if len(s) <= 2:
            return f"{int(s):02d}:00"
    # Accept "9:0" -> "09:00"
    if ":" in s:
        try:
            hh, mm = s.split(":", 1)
            return f"{int(hh):02d}:{int(mm):02d}"
        except Exception:
            return s
    return s


def normalize_windows_df_for_editor(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    if df is None or df.empty:
        return pd.DataFrame(), 0
    out = df.copy()
    dropped_bad = 0
    if "day_type" not in out.columns:
        out["day_type"] = "all"
    if "start" not in out.columns:
        out["start"] = "00:00"
    if "end" not in out.columns:
        out["end"] = "24:00"
    if "min_staff" not in out.columns:
        out["min_staff"] = 1

    # Normalize values
    out["day_type"] = out["day_type"].apply(_normalize_day_type_ui)
    out["start"] = out["start"].apply(_normalize_time_str_ui)
    out["end"] = out["end"].apply(_normalize_time_str_ui)
    out["min_staff"] = pd.to_numeric(out["min_staff"], errors="coerce").fillna(1).astype(int)

    # Drop invalid rows (empty start/end)
    def _ok(r):
        return bool(str(r.get("start", "")).strip()) and bool(str(r.get("end", "")).strip())

    mask = out.apply(_ok, axis=1).astype(bool)
    dropped_bad = int((~mask).sum())
    out = out[mask]
    out = out.reset_index(drop=True)
    return out, dropped_bad


def _normalize_day_type_ui(v: Any) -> str:
    s = str(v or "").strip().lower()
    if not s:
        return "all"
    if s in _DOW_KEYS:
        return s
    if s in ("weekday", "weekend", "all"):
        return s
    if s in ("mon", "monday"):
        return "mon"
    if s in ("tue", "tues", "tuesday"):
        return "tue"
    if s in ("wed", "weds", "wednesday"):
        return "wed"
    if s in ("thu", "thur", "thurs", "thursday"):
        return "thu"
    if s in ("fri", "friday"):
        return "fri"
    if s in ("sat", "saturday"):
        return "sat"
    if s in ("sun", "sunday"):
        return "sun"
    return "all"


def day_type_applies_ui(window_day_type: Any, date_obj: datetime) -> bool:
    day_type = _normalize_day_type_ui(window_day_type)
    if day_type == "all":
        return True
    if day_type == "weekday":
        return date_obj.weekday() <= 4
    if day_type == "weekend":
        return date_obj.weekday() >= 5
    # mon..sun
    if day_type in _DOW_KEYS:
        return date_obj.weekday() == _DOW_KEYS.index(day_type)
    return True


def validate_and_build_windows_df(win_df: pd.DataFrame) -> Tuple[List[Dict[str, Any]], List[str]]:
    if win_df is None or win_df.empty:
        return [], ["规则段不能为空。"]
    errors: List[str] = []
    windows: List[Dict[str, Any]] = []
    for idx, r in win_df.iterrows():
        day_type = _normalize_day_type_ui(r.get("day_type"))
        start = _normalize_time_str_ui(r.get("start"))
        end = _normalize_time_str_ui(r.get("end"))
        if not start or not end:
            errors.append(f"第 {idx + 1} 行：start/end 不能为空。")
            continue
        sm = _parse_time_to_minutes(start)
        em = _parse_time_to_minutes(end)
        if sm is None or em is None:
            errors.append(f"第 {idx + 1} 行：start/end 格式无效。")
            continue
        if end == "24:00" and em != 1440:
            errors.append(f"第 {idx + 1} 行：end 只能为 24:00 或合法时间。")
        if start == "24:00":
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
