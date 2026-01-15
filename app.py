import streamlit as st
import pandas as pd
from datetime import datetime
import json
import uuid
import streamlit.components.v1 as components
import re
from typing import Dict, List, Tuple, Any, Optional
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

# --- Main App ---
st.set_page_config(page_title="BP Schedule Maker", layout="wide")
st.title("Auto-Schedule Maker")



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
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return (summary_df, deficits_df).
    summary_df: per date total shortage-hours
    deficits_df: per date/hour shortage details
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

    rows = []
    for date_key in sorted(parsed.keys()):
        try:
            d = pd.to_datetime(date_key).to_pydatetime()
        except Exception:
            continue
        is_weekend = d.weekday() >= 5

        for w in windows:
            day_type = str(w.get("day_type") or "all").strip().lower()
            if day_type == "weekday" and is_weekend:
                continue
            if day_type == "weekend" and not is_weekend:
                continue
            start_s = str(w.get("start") or "00:00")
            end_s = str(w.get("end") or "24:00")
            min_staff = int(w.get("min_staff") or 0)
            ws, we = _time_window_to_minutes(start_s, end_s)
            if we <= ws:
                continue

            for hour in range(ws // 60, (we + 59) // 60):
                slot_s = hour * 60
                slot_e = min((hour + 1) * 60, 24 * 60)
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
                if shortage > 0:
                    rows.append(
                        {
                            "date": date_key,
                            "hour": f"{hour:02d}:00",
                            "required": min_staff,
                            "staffed": staffed,
                            "shortage": shortage,
                            "on_duty": "、".join(on),
                            "window": f"{day_type} {start_s}-{end_s}",
                        }
                    )

    deficits_df = pd.DataFrame(rows)
    if deficits_df.empty:
        summary_df = pd.DataFrame(columns=["date", "shortage_hours", "total_shortage"])
        return summary_df, deficits_df

    summary_df = (
        deficits_df.groupby("date")
        .agg(shortage_hours=("shortage", "count"), total_shortage=("shortage", "sum"))
        .reset_index()
        .sort_values(["total_shortage", "shortage_hours"], ascending=False)
    )
    return summary_df, deficits_df

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
        only_deficits = st.checkbox("只显示有缺口的小时", value=True, key="validate_only_deficits")
        if st.button("开始验证", type="primary", key="run_validate_group"):
            gsel = name_to_group2.get(sel_name)
            if not gsel:
                st.error("未选择有效小组。")
            else:
                with st.spinner("正在按小时校验覆盖..."):
                    summary_df, deficits_df = validate_group_coverage_from_availability(
                        st.session_state.availability, gsel
                    )
                if deficits_df.empty:
                    st.success(f"✅ 小组「{sel_name}」在当前总表日期范围内：所有规则段均满足（无缺口）。")
                else:
                    st.warning(f"⚠️ 小组「{sel_name}」存在缺口小时：{len(deficits_df)} 条")
                    st.dataframe(summary_df, width="stretch", height=220)
                    if only_deficits:
                        st.dataframe(deficits_df, width="stretch", height=420)
                    else:
                        st.dataframe(deficits_df, width="stretch", height=420)

        # Explicit save hint for imported availability
        st.caption("提示：侧边栏导入总表只会更新本次会话内的数据；如需写入 Firebase，请点击侧边栏的 “Save All Changes”。")

    st.caption("说明：小组规则用于校验排班是否满足“某时段最少需要多少人值更”。目前按“小时”进行覆盖校验。")

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
        default_windows_df = pd.DataFrame([{"day_type": "all", "start": "00:00", "end": "24:00", "min_staff": 1}])
        win_df = st.data_editor(default_windows_df, num_rows="dynamic", width="stretch", key="new_group_windows")

        submitted = st.form_submit_button("创建小组")
        if submitted:
            if not new_name.strip():
                st.error("小组名称不能为空。")
            else:
                # Prevent duplicate names
                if any(g.get("name") == new_name.strip() for g in groups):
                    st.error("已存在同名小组，请换一个名称。")
                else:
                    windows = []
                    for _, r in win_df.iterrows():
                        day_type = str(r.get("day_type", "all")).strip().lower()
                        start = str(r.get("start", "00:00")).strip()
                        end = str(r.get("end", "24:00")).strip()
                        try:
                            min_staff = int(r.get("min_staff", 1))
                        except Exception:
                            min_staff = 1
                        if not start or not end:
                            continue
                        windows.append({"day_type": day_type, "start": start, "end": end, "min_staff": min_staff})

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
                edited_windows_df = st.data_editor(
                    windows_df,
                    num_rows="dynamic",
                    width="stretch",
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
                        new_windows = []
                        for _, r in edited_windows_df.iterrows():
                            day_type = str(r.get("day_type", "all")).strip().lower()
                            start = str(r.get("start", "00:00")).strip()
                            end = str(r.get("end", "24:00")).strip()
                            try:
                                min_staff = int(r.get("min_staff", 1))
                            except Exception:
                                min_staff = 1
                            if not start or not end:
                                continue
                            new_windows.append({"day_type": day_type, "start": start, "end": end, "min_staff": min_staff})

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
                        st.session_state.selected_group_name = new_name_norm
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
                        st.session_state.selected_group_name = remaining[0]
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

