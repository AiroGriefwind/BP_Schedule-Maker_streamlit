import streamlit as st
import pandas as pd
from datetime import datetime
import json
import uuid
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
            result[date][emp] = {"value": new_val, "color": cell_color}
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
            df[emp] = [[] for _ in range(len(df))]
    df = df[col_order]  # This sets the display order

    # Convert lists to comma-separated strings for st.data_editor
    return df.applymap(lambda x: ', '.join(map(str, x)) if isinstance(x, list) else x)

def convert_availability_dates_to_str(availability):
    """
    Recursively convert all datetime keys in the availability dict to strings (YYYY-MM-DD).
    Returns a new dict that is JSON serializable.
    """
    import datetime

    def conv(obj):
        if isinstance(obj, dict):
            return {conv_key(k): conv(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [conv(v) for v in obj]
        return obj

    def conv_key(k):
        if isinstance(k, datetime.date) or isinstance(k, datetime.datetime):
            return k.strftime("%Y-%m-%d")
        return str(k)
    
    return conv(availability)


def dataframe_to_availability(edited_df):
    """Converts the edited DataFrame back to the availability dictionary format."""
    # Convert comma-separated strings back to lists
    df = edited_df.applymap(lambda x: [item.strip() for item in x.split(',')] if isinstance(x, str) and x else [])
    
    # Transpose back to original format {date: {employee: [shifts]}}
    st.session_state.availability = df.T.to_dict()


# --- Initialization ---
initialize_session_state()
availability_df = availability_to_dataframe()

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
availability_json = json.dumps(availability_serializable, indent=4)

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

    group_rules = st.session_state.get("group_rules") or GROUP_RULES
    groups = group_rules.get("groups", [])

    st.caption("说明：小组规则用于校验排班是否满足“某时段最少需要多少人值更”。目前按“小时”进行覆盖校验。")

    # Overview
    if groups:
        summary_rows = []
        for g in groups:
            summary_rows.append({
                "名称": g.get("name"),
                "启用": bool(g.get("active", True)),
                "成员数": len(g.get("members", []) or []),
                "规则段数": len(g.get("requirements_windows", []) or []),
            })
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)
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
        win_df = st.data_editor(default_windows_df, num_rows="dynamic", use_container_width=True, key="new_group_windows")

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
        selected_group_name = st.selectbox("选择小组", options=list(name_to_group.keys()))
        g = name_to_group.get(selected_group_name)

        if g:
            edit_cols = st.columns([2, 2])
            with edit_cols[0]:
                edited_name = st.text_input("小组名称", value=g.get("name", ""), key="edit_group_name")
                edited_desc = st.text_input("备注/说明", value=g.get("description", ""), key="edit_group_desc")
                edited_active = st.checkbox("启用", value=bool(g.get("active", True)), key="edit_group_active")
                edited_headcount = st.number_input(
                    "规划人数（可选）", min_value=0, value=int(g.get("headcount_planned") or 0), step=1, key="edit_group_headcount"
                )
                edited_members = st.multiselect(
                    "成员（从现有员工中选择）",
                    options=employee_names,
                    default=[m for m in (g.get("members") or []) if m in employee_names],
                    key="edit_group_members",
                )

            with edit_cols[1]:
                windows_df = pd.DataFrame(g.get("requirements_windows") or [])
                if windows_df.empty:
                    windows_df = pd.DataFrame([{"day_type": "all", "start": "00:00", "end": "24:00", "min_staff": 1}])
                edited_windows_df = st.data_editor(windows_df, num_rows="dynamic", use_container_width=True, key="edit_group_windows")

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
                        st.session_state.initialized = False
                        st.rerun()

            with action_cols[1]:
                confirm_delete = st.checkbox("确认删除", value=False, key="confirm_delete_group")
                if st.button("删除该小组", type="secondary", disabled=not confirm_delete):
                    group_rules["groups"] = [x for x in group_rules.get("groups", []) if x.get("id") != g.get("id")]
                    st.session_state.group_rules = group_rules
                    save_group_rules(st.session_state.group_rules)
                    st.toast("🗑️ 小组已删除并保存到 Firebase。")
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

