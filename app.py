import streamlit as st
import pandas as pd
from datetime import datetime
import json
from scheduling_logic import (
    load_employees,
    load_data,
    save_employees,
    save_data,
    init_availability,
    generate_schedule,
    import_from_google_form,
    add_employee,
    edit_employee,
    delete_employee,
    export_availability_to_excel,
    clear_availability,
    sync_availability,
    EMPLOYEES,
    ROLE_RULES,
)

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
    if 'initialized' not in st.session_state:
        st.session_state.start_date = datetime(2025, 3, 17)
        st.session_state.employees = load_employees()
        
        # Sync employees with availability data
        sync_availability() 
        
        st.session_state.availability = load_data()
        
        if not st.session_state.availability:
            st.session_state.availability = init_availability(
                st.session_state.start_date, st.session_state.employees
            )
        st.session_state.initialized = True
        st.session_state.warnings = []
        st.session_state.generated_schedule = None

# --- Helper Functions ---
def availability_to_dataframe():
    """Converts the availability dictionary to a Pandas DataFrame for editing."""
    availability_dict = st.session_state.availability
    if not availability_dict:
        return pd.DataFrame()

    df = pd.DataFrame(availability_dict).T
    # Ensure all employees are columns
    emp_names = [emp.name for emp in st.session_state.employees]
    for emp in emp_names:
        if emp not in df.columns:
            df[emp] = [[] for _ in range(len(df))]
            
    df = df[emp_names] # Ensure correct order
    # Convert lists to comma-separated strings for st.data_editor
    return df.applymap(lambda x: ', '.join(map(str, x)) if isinstance(x, list) else x)

def dataframe_to_availability(edited_df):
    """Converts the edited DataFrame back to the availability dictionary format."""
    # Convert comma-separated strings back to lists
    df = edited_df.applymap(lambda x: [item.strip() for item in x.split(',')] if isinstance(x, str) and x else [])
    
    # Transpose back to original format {date: {employee: [shifts]}}
    st.session_state.availability = df.T.to_dict()


# --- Initialization ---
initialize_session_state()


# --- Sidebar UI ---
st.sidebar.title("🗓️ Schedule Maker")
st.sidebar.write("Manage employee availability and generate schedules.")

st.sidebar.header("Actions")
if st.sidebar.button("Generate Schedule"):
    with st.spinner("Generating schedule..."):
        warnings = generate_schedule(st.session_state.availability, st.session_state.start_date, export_to_excel=False)
        st.session_state.warnings = warnings
        st.session_state.generated_schedule = pd.DataFrame(st.session_state.get_last_generated_schedule())
        if not warnings:
            st.toast("✅ Schedule generated successfully!")
        else:
            st.toast(f"⚠️ Schedule generated with {len(warnings)} warnings.")


if st.sidebar.button("Save All Changes", type="primary"):
    with st.spinner("Saving data..."):
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

availability_json = json.dumps(st.session_state.availability, indent=4)
st.sidebar.download_button("Download availability.json", availability_json, "availability.json")


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
    # Filter columns based on selected role
    display_df = availability_df[filtered_employees]
    
    st.info("You can directly edit the cells below. Changes are saved when you click 'Save All Changes'.")
    edited_df = st.data_editor(display_df, height=600)

    # If the dataframe has been edited, update the session state
    # Create a full dataframe with all employees to update the state correctly
    full_edited_df = availability_df.copy()
    full_edited_df.update(edited_df)
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

