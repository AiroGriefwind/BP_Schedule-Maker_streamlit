import pandas as pd
import streamlit as st

from utils.availability_utils import (
    availability_to_color_css_dataframe,
    availability_to_dataframe,
    dataframe_to_availability,
)


def render_availability_tab(
    *,
    role_rules,
    components,
    import_from_google_form,
    export_availability_to_excel,
    generated_schedule,
    availability_df,
    save_data,
    save_employees,
    clear_availability,
    add_employee,
    delete_employee,
):
    # --- Master sheet warnings (show at top) ---
    missing_in_system = st.session_state.get("imported_names_missing") or []
    extra_in_system = st.session_state.get("extra_employees") or []
    if missing_in_system:
        st.warning(f"总表中有 {len(missing_in_system)} 名员工不在系统中（可在下方展开处理）。")
    if extra_in_system:
        st.warning(f"系统中有 {len(extra_in_system)} 名员工未出现在总表中（可在下方展开处理）。")

    # --- Availability Editor ---
    st.header("Availability Grid")
    # Role filter
    all_roles = ["All"] + list(role_rules.keys())
    selected_role = st.selectbox("Filter by Role:", options=all_roles)

    # Filter employees based on role
    if selected_role == "All":
        filtered_employees = [emp.name for emp in st.session_state.employees]
    else:
        filtered_employees = [emp.name for emp in st.session_state.employees if emp.employee_type == selected_role]

    availability_df = availability_to_dataframe()
    availability_color_css_df = availability_to_color_css_dataframe()

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

    # --- Collapsible missing/extra employee UI (after grid, before actions) ---
    if missing_in_system:
        missing_count = len(missing_in_system)
        with st.expander(f"总表有但系统没有的员工（{missing_count}）", expanded=False):
            st.caption("可逐个添加到系统中（如无需处理，可忽略）。")
            for name in missing_in_system:
                with st.form(key=f"add_{name}_form"):
                    st.write(f"Add employee: {name}")
                    role = st.selectbox(f"Role for {name}", list(role_rules.keys()))
                    shift = st.text_input(f"Shift for {name} (e.g., 10-19)")
                    start, end = None, None
                    if "-" in shift:
                        start, end = shift.split("-", 1)
                    submit = st.form_submit_button("Add Employee")
                    if submit:
                        add_employee(name, role, start, end)
                        st.success(f"Employee {name} added.")
                        try:
                            st.session_state.imported_names_missing.remove(name)
                        except Exception:
                            pass
                        st.rerun()

    if extra_in_system:
        extra_count = len(extra_in_system)
        with st.expander(f"系统有但总表没有的员工（{extra_count}）", expanded=False):
            st.caption("如需保持系统与总表一致，可在此移除。")
            for extra_name in list(extra_in_system):
                with st.form(key=f"remove_{extra_name}_form"):
                    st.write(f"员工“{extra_name}”在系统中存在，但未出现在导入的总表中。")
                    remove = st.form_submit_button(f"移除“{extra_name}”")
                    if remove:
                        delete_employee(extra_name)
                        try:
                            st.session_state.extra_employees.remove(extra_name)
                        except Exception:
                            pass
                        st.toast(f"🗑️ Employee '{extra_name}' removed from system (not in latest main sheet import).")
                        st.session_state.initialized = False
                        st.rerun()

    st.divider()
    st.subheader("Actions")
    if st.button("Save All Changes", type="primary"):
        with st.spinner("Saving data..."):
            merged_availability = availability_utils.merge_edited_df_with_color(
                availability_df, st.session_state.availability
            )
            st.session_state.availability = merged_availability
            save_data(st.session_state.availability)
            save_employees(st.session_state.employees)
            st.toast("💾 All changes saved to server files!")

    if st.button("保存当前总表到 Firebase（仅 availability）", type="secondary"):
        try:
            with st.spinner("Saving availability to Firebase..."):
                save_data(st.session_state.availability)
            st.toast("✅ 已保存当前 availability 到 Firebase。")
        except Exception as e:
            st.error(f"保存失败：{e}")

    if st.button("Clear All Availability"):
        st.session_state.availability = clear_availability(
            st.session_state.start_date, st.session_state.employees
        )
        st.toast("🗑️ Availability cleared and reset.")
        st.rerun()

    st.divider()
    st.subheader("Data Import")
    google_form_upload = st.file_uploader("Import from Google Form (Excel)", type=["xlsx"])
    if google_form_upload:
        try:
            with st.spinner("Importing..."):
                result = import_from_google_form(google_form_upload)
                st.toast(f"✅ {result}")
                # Reload data after import
                st.session_state.initialized = False
                st.rerun()
        except Exception as e:
            st.error(f"Import failed: {e}")

    st.subheader("Data Export")
    avail_export_data = export_availability_to_excel(st.session_state.availability)
    st.download_button(
        label="Export Availability to Excel",
        data=avail_export_data,
        file_name="availability_export.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    if generated_schedule is not None and not generated_schedule.empty:
        schedule_export_data = generated_schedule.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Export Schedule to CSV",
            data=schedule_export_data,
            file_name="generated_schedule.csv",
            mime="text/csv",
        )
