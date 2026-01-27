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
):
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
