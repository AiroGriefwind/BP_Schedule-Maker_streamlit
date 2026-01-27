import json
import pandas as pd
import streamlit as st

from utils.group_member_utils import (
    _GROUP_MEMBER_TYPE_LABELS,
    _apply_employee_group_assignments,
    _get_employee_group_assignments,
)


def render_employee_management_tab(
    *,
    refresh_master_data,
    save_employees_to_storage_only,
    add_employee,
    edit_employee,
    delete_employee,
    save_group_rules,
    ROLE_RULES,
    GROUP_RULES,
):
    # --- Employee Management Section ---
    action_cols = st.columns([1, 1, 2])
    with action_cols[0]:
        if st.button("刷新"):
            with st.spinner("Refreshing from Firebase..."):
                refresh_master_data()
            st.toast("🔄 已刷新员工/角色规则，并同步 availability。")
            st.rerun()
    with action_cols[1]:
        if st.button("手动保存"):
            with st.spinner("Saving employees to Storage..."):
                save_employees_to_storage_only(st.session_state.employees)
            st.toast("💾 员工已保存到 Storage/config/employees.json。")

    add_tab, edit_tab, import_tab = st.tabs(["添加员工", "编辑/删除员工", "导入/诊断"])

    with add_tab:
        st.subheader("Add New Employee")
        with st.form("add_employee_form", clear_on_submit=True):
            add_name = st.text_input("Name")
            add_role = st.selectbox("Role", list(ROLE_RULES.keys()))
            add_start_time = st.text_input("Start Time (for fixed time roles)", "10-19")
            add_end_time = ""
            if '-' in add_start_time:
                add_start_time, add_end_time = add_start_time.split('-')

            group_rules_state = st.session_state.get("group_rules") or GROUP_RULES
            group_names = [g.get("name") for g in group_rules_state.get("groups", []) if g.get("name")]
            add_group_df = pd.DataFrame(columns=["小组", "成员类型"])
            if group_names:
                st.markdown("**小组分配（可选）**")
                st.caption("可为该员工选择多个小组，并指定为“通常成员”或“备选成员”。")
                add_group_df = st.data_editor(
                    add_group_df,
                    num_rows="dynamic",
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "小组": st.column_config.SelectboxColumn("小组", options=group_names, required=True),
                        "成员类型": st.column_config.SelectboxColumn(
                            "成员类型",
                            options=list(_GROUP_MEMBER_TYPE_LABELS.values()),
                            required=True,
                        ),
                    },
                    key="add_employee_group_assignments",
                )
            else:
                st.caption("暂无小组规则，创建后可在员工信息中分配。")

            if st.form_submit_button("Add Employee"):
                add_employee(add_name, add_role, start_time=add_start_time, end_time=add_end_time)
                if group_names:
                    assignments = add_group_df.to_dict("records") if isinstance(add_group_df, pd.DataFrame) else []
                    updated_rules, warnings, changed = _apply_employee_group_assignments(
                        add_name.strip(), assignments, group_rules_state
                    )
                    if changed:
                        st.session_state.group_rules = updated_rules
                        save_group_rules(updated_rules)
                    for w in warnings:
                        st.warning(w)
                st.toast(f"✅ Employee '{add_name}' added.")
                st.session_state.initialized = False
                st.rerun()

    with edit_tab:
        st.subheader("Edit or Delete Employee")
        employees_list = st.session_state.employees
        selected_employee_name = st.selectbox("Select Employee to Edit/Delete", [e.name for e in employees_list])

        if selected_employee_name:
            emp_to_edit = next((e for e in employees_list if e.name == selected_employee_name), None)

            with st.form("edit_employee_form"):
                st.write(f"Editing: **{emp_to_edit.name}**")
                new_name = st.text_input("New Name", value=emp_to_edit.name)
                new_role = st.selectbox(
                    "New Role",
                    list(ROLE_RULES.keys()),
                    index=list(ROLE_RULES.keys()).index(emp_to_edit.employee_type),
                )
                new_shift = st.text_input(
                    "New Shift (e.g., 10-19)",
                    value=f"{emp_to_edit.start_time}-{emp_to_edit.end_time}" if emp_to_edit.start_time else "",
                )

                group_rules_state = st.session_state.get("group_rules") or GROUP_RULES
                group_names = [g.get("name") for g in group_rules_state.get("groups", []) if g.get("name")]
                edit_group_df = pd.DataFrame(_get_employee_group_assignments(emp_to_edit.name, group_rules_state))
                if edit_group_df.empty:
                    edit_group_df = pd.DataFrame(columns=["小组", "成员类型"])
                if group_names:
                    st.markdown("**小组分配（可选）**")
                    st.caption("每行代表该员工在一个小组中的身份。")
                    edit_group_df = st.data_editor(
                        edit_group_df,
                        num_rows="dynamic",
                        width="stretch",
                        hide_index=True,
                        column_config={
                            "小组": st.column_config.SelectboxColumn("小组", options=group_names, required=True),
                            "成员类型": st.column_config.SelectboxColumn(
                                "成员类型",
                                options=list(_GROUP_MEMBER_TYPE_LABELS.values()),
                                required=True,
                            ),
                        },
                        key=f"edit_employee_group_assignments_{emp_to_edit.name}",
                    )
                else:
                    st.caption("暂无小组规则，创建后可在员工信息中分配。")

                submitted = st.form_submit_button("Update Employee")
                if submitted:
                    start_time, end_time = (new_shift.split('-') if '-' in new_shift else (None, None))
                    edit_employee(
                        emp_to_edit.name,
                        new_name,
                        new_role,
                        new_start_time=start_time,
                        new_end_time=end_time,
                    )
                    if group_names:
                        assignments = edit_group_df.to_dict("records") if isinstance(edit_group_df, pd.DataFrame) else []
                        name_for_groups = new_name.strip() if new_name.strip() else emp_to_edit.name
                        updated_rules, warnings, changed = _apply_employee_group_assignments(
                            name_for_groups,
                            assignments,
                            group_rules_state,
                        )
                        if changed:
                            st.session_state.group_rules = updated_rules
                            save_group_rules(updated_rules)
                        for w in warnings:
                            st.warning(w)
                    st.toast(f"✅ Employee '{new_name}' updated.")
                    st.session_state.initialized = False
                    st.rerun()

            if st.button(f"Delete {selected_employee_name}", type="secondary"):
                delete_employee(selected_employee_name)
                st.toast(f"🗑️ Employee '{selected_employee_name}' deleted.")
                st.session_state.initialized = False
                st.rerun()

    with import_tab:
        st.subheader("导入 employees.json（可选）")
        st.caption("选择文件后只会在本次会话中解析与预览，不会自动写入 线上数据库。需要你点击“应用/保存”按钮才会生效。")
        uploaded_employees = st.file_uploader(
            "选择一个 employees.json（或 线上数据库的备份文件）",
            type=["json"],
            key="employees_import_uploader",
        )
        if uploaded_employees is not None:
            try:
                raw_text = uploaded_employees.getvalue().decode("utf-8", errors="ignore")
                imported_obj = json.loads(raw_text)
                st.session_state["_imported_employees_preview"] = imported_obj
            except Exception as e:
                st.session_state.pop("_imported_employees_preview", None)
                st.error(f"导入失败：{e}")

        preview_obj = st.session_state.get("_imported_employees_preview")
        if isinstance(preview_obj, list):
            st.success(f"已解析：{len(preview_obj)} 名员工。")
            if preview_obj:
                names = [x.get("name") for x in preview_obj if isinstance(x, dict) and x.get("name")]
                if names:
                    st.caption("预览（前 12 名员工）： " + "、".join([str(x) for x in names[:12]]))

            import_cols = st.columns([1, 1, 2])
            with import_cols[0]:
                if st.button("应用到当前会话", type="secondary", key="apply_imported_employees"):
                    try:
                        from scheduling_logic import _normalize_employee_records, _build_employees  # type: ignore
                        records = _normalize_employee_records(preview_obj)
                        st.session_state.employees = _build_employees(records)
                        st.toast("已应用导入的员工数据到当前会话（未写入 线上数据库）。")
                        st.session_state.initialized = False
                        st.rerun()
                    except Exception as e:
                        st.error(f"应用失败：{e}")
            with import_cols[1]:
                if st.button("应用并保存", type="primary", key="apply_and_save_imported_employees"):
                    try:
                        from scheduling_logic import _normalize_employee_records, _build_employees  # type: ignore
                        records = _normalize_employee_records(preview_obj)
                        st.session_state.employees = _build_employees(records)
                        save_employees_to_storage_only(st.session_state.employees)
                        st.toast("✅ 已导入并保存到 线上数据库。")
                        st.session_state.initialized = False
                        st.rerun()
                    except Exception as e:
                        st.error(f"保存失败：{e}")
            with import_cols[2]:
                st.caption("说明：保存仅写入 Storage/config/employees.json，不会自动覆盖 RTDB。")

        st.subheader("当前会话员工（JSON 预览）")
        def _employee_to_json(emp):
            return {
                "name": getattr(emp, "name", None),
                "role": getattr(emp, "employee_type", None) or getattr(emp, "role", None),
                "additional_roles": getattr(emp, "additional_roles", None) or [],
                "start_time": getattr(emp, "start_time", None),
                "end_time": getattr(emp, "end_time", None),
            }
        current_json = [_employee_to_json(e) for e in (st.session_state.employees or [])]
        st.json(current_json)

        with st.expander("诊断：线上数据库读取到的员工（只读）", expanded=False):
            try:
                import firebase_manager as fm  # fallback to default manager
                raw = fm.get_data("employees")
                if raw is None:
                    st.warning("fm.get_data('employees') 返回 None（线上数据库该路径可能为空/无权限/连接异常）。")
                else:
                    st.caption(f"fm.get_data('employees') 类型：{type(raw).__name__}")
                    if isinstance(raw, list):
                        st.caption(f"records 数量: {len(raw)}")
                    st.json(raw)
            except Exception as e:
                st.error(f"诊断读取失败：{e}")
