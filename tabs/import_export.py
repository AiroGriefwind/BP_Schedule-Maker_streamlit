import streamlit as st

from utils.export_utils import (
    export_role_rules_employees_excel,
    export_role_rules_employees_zip,
)


def _load_from_storage_or_session(fm, role_rules, employees):
    # Prefer Firebase Storage config/*.json if available.
    storage_group_rules = None
    storage_employees = None
    try:
        storage_group_rules = fm.get_json_from_storage("config/group_rules.json")
    except Exception:
        storage_group_rules = None
    try:
        storage_employees = fm.get_json_from_storage("config/employees.json")
    except Exception:
        storage_employees = None

    final_group_rules = storage_group_rules if storage_group_rules is not None else role_rules
    final_employees = storage_employees if storage_employees is not None else employees
    return final_group_rules, final_employees


def render_import_export_tab(*, role_rules, employees, fm):
    st.header("导入/导出")
    st.caption("用于备份 config 中的规则与员工信息（优先读取 Storage 的 config/*.json）。")

    export_role_rules, export_employees = _load_from_storage_or_session(fm, role_rules, employees)

    excel_bytes = export_role_rules_employees_excel(export_role_rules, export_employees)
    st.download_button(
        label="导出为Excel",
        data=excel_bytes,
        file_name="role_rules_employees.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    zip_bytes = export_role_rules_employees_zip(export_role_rules, export_employees)
    st.download_button(
        label="导出为JSON",
        data=zip_bytes,
        file_name="role_rules_employees.zip",
        mime="application/zip",
    )
