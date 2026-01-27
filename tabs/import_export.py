import streamlit as st

from utils.export_utils import (
    export_role_rules_employees_excel,
    export_role_rules_employees_zip,
)


def render_import_export_tab(*, role_rules, employees):
    st.header("导入/导出")
    st.caption("用于备份当前系统配置数据（角色规则、员工信息）。")

    excel_bytes = export_role_rules_employees_excel(role_rules, employees)
    st.download_button(
        label="导出为Excel",
        data=excel_bytes,
        file_name="role_rules_employees.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    zip_bytes = export_role_rules_employees_zip(role_rules, employees)
    st.download_button(
        label="导出为JSON",
        data=zip_bytes,
        file_name="role_rules_employees.zip",
        mime="application/zip",
    )
