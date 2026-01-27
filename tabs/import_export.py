import json
import streamlit as st

from utils.export_utils import (
    export_group_rules_employees_excel,
    export_group_rules_employees_zip,
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


def render_import_export_tab(
    *,
    role_rules,
    employees,
    fm=None,
    main_shift_file_handler=None,
    **_ignored,
):
    if fm is None:
        import firebase_manager as fm  # fallback when caller doesn't pass it
    st.header("导入/导出")

    import_tab, export_tab = st.tabs(["导入", "导出"])

    with import_tab:
        if main_shift_file_handler is not None:
            main_shift_file_handler()

        st.subheader("恢复线上数据")
        st.caption("用于检查并恢复 Storage/config 下的 employees.json 与 role_rules.json。")

        if st.button("验证线上数据完整度", key="validate_config_storage"):
            errors = []
            employees_obj = None
            role_rules_obj = None
            try:
                employees_obj = fm.get_json_from_storage("config/employees.json")
            except Exception as e:
                errors.append(f"employees.json 读取失败：{e}")
            try:
                role_rules_obj = fm.get_json_from_storage("config/role_rules.json")
            except Exception as e:
                errors.append(f"role_rules.json 读取失败：{e}")

            if employees_obj is None:
                errors.append("employees.json 缺失或无法解析。")
            if role_rules_obj is None:
                errors.append("role_rules.json 缺失或无法解析。")

            if errors:
                st.warning("检测到问题：\n\n- " + "\n- ".join(errors))
            else:
                st.success("✅ 两个文件均可正常读取。")

            with st.expander("employees.json 预览（只读）", expanded=False):
                st.json(employees_obj)
            with st.expander("role_rules.json 预览（只读）", expanded=False):
                st.json(role_rules_obj)

        uploader_cols = st.columns(2)
        with uploader_cols[0]:
            uploaded_employees = st.file_uploader(
                "上传 employees.json",
                type=["json"],
                key="restore_employees_json",
            )
            if uploaded_employees is not None:
                try:
                    raw_text = uploaded_employees.getvalue().decode("utf-8", errors="ignore")
                    employees_obj = json.loads(raw_text)
                    fm.save_json_to_storage("config/employees.json", employees_obj)
                    st.success("已上传并覆盖 config/employees.json。")
                except Exception as e:
                    st.error(f"employees.json 上传失败：{e}")

        with uploader_cols[1]:
            uploaded_role_rules = st.file_uploader(
                "上传 role_rules.json",
                type=["json"],
                key="restore_role_rules_json",
            )
            if uploaded_role_rules is not None:
                try:
                    raw_text = uploaded_role_rules.getvalue().decode("utf-8", errors="ignore")
                    role_rules_obj = json.loads(raw_text)
                    fm.save_json_to_storage("config/role_rules.json", role_rules_obj)
                    st.success("已上传并覆盖 config/role_rules.json。")
                except Exception as e:
                    st.error(f"role_rules.json 上传失败：{e}")

    with export_tab:
        st.caption("用于备份 config 中的规则与员工信息（优先读取 Storage 的 config/*.json）。")

        export_group_rules, export_employees = _load_from_storage_or_session(fm, role_rules, employees)

        excel_bytes = export_group_rules_employees_excel(export_group_rules, export_employees)
        st.download_button(
            label="导出为Excel",
            data=excel_bytes,
            file_name="group_rules_employees.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        zip_bytes = export_group_rules_employees_zip(export_group_rules, export_employees)
        st.download_button(
            label="导出为JSON",
            data=zip_bytes,
            file_name="group_rules_employees.zip",
            mime="application/zip",
        )
