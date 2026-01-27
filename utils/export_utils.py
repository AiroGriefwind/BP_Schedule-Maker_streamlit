import io
import json
import zipfile
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd


def _join_list(values: Iterable[Any]) -> str:
    return ", ".join([str(v) for v in values if str(v).strip()])


def _normalize_employees(employees: List[Any]) -> List[Dict[str, Any]]:
    data = []
    for emp in employees or []:
        if isinstance(emp, dict):
            data.append(emp)
        else:
            data.append(getattr(emp, "__dict__", {"name": str(emp)}))
    return data


def build_group_rules_dfs(group_rules: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    meta_rows = []
    groups = []
    if isinstance(group_rules, dict):
        meta_rows.append({"key": "version", "value": group_rules.get("version")})
        meta_rows.append({"key": "updated_at", "value": group_rules.get("updated_at")})
        groups = group_rules.get("groups") or []

    summary_rows = []
    window_rows = []
    for g in groups or []:
        members = g.get("members") or []
        backups = g.get("backup_members") or []
        windows = g.get("requirements_windows") or []
        summary_rows.append(
            {
                "group_id": g.get("id"),
                "group_name": g.get("name"),
                "description": g.get("description"),
                "rule_type": g.get("rule_type"),
                "active": g.get("active"),
                "headcount_planned": g.get("headcount_planned"),
                "members": _join_list(members),
                "backup_members": _join_list(backups),
                "windows_count": len(windows),
            }
        )
        for w in windows:
            window_rows.append(
                {
                    "group_id": g.get("id"),
                    "group_name": g.get("name"),
                    "day_type": w.get("day_type"),
                    "start": w.get("start"),
                    "end": w.get("end"),
                    "min_staff": w.get("min_staff"),
                }
            )

    meta_df = pd.DataFrame(meta_rows, columns=["key", "value"])
    summary_df = pd.DataFrame(
        summary_rows,
        columns=[
            "group_id",
            "group_name",
            "description",
            "rule_type",
            "active",
            "headcount_planned",
            "members",
            "backup_members",
            "windows_count",
        ],
    )
    windows_df = pd.DataFrame(
        window_rows,
        columns=["group_id", "group_name", "day_type", "start", "end", "min_staff"],
    )
    return meta_df, summary_df, windows_df


def build_employees_dfs(employees: List[Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data = _normalize_employees(employees)
    base_rows = []
    group_rows = []
    for emp in data:
        base_rows.append(
            {
                "name": emp.get("name"),
                "role": emp.get("role"),
                "additional_roles": _join_list(emp.get("additional_roles") or []),
                "start_time": emp.get("start_time"),
                "end_time": emp.get("end_time"),
            }
        )
        for g in emp.get("group_assignments") or []:
            group_rows.append(
                {
                    "name": emp.get("name"),
                    "group": g.get("group"),
                    "member_type": g.get("member_type"),
                }
            )

    employees_df = pd.DataFrame(
        base_rows,
        columns=["name", "role", "additional_roles", "start_time", "end_time"],
    )
    assignments_df = pd.DataFrame(
        group_rows,
        columns=["name", "group", "member_type"],
    )
    return employees_df, assignments_df


def export_group_rules_employees_excel(group_rules: Dict[str, Any], employees: List[Any]) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output) as writer:
        meta_df, summary_df, windows_df = build_group_rules_dfs(group_rules)
        employees_df, assignments_df = build_employees_dfs(employees)
        meta_df.to_excel(writer, index=False, sheet_name="group_rules_meta")
        summary_df.to_excel(writer, index=False, sheet_name="group_rules")
        windows_df.to_excel(writer, index=False, sheet_name="group_rules_windows")
        employees_df.to_excel(writer, index=False, sheet_name="employees")
        assignments_df.to_excel(writer, index=False, sheet_name="employee_groups")
    return output.getvalue()


def export_group_rules_employees_zip(group_rules: Dict[str, Any], employees: List[Any]) -> bytes:
    employees_json = json.dumps(
        _normalize_employees(employees),
        ensure_ascii=False,
        indent=2,
    )
    group_rules_json = json.dumps(group_rules or {}, ensure_ascii=False, indent=2)

    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("employees.json", employees_json)
        zf.writestr("group_rules.json", group_rules_json)
    return output.getvalue()
