import io
import json
import zipfile
from typing import Any, Dict, List

import pandas as pd


def build_role_rules_df(role_rules: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for role, rule in (role_rules or {}).items():
        if isinstance(rule, dict):
            rule_str = json.dumps(rule, ensure_ascii=False)
        else:
            rule_str = str(rule)
        rows.append({"role": str(role), "rule": rule_str})
    return pd.DataFrame(rows)


def build_employees_df(employees: List[Any]) -> pd.DataFrame:
    data = []
    for emp in employees or []:
        if isinstance(emp, dict):
            data.append(emp)
        else:
            data.append(getattr(emp, "__dict__", {"name": str(emp)}))
    return pd.DataFrame(data)


def export_role_rules_employees_excel(role_rules: Dict[str, Any], employees: List[Any]) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output) as writer:
        build_role_rules_df(role_rules).to_excel(writer, index=False, sheet_name="role_rules")
        build_employees_df(employees).to_excel(writer, index=False, sheet_name="employees")
    return output.getvalue()


def export_role_rules_employees_zip(role_rules: Dict[str, Any], employees: List[Any]) -> bytes:
    employees_json = json.dumps(
        [e if isinstance(e, dict) else getattr(e, "__dict__", {}) for e in employees or []],
        ensure_ascii=False,
        indent=2,
    )
    role_rules_json = json.dumps(role_rules or {}, ensure_ascii=False, indent=2)

    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("employees.json", employees_json)
        zf.writestr("role_rules.json", role_rules_json)
    return output.getvalue()
