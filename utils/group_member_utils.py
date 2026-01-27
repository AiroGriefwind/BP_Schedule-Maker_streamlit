from typing import Any, Dict, List, Tuple


# -----------------------------
# Group member helpers (members vs backups)
# -----------------------------
_GROUP_MEMBER_TYPE_LABELS = {"member": "通常成员", "backup": "备选成员"}


def _normalize_member_list(values: Any) -> List[str]:
    if not isinstance(values, list):
        values = []
    cleaned = [str(x).strip() for x in values if str(x).strip()]
    seen = set()
    return [m for m in cleaned if not (m in seen or seen.add(m))]


def _get_employee_group_assignments(employee_name: str, group_rules: Dict[str, Any]) -> List[Dict[str, str]]:
    assignments: List[Dict[str, str]] = []
    for g in (group_rules or {}).get("groups", []) or []:
        gname = str(g.get("name") or "").strip()
        if not gname:
            continue
        members = g.get("members", []) or []
        backups = g.get("backup_members", []) or []
        if employee_name in members:
            assignments.append({"小组": gname, "成员类型": _GROUP_MEMBER_TYPE_LABELS["member"]})
        if employee_name in backups:
            confirm = {"小组": gname, "成员类型": _GROUP_MEMBER_TYPE_LABELS["backup"]}
            assignments.append(confirm)
    return assignments


def _apply_employee_group_assignments(
    employee_name: str,
    assignments: List[Dict[str, Any]],
    group_rules: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[str], bool]:
    warnings: List[str] = []
    changed = False
    if not employee_name:
        return group_rules, warnings, changed

    groups = (group_rules or {}).get("groups", []) or []
    name_to_group = {str(g.get("name") or "").strip(): g for g in groups if str(g.get("name") or "").strip()}

    # Remove existing memberships first (treat assignments as the full desired list).
    for g in groups:
        members = g.get("members", []) or []
        backups = g.get("backup_members", []) or []
        if employee_name in members:
            g["members"] = [m for m in members if m != employee_name]
            changed = True
        if employee_name in backups:
            g["backup_members"] = [m for m in backups if m != employee_name]
            changed = True

    seen_groups = set()
    for row in assignments or []:
        if not isinstance(row, dict):
            continue
        gname = str(row.get("小组") or "").strip()
        pool_label = str(row.get("成员类型") or "").strip()
        if not gname:
            continue
        if gname in seen_groups:
            warnings.append(f"小组“{gname}”在列表中重复，已保留第一条。")
            continue
        seen_groups.add(gname)
        g = name_to_group.get(gname)
        if not g:
            warnings.append(f"未找到小组“{gname}”，已跳过。")
            continue
        target_key = "members"
        if pool_label == _GROUP_MEMBER_TYPE_LABELS["backup"]:
            target_key = "backup_members"
        elif pool_label and pool_label != _GROUP_MEMBER_TYPE_LABELS["member"]:
            warnings.append(f"小组“{gname}”的成员类型无效，已按通常成员处理。")
        cur = _normalize_member_list(g.get(target_key) or [])
        if employee_name not in cur:
            cur.append(employee_name)
            g[target_key] = cur
            changed = True
        other_key = "backup_members" if target_key == "members" else "members"
        other = _normalize_member_list(g.get(other_key) or [])
        if employee_name in other:
            g[other_key] = [m for m in other if m != employee_name]
            changed = True

    # Final cleanup: de-dup + remove overlaps
    for g in groups:
        members = _normalize_member_list(g.get("members") or [])
        backups = _normalize_member_list(g.get("backup_members") or [])
        backups = [m for m in backups if m not in members]
        g["members"] = members
        g["backup_members"] = backups

    return group_rules, warnings, changed
