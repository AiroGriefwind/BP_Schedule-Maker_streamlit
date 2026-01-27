import json
import uuid
from datetime import datetime
from typing import Any, Tuple

import pandas as pd
import streamlit as st


def render_group_rules_tab(
    *,
    group_rules_enabled,
    load_group_rules,
    save_group_rules,
    group_rules_default,
    fm,
    validate_group_coverage_from_availability,
    build_week_bins_from_dates,
    build_week_grid_df,
    build_cell_member_detail_df,
    extract_date_time_from_obj,
    availability_cell_css,
    normalize_windows_df_for_editor,
    validate_and_build_windows_df,
    day_type_options_base,
    time_options_base,
    alt,
):
    # --- Custom Group Rules (Team Rules) ---
    with st.expander("自定义更表规则（小组）"):
        if not group_rules_enabled:
            st.warning("当前部署环境的 `scheduling_logic.py` 版本不包含小组规则功能（load_group_rules）。请确保已把最新代码部署/推送后再使用此功能。")
            st.stop()

        # --- Import group_rules.json (dry-run preview; does NOT write to Firebase unless you click save) ---
        st.markdown("**导入 group_rules.json（可选）**")
        st.caption("选择文件后只会在本次会话中解析与预览，不会自动写入 Firebase。需要你点击“应用/保存”按钮才会生效。")
        uploaded_group_rules = st.file_uploader(
            "选择一个 group_rules.json（或 Firebase 的备份文件）",
            type=["json"],
            key="group_rules_import_uploader",
        )
        if uploaded_group_rules is not None:
            try:
                raw_text = uploaded_group_rules.getvalue().decode("utf-8", errors="ignore")
                imported_obj = json.loads(raw_text)
                # Best-effort normalize using scheduling_logic internal helper if available
                try:
                    from scheduling_logic import _normalize_group_rules  # type: ignore
                    imported_obj = _normalize_group_rules(imported_obj)  # type: ignore[misc]
                except Exception:
                    pass
                st.session_state["_imported_group_rules_preview"] = imported_obj
            except Exception as e:
                st.session_state.pop("_imported_group_rules_preview", None)
                st.error(f"导入失败：{e}")

        preview_obj = st.session_state.get("_imported_group_rules_preview")
        if isinstance(preview_obj, dict) and isinstance(preview_obj.get("groups", None), list):
            groups_preview = preview_obj.get("groups") or []
            st.success(f"已解析：{len(groups_preview)} 个小组。")
            if groups_preview:
                names = [g.get("name") for g in groups_preview if isinstance(g, dict) and g.get("name")]
                if names:
                    st.caption("预览（前 12 个小组名）：" + "、".join([str(x) for x in names[:12]]))

            import_cols = st.columns([1, 1, 2])
            with import_cols[0]:
                if st.button("应用到当前会话", type="secondary", key="apply_imported_group_rules"):
                    st.session_state.group_rules = preview_obj
                    st.toast("已应用导入的小组规则到当前会话（未写入 Firebase）。")
                    st.session_state.initialized = False
                    st.rerun()
            with import_cols[1]:
                if st.button("应用并保存到 Firebase", type="primary", key="apply_and_save_imported_group_rules"):
                    st.session_state.group_rules = preview_obj
                    save_group_rules(st.session_state.group_rules)
                    st.toast("✅ 已导入并保存到 Firebase。")
                    st.session_state.initialized = False
                    st.rerun()
            with import_cols[2]:
                st.caption("说明：保存时会进行 schema 规范化；无效规则段（如 start/end 为 None）不会写回。")

        def _reset_group_edit_widgets():
            """
            When switching the selected group, we must clear the edit widget keys.
            Otherwise Streamlit will reuse previous widget state and the UI appears "not refreshed".
            """
            # Clear any previously created per-group edit widgets
            for k in list(st.session_state.keys()):
                if k.startswith("edit_group_ui__") or k.startswith("confirm_delete_group_ui__"):
                    del st.session_state[k]

        # Refresh from Firebase
        cols = st.columns([1, 1, 2])
        with cols[0]:
            if st.button("🔄 从Firebase刷新小组规则"):
                st.session_state.group_rules = load_group_rules()
                st.toast("已刷新小组规则。")
        with cols[1]:
            if st.button("💾 保存小组规则到Firebase", type="primary"):
                save_group_rules(st.session_state.group_rules)
                st.toast("小组规则已保存到 Firebase。")

        # Diagnostics (helps when different deployments / Firebase envs appear inconsistent)
        with st.expander("诊断：Firebase 读取到的小组规则（只读）", expanded=False):
            try:
                proj = None
                try:
                    proj = st.secrets.get("firebase", {}).get("service_account", {}).get("project_id")
                except Exception:
                    proj = None
                if proj:
                    st.caption(f"Firebase project_id: {proj}")

                raw = fm.get_data("group_rules")
                if raw is None:
                    st.warning("fm.get_data('group_rules') 返回 None（Firebase 中该路径可能为空/无权限/连接异常）。")
                else:
                    st.caption(f"fm.get_data('group_rules') 类型：{type(raw).__name__}")
                    if isinstance(raw, dict):
                        st.caption(f"keys: {list(raw.keys())}")
                        st.caption(f"updated_at: {raw.get('updated_at')}")
                        gs = raw.get("groups") or []
                        st.caption(f"groups 数量: {len(gs) if isinstance(gs, list) else 'N/A'}")
                    st.json(raw)

                # Storage backup check
                try:
                    backup = None
                    if hasattr(fm, "get_json_from_storage"):
                        backup = fm.get_json_from_storage("config/group_rules.json")
                    if backup is None:
                        st.warning("Storage 备份读取结果：None（可能 bucket 名称不匹配或无权限）。")
                    else:
                        st.success("Storage 备份读取成功：config/group_rules.json")
                        if isinstance(backup, dict):
                            st.caption(f"backup keys: {list(backup.keys())}")
                            st.caption(f"backup updated_at: {backup.get('updated_at')}")
                            bg = backup.get('groups') or []
                            st.caption(f"backup groups 数量: {len(bg) if isinstance(bg, list) else 'N/A'}")
                        st.json(backup)
                except Exception as e:
                    st.error(f"Storage 备份读取异常：{e}")
            except Exception as e:
                st.error(f"诊断读取失败：{e}")

        group_rules = st.session_state.get("group_rules") or group_rules_default
        groups = group_rules.get("groups", [])
        # --- Rule type labels (routine/task) ---
        # Choose Chinese-friendly names while keeping stored values stable: "routine" | "task".
        _GROUP_RULE_TYPE_LABELS = {
            "routine": "例行工作（Routine）",
            "task": "临时任务（Task）",
        }
        _GROUP_RULE_TYPE_HELP = (
            "例行工作：需要专注、耗时较长的日常办公工作（后续会支持只有特定员工可同时承担多项例行工作）。\n\n"
            "临时任务：碎片化但重要的小事，通常办公室时间内完成，组内被标记成员一般都可同时处理。"
        )

        # --- Validate group coverage based on imported "total sheet" (availability) ---
        st.subheader("验证小组需求（基于已导入的总表）")
        if not groups:
            st.info("暂无小组可验证。请先创建并保存小组规则。")
        elif not st.session_state.get("availability"):
            st.warning("当前还没有导入总表（availability）。请先在侧边栏导入主更表。")
        else:
            name_to_group2 = {g.get("name"): g for g in groups if g.get("name")}
            sel_name = st.selectbox("选择要验证的小组", options=list(name_to_group2.keys()), key="validate_group_name")
            # UI simplified: week selector + clickable grid + always-on detail panel
            gsel = name_to_group2.get(sel_name)

            # Persist last validation result in session_state so widget interactions won't wipe the UI.
            if st.button("开始验证", type="primary", key="run_validate_group"):
                if not gsel:
                    st.error("未选择有效小组。")
                else:
                    with st.spinner("正在按 60 分钟时段校验覆盖..."):
                        summary_df, deficits_df, all_checked_df = validate_group_coverage_from_availability(
                            st.session_state.availability,
                            gsel,
                            group_rules=st.session_state.get("group_rules") or group_rules_default,
                            step_minutes=60,
                        )
                    # Build week bins from imported dates
                    date_keys = []
                    try:
                        date_keys = sorted(list(set([str(x) for x in all_checked_df.get("date", []).tolist()])))
                    except Exception:
                        date_keys = sorted(list(set([str(x) for x in (st.session_state.availability or {}).keys()])))
                    week_bins = build_week_bins_from_dates(date_keys)
                    # Default to first bin
                    if week_bins:
                        st.session_state["validate_week_bin_idx"] = 0
                    st.session_state["_validate_group_last_result"] = {
                        "group_name": sel_name,
                        "step_minutes": 60,
                        "summary_df": summary_df,
                        "deficits_df": deficits_df,
                        "all_checked_df": all_checked_df,
                        "week_bins": week_bins,
                        "computed_at": datetime.now().isoformat(timespec="seconds"),
                    }

            # Render from last result (if it matches current selected group)
            last = st.session_state.get("_validate_group_last_result")
            if not gsel:
                st.info("请选择一个有效小组，然后点击“开始验证”。")
            elif not isinstance(last, dict) or last.get("group_name") != sel_name:
                st.info("请点击“开始验证”生成结果后，再进行热力网格/单格明细查看。")
            else:
                summary_df = last.get("summary_df")
                deficits_df = last.get("deficits_df")
                all_checked_df = last.get("all_checked_df")
                step_minutes = int(last.get("step_minutes") or 30)
                week_bins = last.get("week_bins") or []

                # Defensive: ensure dataframes exist
                if not isinstance(all_checked_df, pd.DataFrame) or all_checked_df.empty:
                    st.info("暂无可展示结果（可能规则段为空或导入日期为空）。")
                else:
                    has_deficit = isinstance(deficits_df, pd.DataFrame) and (not deficits_df.empty)
                    if has_deficit:
                        approx_hours = len(deficits_df) * (step_minutes / 60.0)
                        st.warning(
                            f"⚠️ 小组「{sel_name}」存在缺口时段（{step_minutes}min/格）：{len(deficits_df)} 条（约 {approx_hours:.1f} 小时）"
                        )
                    else:
                        st.success(f"✅ 小组「{sel_name}」在当前总表日期范围内：所有规则段均满足（无缺口）。")

                    if not week_bins:
                        st.info("无法生成周分段（日期解析失败或导入日期为空）。")
                    else:
                        labels = [b.get("label") for b in week_bins]
                        idx = st.selectbox(
                            "选择时间范围（每 7 天一段）",
                            options=list(range(len(labels))),
                            format_func=lambda i: labels[i],
                            key="validate_week_bin_idx",
                        )
                        wb = week_bins[int(idx)]
                        week_start = datetime.fromisoformat(str(wb["start_date"])).date() if isinstance(wb.get("start_date"), str) else wb.get("start_date")
                        week_end = datetime.fromisoformat(str(wb["end_date"])).date() if isinstance(wb.get("end_date"), str) else wb.get("end_date")

                        # Title + subtitle (group name + date range + members/backups)
                        group_name = str(gsel.get("name") or "").strip()
                        start_label = week_start.strftime("%d/%m/%Y") if week_start else ""
                        end_label = week_end.strftime("%d/%m/%Y") if week_end else ""
                        members_list = [str(m) for m in (gsel.get("members") or []) if str(m).strip()]
                        members_label = ", ".join(members_list) if members_list else "无"
                        backups_list = [str(m) for m in (gsel.get("backup_members") or []) if str(m).strip()]
                        backups_label = ", ".join(backups_list) if backups_list else "无"
                        st.markdown(
                            f"""
                            <div style="font-size: 20px; font-weight: 600; margin-top: 8px;">
                              {group_name}: {start_label} - {end_label}
                            </div>
                            <div style="font-size: 14px; color: #6b7280; margin-bottom: 6px;">
                              员工：{members_label}<br/>
                              后备：{backups_label}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        grid_df = build_week_grid_df(
                            all_checked_df=all_checked_df,
                            week_start=week_start,
                            week_end=week_end,
                            step_minutes=step_minutes,
                        )

                        # default selected cell within this week (first deficit -> first ok -> first)
                        def _pick_default_cell() -> Tuple[str, str]:
                            sub = grid_df[grid_df["status"] != "na"].copy()
                            if sub.empty:
                                return (week_start.isoformat(), "00:00")
                            d1 = sub[sub["status"] == "deficit"]
                            if not d1.empty:
                                r = d1.iloc[0]
                                return (str(r["date"]), str(r["time"]))
                            r = sub.iloc[0]
                            return (str(r["date"]), str(r["time"]))

                        cur_sel = st.session_state.get("_validate_selected_cell")
                        if not isinstance(cur_sel, dict):
                            cur_sel = {}
                        sel_date = str(cur_sel.get("date") or "")
                        sel_time = str(cur_sel.get("time") or "")
                        in_week = False
                        try:
                            sd = datetime.fromisoformat(sel_date).date()
                            in_week = (sd >= week_start) and (sd <= week_end)
                        except Exception:
                            in_week = False
                        if (not in_week) or (not sel_time):
                            d0, t0 = _pick_default_cell()
                            st.session_state["_validate_selected_cell"] = {"date": d0, "time": t0}
                            sel_date, sel_time = d0, t0

                        # Render clickable chart if Altair is available; otherwise fallback table.
                        st.caption("点击热力图任意一格，下方明细会自动切换到该格对应的日期+时间。")
                        if alt is not None and not grid_df.empty:
                            sel_param = alt.selection_point(fields=["date", "time"], on="click", empty=False, name="cell")
                            time_sort = sorted(grid_df["time"].unique())
                            chart_top = (
                                alt.Chart(grid_df)
                                .mark_rect(opacity=0)
                                .encode(
                                    x=alt.X(
                                        "weekday:N",
                                        sort=["周一", "周二", "周三", "周四", "周五", "周六", "周日"],
                                        title=None,
                                        axis=alt.Axis(orient="top", labelAngle=0),
                                    ),
                                    y=alt.Y("time:N", sort=time_sort, title=None, axis=None),
                                )
                                .properties(height=30)
                            )
                            chart_main = (
                                alt.Chart(grid_df)
                                .mark_rect()
                                .encode(
                                    x=alt.X(
                                        "weekday:N",
                                        sort=["周一", "周二", "周三", "周四", "周五", "周六", "周日"],
                                        title=None,
                                        axis=alt.Axis(labelAngle=0),
                                    ),
                                    y=alt.Y("time:N", sort=time_sort, title=None),
                                    color=alt.Color(
                                        "status:N",
                                        scale=alt.Scale(domain=["na", "ok", "deficit"], range=["#f3f4f6", "#d9f2d9", "#f8d7da"]),
                                        legend=None,
                                    ),
                                    tooltip=[
                                        alt.Tooltip("date:N", title="日期"),
                                        alt.Tooltip("weekday:N", title="周几"),
                                        alt.Tooltip("time:N", title="时间格"),
                                        alt.Tooltip("required:Q", title="required"),
                                        alt.Tooltip("staffed:Q", title="staffed"),
                                        alt.Tooltip("shortage:Q", title="shortage"),
                                    ],
                                )
                                .add_params(sel_param)
                                .properties(height=720)
                            )
                            # attempt to get selection payload from Streamlit (version-dependent)
                            try:
                                st.altair_chart(chart_top, use_container_width=True, key="validate_group_week_heatmap_top")
                                evt = st.altair_chart(chart_main, use_container_width=True, on_select="rerun", key="validate_group_week_heatmap")
                                got = extract_date_time_from_obj(evt)
                                if got:
                                    st.session_state["_validate_selected_cell"] = {"date": got[0], "time": got[1]}
                                    sel_date, sel_time = got[0], got[1]
                            except TypeError:
                                # older Streamlit: no on_select support
                                st.altair_chart(chart_top, use_container_width=True, key="validate_group_week_heatmap_top")
                                st.altair_chart(chart_main, use_container_width=True)
                        else:
                            # fallback
                            st.dataframe(
                                grid_df.pivot_table(index="time", columns="weekday", values="status", aggfunc="first"),
                                width="stretch",
                                height=720,
                            )
                            st.caption("提示：当前环境不支持点击热力图取值（Altair 或 on_select 不可用）。如需联动，请升级 Streamlit 或安装 Altair。")

                        # Detail panel (always visible)
                        st.subheader("明细")
                        st.caption(f"当前选择：{sel_date} {sel_time}（{step_minutes}min/格）")
                        row0 = None
                        try:
                            row0 = all_checked_df[(all_checked_df["date"] == sel_date) & (all_checked_df["time"] == sel_time)].head(1)
                        except Exception:
                            row0 = None
                        if isinstance(row0, pd.DataFrame) and (not row0.empty):
                            r0 = row0.iloc[0].to_dict()
                            st.caption(
                                f"该格校验结果：required={int(r0.get('required') or 0)} / staffed={int(r0.get('staffed') or 0)} / shortage={int(r0.get('shortage') or 0)}"
                            )
                        detail_df = build_cell_member_detail_df(
                            availability=st.session_state.availability,
                            group=gsel,
                            group_rules=st.session_state.get("group_rules") or group_rules_default,
                            date_key=sel_date,
                            time_hhmm=sel_time,
                            step_minutes=step_minutes,
                        )
                        # Render detail with availability-style colors (DataFrame / Styler)
                        try:
                            cell_map = (st.session_state.availability or {}).get(sel_date, {}) or {}
                        except Exception:
                            cell_map = {}

                        def _status_css(s: Any) -> str:
                            v = str(s or "")
                            if v == "到岗":
                                return "background-color: #d9f2d9; color: #111827;"
                            if v == "未到岗":
                                return "background-color: #f8d7da; color: #111827;"
                            if v == "请假":
                                return "background-color: #fff3cd; color: #111827;"
                            if v == "无优先级":
                                return "background-color: #e2e8f0; color: #111827;"
                            return ""

                        def _raw_css_for_member(member: Any) -> str:
                            try:
                                cell = cell_map.get(str(member))
                            except Exception:
                                cell = None
                            return availability_cell_css(cell)

                        if isinstance(detail_df, pd.DataFrame) and (not detail_df.empty) and ("成员" in detail_df.columns):
                            styler = detail_df.style
                            if "状态" in detail_df.columns:
                                styler = styler.applymap(_status_css, subset=["状态"])
                            if "明细" in detail_df.columns:
                                # colorize "明细" using the imported availability cell colors
                                styler = styler.apply(lambda r: [_raw_css_for_member(r.get("成员"))], axis=1, subset=["明细"])
                            st.dataframe(styler, width="stretch", height=320)
                        else:
                            st.dataframe(detail_df, width="stretch", height=320)

                        with st.expander("高级：查看缺口明细/按日期汇总", expanded=False):
                            if isinstance(summary_df, pd.DataFrame):
                                st.markdown("**按日期汇总**")
                                st.dataframe(summary_df, width="stretch", height=220)
                            if isinstance(deficits_df, pd.DataFrame):
                                st.markdown("**缺口明细（仅缺口）**")
                                st.dataframe(deficits_df, width="stretch", height=360)

            # Explicit save hint for imported availability
            st.caption("提示：侧边栏导入总表只会更新本次会话内的数据；如需写入 Firebase，请点击侧边栏的 “Save All Changes”。")

        st.caption("说明：小组规则用于校验排班是否满足“某时段最少需要多少人值更”。此处按“30 分钟时段”进行覆盖校验与可视化。")

        # Overview
        if groups:
            st.markdown("**概览（点击“成员/备选”可展开查看）**")
            header_cols = st.columns([2, 2, 4, 1, 1, 1])
            header_cols[0].markdown("**名称**")
            header_cols[1].markdown("**类型**")
            header_cols[2].markdown("**成员/备选**")
            header_cols[3].markdown("**成员数**")
            header_cols[4].markdown("**备选数**")
            header_cols[5].markdown("**规则段数**")

            for g in groups:
                name = g.get("name")
                rt = str(g.get("rule_type") or "routine").strip().lower()
                rt = rt if rt in _GROUP_RULE_TYPE_LABELS else "routine"
                rt_label = _GROUP_RULE_TYPE_LABELS.get(rt, rt)
                members = g.get("members", []) or []
                backups = g.get("backup_members", []) or []
                rules = g.get("requirements_windows", []) or []
                member_count = len(members)
                backup_count = len(backups)
                rules_count = len(rules)

                row_cols = st.columns([2, 2, 4, 1, 1, 1], vertical_alignment="center")
                with row_cols[0]:
                    st.write(name)
                with row_cols[1]:
                    st.caption(rt_label)
                with row_cols[2]:
                    with st.expander(f"成员/备选（{member_count}/{backup_count}）", expanded=False):
                        if members:
                            st.write("成员：" + "、".join(members))
                        else:
                            st.caption("成员：（无）")
                        if backups:
                            st.write("备选：" + "、".join(backups))
                        else:
                            st.caption("备选：（无）")
                row_cols[3].write(member_count)
                row_cols[4].write(backup_count)
                row_cols[5].write(rules_count)
        else:
            st.info("当前还没有任何小组。你可以在下面创建一个。")

        employee_names = [e.name for e in st.session_state.employees]

        st.subheader("创建新小组")
        with st.form("create_group_form", clear_on_submit=True):
            new_name = st.text_input("小组名称（必填）")
            new_desc = st.text_input("备注/说明（可选）")
            new_rule_type_label = st.selectbox(
                "规则类型（必选）",
                options=[_GROUP_RULE_TYPE_LABELS["routine"], _GROUP_RULE_TYPE_LABELS["task"]],
                index=0,
                help=_GROUP_RULE_TYPE_HELP,
            )
            new_rule_type = "routine" if new_rule_type_label == _GROUP_RULE_TYPE_LABELS["routine"] else "task"
            new_active = st.checkbox("启用", value=True)
            new_headcount = st.number_input("规划人数（可选）", min_value=0, value=0, step=1)
            new_members = st.multiselect("成员（从现有员工中选择）", options=employee_names, default=[])
            new_backup_members = st.multiselect(
                "备选成员（从现有员工中选择）",
                options=[e for e in employee_names if e not in new_members],
                default=[],
            )
            st.caption("同一员工不可同时出现在成员与备选中。")

            st.markdown("规则段（可多段）：每一段表示在该时间窗内，每个小时至少需要多少名成员在岗。")
            st.caption("day_type 建议：all=每天；mon..sun=周一..周日。start/end 为 30 分钟刻度，end 可选 24:00。")
            default_windows_df = pd.DataFrame([{"day_type": "all", "start": "00:00", "end": "24:00", "min_staff": 1}])
            # Include any existing values (if rerun keeps state) so editor won't blank them out,
            # but validation will still require selections to be from base options.
            day_opts = list(dict.fromkeys(day_type_options_base + [str(x).strip().lower() for x in default_windows_df.get("day_type", []) if str(x).strip()]))
            start_opts = list(dict.fromkeys(time_options_base + [str(x).strip() for x in default_windows_df.get("start", []) if str(x).strip()]))
            end_opts = list(dict.fromkeys(time_options_base + [str(x).strip() for x in default_windows_df.get("end", []) if str(x).strip()]))
            win_df = st.data_editor(
                default_windows_df,
                num_rows="dynamic",
                width="stretch",
                hide_index=True,
                column_config={
                    "day_type": st.column_config.SelectboxColumn("day_type", options=day_opts, required=True, help="all=每天；mon..sun=周一..周日（兼容 weekday/weekend）。"),
                    "start": st.column_config.SelectboxColumn("start", options=start_opts, required=True, help="开始时间（30 分钟刻度）。"),
                    "end": st.column_config.SelectboxColumn("end", options=end_opts, required=True, help="结束时间（30 分钟刻度；可选 24:00）。"),
                    "min_staff": st.column_config.NumberColumn("min_staff", min_value=0, step=1, required=True, help="该时间窗内，每小时最少在岗人数。"),
                },
                key="new_group_windows",
            )

            submitted = st.form_submit_button("创建小组")
            if submitted:
                if not new_name.strip():
                    st.error("小组名称不能为空。")
                else:
                    # Prevent duplicate names
                    if any(g.get("name") == new_name.strip() for g in groups):
                        st.error("已存在同名小组，请换一个名称。")
                    else:
                        windows, win_errors = validate_and_build_windows_df(win_df)
                        if win_errors:
                            st.error("规则段存在问题，请修正后再提交：\n\n- " + "\n- ".join(win_errors))
                            st.stop()

                        name_lookup = {str(n).strip(): n for n in employee_names if str(n).strip()}
                        primary_members = [name_lookup.get(str(m).strip()) for m in new_members]
                        primary_members = [m for m in primary_members if m]
                        primary_members = list(dict.fromkeys(primary_members))
                        backup_members = [name_lookup.get(str(m).strip()) for m in new_backup_members]
                        backup_members = [m for m in backup_members if m]
                        backup_members = list(dict.fromkeys(backup_members))
                        overlap = sorted(set(primary_members) & set(backup_members))
                        if overlap:
                            st.error("成员与备选不能重复：" + "、".join(overlap))
                            st.stop()
                        backup_members = [m for m in backup_members if m not in primary_members]

                        new_group = {
                            "id": uuid.uuid4().hex,
                            "name": new_name.strip(),
                            "description": new_desc.strip(),
                            "rule_type": new_rule_type,
                            "active": bool(new_active),
                            "headcount_planned": int(new_headcount) if new_headcount else None,
                            "members": primary_members,
                            "backup_members": backup_members,
                            "requirements_windows": windows,
                        }
                        group_rules.setdefault("groups", []).append(new_group)
                        st.session_state.group_rules = group_rules
                        save_group_rules(st.session_state.group_rules)
                        st.toast(f"✅ 小组“{new_name.strip()}”已创建并保存。")
                        st.session_state.initialized = False
                        st.rerun()

        st.subheader("编辑/删除现有小组")
        if groups:
            name_to_group = {g.get("name"): g for g in groups if g.get("name")}

            # If we need to update the selected group programmatically (e.g. after rename/delete),
            # do it BEFORE the selectbox is instantiated to avoid StreamlitAPIException.
            pending_key = "_pending_selected_group_name"
            if pending_key in st.session_state:
                st.session_state["selected_group_name"] = st.session_state[pending_key]
                del st.session_state[pending_key]

            selected_group_name = st.selectbox(
                "选择小组",
                options=list(name_to_group.keys()),
                key="selected_group_name",
                on_change=_reset_group_edit_widgets,
            )
            g = name_to_group.get(selected_group_name)

            if g:
                gid = str(g.get("id") or g.get("name") or "unknown")
                key_prefix = f"edit_group_ui__{gid}__"
                edit_cols = st.columns([2, 2])
                with edit_cols[0]:
                    edited_name = st.text_input("小组名称", value=g.get("name", ""), key=f"{key_prefix}name")
                    edited_desc = st.text_input("备注/说明", value=g.get("description", ""), key=f"{key_prefix}desc")
                    cur_rt = str(g.get("rule_type") or "routine").strip().lower()
                    if cur_rt not in _GROUP_RULE_TYPE_LABELS:
                        cur_rt = "routine"
                    edited_rule_type_label = st.selectbox(
                        "规则类型",
                        options=[_GROUP_RULE_TYPE_LABELS["routine"], _GROUP_RULE_TYPE_LABELS["task"]],
                        index=0 if cur_rt == "routine" else 1,
                        key=f"{key_prefix}rule_type",
                        help=_GROUP_RULE_TYPE_HELP,
                    )
                    edited_rule_type = "routine" if edited_rule_type_label == _GROUP_RULE_TYPE_LABELS["routine"] else "task"
                    edited_active = st.checkbox("启用", value=bool(g.get("active", True)), key=f"{key_prefix}active")
                    edited_headcount = st.number_input(
                        "规划人数（可选）",
                        min_value=0,
                        value=int(g.get("headcount_planned") or 0),
                        step=1,
                        key=f"{key_prefix}headcount",
                    )
                    default_members = [m for m in (g.get("members") or []) if m in employee_names]
                    default_backups = [
                        m
                        for m in (g.get("backup_members") or [])
                        if m in employee_names and m not in default_members
                    ]
                    edited_members = st.multiselect(
                        "成员（从现有员工中选择）",
                        options=employee_names,
                        default=default_members,
                        key=f"{key_prefix}members",
                    )
                    backup_options = [e for e in employee_names if e not in edited_members]
                    default_backups = [m for m in default_backups if m in backup_options]
                    edited_backup_members = st.multiselect(
                        "备选成员（从现有员工中选择）",
                        options=backup_options,
                        default=default_backups,
                        key=f"{key_prefix}backup_members",
                    )
                    st.caption("同一员工不可同时出现在成员与备选中。")

                with edit_cols[1]:
                    windows_df = pd.DataFrame(g.get("requirements_windows") or [])
                    if windows_df.empty:
                        windows_df = pd.DataFrame([{"day_type": "all", "start": "00:00", "end": "24:00", "min_staff": 1}])
                    windows_df, dropped_bad = normalize_windows_df_for_editor(windows_df)
                    if dropped_bad:
                        st.caption(f"已自动忽略 {dropped_bad} 行无效规则段（start/end 为空或为 None）。保存后这些无效行也不会写回。")
                    # Include any existing values so the editor can display legacy data,
                    # but validator will still enforce base options on save.
                    existing_day = [str(x).strip().lower() for x in windows_df.get("day_type", []) if str(x).strip()]
                    existing_start = [str(x).strip() for x in windows_df.get("start", []) if str(x).strip()]
                    existing_end = [str(x).strip() for x in windows_df.get("end", []) if str(x).strip()]
                    day_opts = list(dict.fromkeys(day_type_options_base + existing_day))
                    start_opts = list(dict.fromkeys(time_options_base + existing_start))
                    end_opts = list(dict.fromkeys(time_options_base + existing_end))
                    edited_windows_df = st.data_editor(
                        windows_df,
                        num_rows="dynamic",
                        width="stretch",
                        hide_index=True,
                        column_config={
                            "day_type": st.column_config.SelectboxColumn("day_type", options=day_opts, required=True, help="all=每天；mon..sun=周一..周日（兼容 weekday/weekend）。"),
                            "start": st.column_config.SelectboxColumn("start", options=start_opts, required=True, help="开始时间（30 分钟刻度）。"),
                            "end": st.column_config.SelectboxColumn("end", options=end_opts, required=True, help="结束时间（30 分钟刻度；可选 24:00）。"),
                            "min_staff": st.column_config.NumberColumn("min_staff", min_value=0, step=1, required=True, help="该时间窗内，每小时最少在岗人数。"),
                        },
                        key=f"{key_prefix}windows",
                    )

                action_cols = st.columns([1, 1, 2])
                with action_cols[0]:
                    if st.button("保存该小组修改", type="primary"):
                        # Validate rename collisions
                        new_name_norm = edited_name.strip()
                        if not new_name_norm:
                            st.error("小组名称不能为空。")
                        elif new_name_norm != g.get("name") and any(x.get("name") == new_name_norm for x in groups):
                            st.error("已存在同名小组，请换一个名称。")
                        else:
                            new_windows, win_errors = validate_and_build_windows_df(edited_windows_df)
                            if win_errors:
                                st.error("规则段存在问题，请修正后再保存：\n\n- " + "\n- ".join(win_errors))
                                st.stop()

                            g["name"] = new_name_norm
                            g["description"] = edited_desc.strip()
                            g["rule_type"] = edited_rule_type
                            g["active"] = bool(edited_active)
                            g["headcount_planned"] = int(edited_headcount) if edited_headcount else None
                            name_lookup = {str(n).strip(): n for n in employee_names if str(n).strip()}
                            members_clean = [name_lookup.get(str(m).strip()) for m in edited_members]
                            members_clean = [m for m in members_clean if m]
                            members_clean = list(dict.fromkeys(members_clean))
                            backups_clean = [name_lookup.get(str(m).strip()) for m in edited_backup_members]
                            backups_clean = [m for m in backups_clean if m]
                            backups_clean = list(dict.fromkeys(backups_clean))
                            overlap = sorted(set(members_clean) & set(backups_clean))
                            if overlap:
                                st.error("成员与备选不能重复：" + "、".join(overlap))
                                st.stop()
                            backups_clean = [m for m in backups_clean if m not in members_clean]
                            g["members"] = members_clean
                            g["backup_members"] = backups_clean
                            g["requirements_windows"] = new_windows

                            st.session_state.group_rules = group_rules
                            save_group_rules(st.session_state.group_rules)
                            st.toast("✅ 已保存小组修改到 Firebase。")
                            # If renamed, keep selection in sync
                            st.session_state["_pending_selected_group_name"] = new_name_norm
                            st.session_state.initialized = False
                            st.rerun()

                with action_cols[1]:
                    confirm_delete = st.checkbox(
                        "确认删除",
                        value=False,
                        key=f"confirm_delete_group_ui__{gid}",
                    )
                    if st.button("删除该小组", type="secondary", disabled=not confirm_delete):
                        group_rules["groups"] = [x for x in group_rules.get("groups", []) if x.get("id") != g.get("id")]
                        st.session_state.group_rules = group_rules
                        save_group_rules(st.session_state.group_rules)
                        st.toast("🗑️ 小组已删除并保存到 Firebase。")
                        # After delete, reset selection to the first group (if any)
                        remaining = [x.get("name") for x in group_rules.get("groups", []) if x.get("name")]
                        if remaining:
                            st.session_state["_pending_selected_group_name"] = remaining[0]
                        elif "selected_group_name" in st.session_state:
                            del st.session_state["selected_group_name"]
                        st.session_state.initialized = False
                        st.rerun()
