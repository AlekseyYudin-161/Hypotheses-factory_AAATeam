from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from i18n import LANGUAGE_OPTIONS, get_translations
from services.exporters import (
    build_csv,
    build_docx,
    build_feedback_csv,
    build_json,
    build_pdf,
    build_tasks_csv,
)
from services.generator import calculate_final_score, generate_hypotheses
from services.storage import (
    add_feedback,
    create_run,
    get_hypotheses,
    get_latest_run,
    get_run,
    init_db,
    list_feedback,
    save_expert_review,
    save_hypotheses,
    update_status,
)

APP_DIR = Path(__file__).resolve().parent
DB_PATH = Path(os.getenv("APP_DB_PATH", APP_DIR / "data" / "workbench.db"))

ALL_KNOWLEDGE_BASES = "all"
KNOWLEDGE_BASE_CODES = [
    "scientific_publications",
    "patents",
    "internal_reports",
    "historical_experiments",
    "materials_data",
    "process_data",
    "open_sources",
]

st.set_page_config(
    page_title="Hypothesis Workbench",
    page_icon="🧪",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top:1.15rem; padding-bottom:3rem; max-width:1500px;}
    .subtitle {color:#667085; margin-top:-.45rem; margin-bottom:1rem;}
    .score-badge {display:inline-block; border-radius:999px; padding:.28rem .62rem;
        margin-right:.35rem; margin-bottom:.25rem; font-size:.86rem; font-weight:650;}
    .novelty {background:#E1F5EE; color:#0F6E56;}
    .risk {background:#FAEEDA; color:#854F0B;}
    .value {background:#E6F1FB; color:#185FA5;}
    .economic {background:#F3E8FF; color:#6B21A8;}
    .success-probability {background:#ECFDF3; color:#027A48;}
    .status {display:inline-block; border-radius:999px; padding:.22rem .55rem;
        font-size:.82rem; font-weight:650; background:#F2F4F7; color:#475467; margin-right:.3rem;}
    .verified {background:#ECFDF3; color:#027A48;}
    .triplet {display:inline-block; font-family:ui-monospace, SFMono-Regular, Menlo, monospace;
        background:#F2F4F7; border:1px solid #D0D5DD; border-radius:8px;
        padding:.28rem .5rem; margin:.12rem .2rem .12rem 0;}
    .chain-wrap {display:flex; align-items:center; gap:.45rem; flex-wrap:wrap; margin:.5rem 0 1rem 0;}
    .chain-node {background:#F8FAFC; border:1px solid #CBD5E1; border-radius:10px;
        padding:.65rem .8rem; font-weight:600; min-width:130px; text-align:center;}
    .chain-arrow {font-size:1.35rem; color:#64748B; font-weight:700;}
    .source-card {border-left:4px solid #185FA5; padding:.6rem .75rem; margin:.5rem 0;
        background:#F8FAFC; border-radius:8px;}
    div[data-testid="stButton"] > button {border-radius:9px;}
    </style>
    """,
    unsafe_allow_html=True,
)


def init_state() -> None:
    defaults = {
        "run_id": None,
        "language": "ru",
        "w_novelty": 0.4,
        "w_risk": 0.3,
        "w_value": 0.3,
        "flash": "",
        "status_filter": "all",
        "knowledge_base_selection": [ALL_KNOWLEDGE_BASES],
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    if st.session_state["run_id"] is None:
        latest = get_latest_run(DB_PATH)
        if latest:
            st.session_state["run_id"] = latest["id"]


def normalize_knowledge_bases(selection: list[str]) -> list[str]:
    if ALL_KNOWLEDGE_BASES in selection:
        return KNOWLEDGE_BASE_CODES.copy()
    return list(dict.fromkeys(code for code in selection if code in KNOWLEDGE_BASE_CODES))


def knowledge_bases_text(codes: list[str], tr: dict[str, str]) -> str:
    if set(codes) == set(KNOWLEDGE_BASE_CODES):
        return tr["kb_all"]
    return ", ".join(tr[f"kb_{code}"] for code in codes)


def rerank(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = []
    for item in items:
        row = dict(item)
        row["final_score"] = calculate_final_score(
            row["novelty_score"],
            row["risk_score"],
            row["value_score"],
            row["economic_value_score"],
            st.session_state["w_novelty"],
            st.session_state["w_risk"],
            st.session_state["w_value"],
        )
        ranked.append(row)
    return sorted(ranked, key=lambda x: x["final_score"], reverse=True)


def status_label(status: str, tr: dict[str, str]) -> str:
    return {
        "draft": tr["draft"],
        "accepted": tr["accepted_status"],
        "rejected": tr["rejected_status"],
    }.get(status, status)


def render_causal_chain(chain: list[str]) -> None:
    if not chain:
        return
    blocks = []
    for index, node in enumerate(chain):
        blocks.append(f'<div class="chain-node">{html.escape(str(node))}</div>')
        if index < len(chain) - 1:
            blocks.append('<div class="chain-arrow">→</div>')
    st.markdown(f'<div class="chain-wrap">{"".join(blocks)}</div>', unsafe_allow_html=True)


def render_sources(evidence: list[dict[str, Any]], tr: dict[str, str]) -> None:
    if not evidence:
        st.warning(tr["no_sources"])
        return
    for source in evidence:
        st.markdown(
            f'<span class="triplet">{html.escape(str(source.get("triplet", "")))}</span>',
            unsafe_allow_html=True,
        )
        source_name = html.escape(str(source.get("source", tr["source"])))
        fragment = html.escape(str(source.get("evidence_fragment", "")))
        meta = []
        if source.get("page") is not None:
            meta.append(f"{tr['page']}: {source['page']}")
        if source.get("chunk_id"):
            meta.append(f"{tr['chunk']}: {source['chunk_id']}")
        if source.get("confidence") is not None:
            meta.append(f"{tr['confidence']}: {float(source['confidence']):.2f}")
        st.markdown(
            f'<div class="source-card"><b>{source_name}</b><br>{" · ".join(meta)}<br>{fragment}</div>',
            unsafe_allow_html=True,
        )
        url = source.get("source_url")
        if url:
            st.markdown(f"[{tr['open_source']}]({url})")
        else:
            st.error(tr["missing_link"])


def save_status(item: dict[str, Any], new_status: str, comment: str = "") -> None:
    old_status = item["status"]
    update_status(DB_PATH, item["id"], status=new_status, final_score=item["final_score"])
    add_feedback(
        DB_PATH,
        run_id=item["run_id"],
        hypothesis_id=item["id"],
        action=new_status,
        old_value=old_status,
        new_value=new_status,
        comment=comment,
    )


def render_expert_review(item: dict[str, Any], tr: dict[str, str]) -> None:
    with st.form(f"review_{item['id']}"):
        verified = st.checkbox(
            tr["hypothesis_verified"],
            value=bool(item.get("is_verified")),
        )
        rating = st.slider(
            tr["expert_rating"],
            min_value=1,
            max_value=5,
            value=int(item.get("expert_rating") or 3),
        )
        comment = st.text_area(
            tr["expert_comment"],
            value=item.get("expert_comment", ""),
            placeholder=tr["expert_comment_placeholder"],
            height=90,
        )
        if st.form_submit_button(tr["save_review"], use_container_width=True):
            old_value = json.dumps(
                {
                    "is_verified": item.get("is_verified", False),
                    "expert_rating": item.get("expert_rating"),
                    "expert_comment": item.get("expert_comment", ""),
                },
                ensure_ascii=False,
            )
            new_value = json.dumps(
                {
                    "is_verified": verified,
                    "expert_rating": rating,
                    "expert_comment": comment,
                },
                ensure_ascii=False,
            )
            save_expert_review(
                DB_PATH,
                item["id"],
                is_verified=verified,
                expert_rating=rating,
                expert_comment=comment,
            )
            add_feedback(
                DB_PATH,
                run_id=item["run_id"],
                hypothesis_id=item["id"],
                action="verified" if verified else "review_updated",
                old_value=old_value,
                new_value=new_value,
                comment=comment,
            )
            st.session_state["flash"] = tr["review_saved"]
            st.rerun()


def render_card(item: dict[str, Any], rank: int, tr: dict[str, str]) -> None:
    with st.container(border=True):
        c_rank, c_main, c_actions = st.columns([0.8, 6.4, 1.8])
        with c_rank:
            st.markdown(f"### #{rank}")
            st.metric(tr["final"], f"{item['final_score']:.2f}")
        with c_main:
            st.markdown(f"### {item['hypothesis']}")
            st.markdown(
                f'<span class="score-badge novelty">{tr["novelty"]} {item["novelty_score"]:.2f}</span>'
                f'<span class="score-badge risk">{tr["risk"]} {item["risk_score"]:.2f}</span>'
                f'<span class="score-badge value">{tr["value"]} {item["value_score"]:.2f}</span>'
                f'<span class="score-badge economic">{tr["economic_value"]} {item["economic_value_score"]:.2f}</span>'
                f'<span class="score-badge success-probability">{tr["success_probability"]} {item["success_probability_score"]:.2f}</span>',
                unsafe_allow_html=True,
            )
            verification_class = "status verified" if item.get("is_verified") else "status"
            verification_text = tr["verified_badge"] if item.get("is_verified") else tr["not_verified_badge"]
            st.markdown(
                f'<span class="status">{tr["status"]}: {status_label(item["status"], tr)}</span>'
                f'<span class="{verification_class}">{verification_text}</span>',
                unsafe_allow_html=True,
            )
        with c_actions:
            if st.button(
                tr["accept"],
                key=f"accept_{item['id']}",
                use_container_width=True,
                disabled=item["status"] == "accepted",
            ):
                save_status(item, "accepted")
                st.session_state["flash"] = tr["saved_status"]
                st.rerun()
            reason = st.text_input(
                tr["rejection_reason"],
                key=f"reason_{item['id']}",
                label_visibility="collapsed",
                placeholder=tr["rejection_reason"],
            )
            if st.button(
                tr["reject"],
                key=f"reject_{item['id']}",
                use_container_width=True,
                disabled=item["status"] == "rejected",
            ):
                save_status(item, "rejected", reason)
                st.session_state["flash"] = tr["saved_status"]
                st.rerun()

        with st.expander(tr["interpretability"], expanded=(rank == 1)):
            st.markdown(f"#### {tr['rationale']}")
            st.write(item.get("rationale", ""))
            st.markdown(f"#### {tr['influence_diagram']}")
            render_causal_chain(item.get("causal_chain", []))
            st.markdown(f"#### {tr['mechanism']}")
            st.write(item.get("mechanism", ""))
            st.markdown(f"#### {tr['expected_effect']}")
            st.write(item.get("expected_effect", ""))
            st.markdown(f"#### {tr['industrial_scale']}")
            st.write(item.get("industrial_scale", ""))
            st.markdown(f"#### {tr['kpi_link']}")
            st.write(item.get("kpi_link", ""))
            st.markdown(f"#### {tr['constraints_fit']}")
            st.write(item.get("constraints_fit", ""))

            st.markdown(f"#### {tr['score_reasons']}")
            q1, q2, q3, q4 = st.columns(4)
            q1.markdown(f"**{tr['novelty']}:** {item['novelty_why']}")
            q2.markdown(f"**{tr['risk']}:** {item['risk_why']}")
            q3.markdown(f"**{tr['value']}:** {item['value_why']}")
            q4.markdown(f"**{tr['economic_value']}:** {item['economic_value_why']}")
            st.markdown(f"**{tr['success_probability']}:** {item['success_probability_why']}")

            st.markdown(f"#### {tr['verification']}")
            st.info(item.get("verification_recommendation", ""))

            resources = item.get("resource_estimate", {})
            st.markdown(f"#### {tr['resource_estimate']}")
            r1, r2, r3 = st.columns(3)
            r1.metric(tr["time"], resources.get("time", "—"))
            r2.metric(tr["cost"], resources.get("cost", "—"))
            r3.metric(tr["volume"], resources.get("volume", "—"))

            st.markdown(f"#### {tr['sources']}")
            render_sources(item.get("evidence", []), tr)

            if item.get("roadmap"):
                st.markdown(f"#### {tr['roadmap']}")
                for step in item["roadmap"]:
                    st.markdown(
                        f"**{step.get('step', '')}. {step.get('title', '')}**  \n"
                        f"{step.get('description', '')}  \n"
                        f"{tr['resources']}: {step.get('resources', '')} · "
                        f"{tr['criterion']}: {step.get('success_criterion', '')}"
                    )

            st.markdown(f"#### {tr['expert_review']}")
            render_expert_review(item, tr)


init_db(DB_PATH)
init_state()

language_col, _ = st.columns([1.2, 4.8])
with language_col:
    current_name = next(name for name, code in LANGUAGE_OPTIONS.items() if code == st.session_state["language"])
    selected_name = st.selectbox(
        get_translations(st.session_state["language"])["language"],
        list(LANGUAGE_OPTIONS.keys()),
        index=list(LANGUAGE_OPTIONS.keys()).index(current_name),
    )
selected_language = LANGUAGE_OPTIONS[selected_name]
if selected_language != st.session_state["language"]:
    st.session_state["language"] = selected_language
    st.rerun()

tr = get_translations(st.session_state["language"])

if st.session_state["flash"]:
    st.success(st.session_state["flash"])
    st.session_state["flash"] = ""

st.title(tr["app_title"])
st.markdown(f'<div class="subtitle">{tr["subtitle"]}</div>', unsafe_allow_html=True)

workbench_tab, download_tab, feedback_tab = st.tabs(
    [tr["workbench"], tr["download"], tr["feedback"]]
)

with workbench_tab:
    with st.form("generation_form"):
        kpi = st.text_area(
            tr["kpi_label"],
            placeholder=tr["kpi_placeholder"],
            height=110,
        )
        knowledge_base_selection = st.multiselect(
            tr["knowledge_bases_label"],
            options=[ALL_KNOWLEDGE_BASES, *KNOWLEDGE_BASE_CODES],
            default=st.session_state["knowledge_base_selection"],
            format_func=lambda code: tr[f"kb_{code}"],
            help=tr["knowledge_bases_help"],
        )
        constraints = st.text_area(
            tr["constraints_label"],
            placeholder=tr["constraints_placeholder"],
            height=100,
        )
        generate = st.form_submit_button(
            tr["generate"],
            type="primary",
            use_container_width=True,
        )

    if generate:
        if not kpi.strip():
            st.warning(tr["enter_kpi"])
        elif not knowledge_base_selection:
            st.warning(tr["knowledge_bases_required"])
        else:
            st.session_state["knowledge_base_selection"] = knowledge_base_selection
            selected_knowledge_bases = normalize_knowledge_bases(knowledge_base_selection)
            try:
                with st.spinner(tr["generating"]):
                    hypotheses = generate_hypotheses(
                        kpi=kpi.strip(),
                        constraints=constraints.strip(),
                        language=st.session_state["language"],
                        knowledge_bases=selected_knowledge_bases,
                    )
                    run_id = create_run(
                        DB_PATH,
                        kpi=kpi.strip(),
                        constraints_text=constraints.strip(),
                        language=st.session_state["language"],
                        knowledge_bases=selected_knowledge_bases,
                    )
                    save_hypotheses(DB_PATH, run_id, hypotheses)
                    st.session_state["run_id"] = run_id
                st.session_state["flash"] = tr["generated"]
                st.rerun()
            except Exception as error:
                st.error(f"{tr['generation_error']}: {error}")

    run = get_run(DB_PATH, st.session_state["run_id"]) if st.session_state["run_id"] else None
    items = get_hypotheses(DB_PATH, st.session_state["run_id"]) if st.session_state["run_id"] else []

    st.markdown(f"### {tr['weights']}")
    w1, w2, w3 = st.columns(3)
    st.session_state["w_novelty"] = w1.slider(tr["weight_novelty"], 0.0, 1.0, st.session_state["w_novelty"], 0.1)
    st.session_state["w_risk"] = w2.slider(tr["weight_risk"], 0.0, 1.0, st.session_state["w_risk"], 0.1)
    st.session_state["w_value"] = w3.slider(tr["weight_value"], 0.0, 1.0, st.session_state["w_value"], 0.1)
    st.caption(tr["ranking_formula"])

    if run:
        selected_kb_text = knowledge_bases_text(run.get("knowledge_bases", []), tr)
        st.caption(
            f"{tr['current_request']}: {run['kpi']} · "
            f"{tr['knowledge_bases_selected']}: {selected_kb_text} · "
            f"{tr['constraints']}: {run['constraints_text'] or tr['not_set']} · "
            f"{tr['date']}: {run['created_at']}"
        )

    if not items:
        st.info(tr["start_hint"])
    else:
        ranked = rerank(items)
        filter_codes = ["all", "pending", "accepted", "rejected", "verified"]
        filter_labels = [tr["all"], tr["pending"], tr["accepted"], tr["rejected"], tr["verified"]]
        selected_filter_label = st.radio(
            tr["show"],
            filter_labels,
            horizontal=True,
            index=filter_codes.index(st.session_state["status_filter"]),
        )
        st.session_state["status_filter"] = filter_codes[filter_labels.index(selected_filter_label)]

        current_filter = st.session_state["status_filter"]
        if current_filter == "pending":
            ranked = [x for x in ranked if x["status"] == "draft"]
        elif current_filter == "accepted":
            ranked = [x for x in ranked if x["status"] == "accepted"]
        elif current_filter == "rejected":
            ranked = [x for x in ranked if x["status"] == "rejected"]
        elif current_filter == "verified":
            ranked = [x for x in ranked if x.get("is_verified")]

        if not ranked:
            st.warning(tr["no_results"])
        else:
            st.markdown(f"## {tr['ranked_list']}")
            for rank, item in enumerate(ranked, start=1):
                render_card(item, rank, tr)

with download_tab:
    st.subheader(tr["download_results"])
    export_run = get_run(DB_PATH, st.session_state["run_id"]) if st.session_state["run_id"] else None
    export_items = rerank(get_hypotheses(DB_PATH, st.session_state["run_id"])) if st.session_state["run_id"] else []

    if not export_run or not export_items:
        st.info(tr["start_hint"])
    else:
        scopes = [tr["all_scope"], tr["accepted_scope"], tr["rejected_scope"], tr["verified_scope"]]
        scope = st.selectbox(tr["scope"], scopes)
        selected_items = export_items
        if scope == tr["accepted_scope"]:
            selected_items = [x for x in selected_items if x["status"] == "accepted"]
        elif scope == tr["rejected_scope"]:
            selected_items = [x for x in selected_items if x["status"] == "rejected"]
        elif scope == tr["verified_scope"]:
            selected_items = [x for x in selected_items if x.get("is_verified")]

        export_format = st.selectbox(
            tr["format"],
            ["PDF", "DOCX", "CSV", "JSON", "CSV Jira / YouTrack"],
        )
        file_data: bytes | None
        if export_format == "PDF":
            try:
                file_data = build_pdf(export_run, selected_items)
                file_name, mime = "hypothesis_report.pdf", "application/pdf"
            except RuntimeError as error:
                st.error(str(error))
                file_data, file_name, mime = None, "", ""
        elif export_format == "DOCX":
            file_data = build_docx(export_run, selected_items)
            file_name = "hypothesis_report.docx"
            mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        elif export_format == "CSV":
            file_data = build_csv(selected_items)
            file_name, mime = "hypotheses.csv", "text/csv"
        elif export_format == "JSON":
            file_data = build_json(export_run, selected_items)
            file_name, mime = "hypotheses.json", "application/json"
        else:
            file_data = build_tasks_csv(export_run, selected_items)
            file_name, mime = "tasks.csv", "text/csv"

        st.caption(f"{tr['export_count']}: {len(selected_items)}")
        if file_data is not None:
            st.download_button(
                tr["download_button"],
                data=file_data,
                file_name=file_name,
                mime=mime,
                use_container_width=True,
                type="primary",
            )

with feedback_tab:
    st.subheader(tr["feedback_log"])
    feedback_rows = list_feedback(DB_PATH)
    if feedback_rows:
        st.dataframe(pd.DataFrame(feedback_rows), use_container_width=True, hide_index=True)
        st.download_button(
            tr["download_feedback"],
            data=build_feedback_csv(feedback_rows),
            file_name="expert_feedback.csv",
            mime="text/csv",
        )
        df = pd.DataFrame(feedback_rows)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(tr["total_actions"], len(df))
        c2.metric(tr["accepted_count"], int((df["action"] == "accepted").sum()))
        c3.metric(tr["rejected_count"], int((df["action"] == "rejected").sum()))
        c4.metric(tr["verified_count"], int((df["action"] == "verified").sum()))
    else:
        st.info(tr["no_feedback"])
