from __future__ import annotations

import html
import sys
from pathlib import Path

import streamlit as st  # type: ignore[import-not-found]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import extract_uploaded_text, render_global_styles, render_hero, render_skill_chips
from job_sources import (
    JobSourceError,
    configured_job_sources,
    fetch_jsearch_jobs,
    fetch_remotive_jobs,
    live_jobs_to_postings,
)
from resume_analyzer import format_report, infer_resume_job_queries, rank_jobs_for_resume


def resolve_resume_text(uploaded_resume, pasted_text: str) -> tuple[str, str | None]:
    if uploaded_resume is not None:
        extracted = extract_uploaded_text(uploaded_resume)
        if extracted.strip():
            return extracted, f"Using text extracted from {uploaded_resume.name}."
    return pasted_text, None


def render_source_status() -> None:
    sources = configured_job_sources()
    st.caption(
        "Job sources: "
        f"Remotive {'available' if sources['remotive'] else 'unavailable'}; "
        f"RapidAPI JSearch {'environment key configured' if sources['rapidapi_jsearch'] else 'paste key in page or set env'}; "
        f"Adzuna {'configured' if sources['adzuna'] else 'not configured'}."
    )


def render_job_card(item: dict[str, object], live_job) -> None:
    with st.container(border=True):
        title = html.escape(str(item["title"]))
        company = html.escape(str(item["company"] or "Company not listed"))
        source = html.escape(live_job.source if live_job else "Unknown source")
        location = html.escape(live_job.location if live_job and live_job.location else "Remote/Not specified")
        st.markdown(f"### #{item['rank']} {title}")
        st.caption(f"{company} | {location} | {source}")

        score_cols = st.columns(4)
        score_cols[0].metric("Fit Score", f"{item['final_score']}/100")
        score_cols[1].metric("Semantic", f"{item['semantic_score']}%")
        score_cols[2].metric("Skill", f"{item['skill_score']}%")
        score_cols[3].metric("Method", str(item["analysis"]["scoring_method"]).replace("_", " ").title())

        chip_cols = st.columns(2)
        with chip_cols[0]:
            render_skill_chips("Matched", item["matched_skills"], "good")
        with chip_cols[1]:
            render_skill_chips("Missing", item["missing_skills"], "bad")

        st.write(item["summary"])
        if live_job and live_job.url:
            st.link_button("Open Job Posting", live_job.url)
        with st.expander("Why this job matched"):
            st.markdown(format_report(item["analysis"]))


def main() -> None:
    st.set_page_config(page_title="Live Job Finder", layout="wide")
    render_global_styles()
    render_hero(
        "Live Job Finder",
        "Upload one resume, search live jobs, and get ranked opportunities with clear match explanations.",
    )

    with st.sidebar:
        st.header("Search Settings")
        render_source_status()
        provider = st.radio(
            "Live job source",
            ["JSearch via RapidAPI", "Remotive"],
            help="JSearch can return jobs from multiple sources. Remotive is no-key fallback but only returns Remotive jobs.",
        )
        rapidapi_key = ""
        country = "in"
        date_posted = "month"
        if provider == "JSearch via RapidAPI":
            rapidapi_key = st.text_input(
                "RapidAPI Key",
                type="password",
                help="Paste your RapidAPI key for the JSearch API. If blank, the app uses RAPIDAPI_KEY from your environment.",
            )
            country = st.selectbox("Country", ["in", "us", "gb", "ca", "au", "de", "sg"], index=0)
            date_posted = st.selectbox("Date posted", ["all", "today", "3days", "week", "month"], index=4)

        result_limit = st.number_input("Live jobs to fetch", min_value=5, max_value=100, value=25, step=5)
        top_n = st.number_input("Top ranked jobs to show", min_value=1, max_value=25, value=10)

    uploaded_resume = st.file_uploader("Upload resume (.txt, .pdf, .docx)", type=["txt", "pdf", "docx"])
    pasted_resume = st.text_area("Or paste resume text", height=220)
    resume_text, notice = resolve_resume_text(uploaded_resume, pasted_resume)
    if notice:
        st.info(notice)

    if resume_text.strip():
        with st.expander("Extracted Resume Text Preview"):
            st.write(resume_text[:3000] + ("..." if len(resume_text) > 3000 else ""))

    suggested_queries = infer_resume_job_queries(resume_text) if resume_text.strip() else ["software developer"]
    query_col, action_col = st.columns([2, 1])
    with query_col:
        selected_query = st.selectbox("Suggested search query", suggested_queries)
        query = st.text_input("Edit search query", value=selected_query)
    with action_col:
        st.write("")
        st.write("")
        find_clicked = st.button("Find Matching Jobs", type="primary", use_container_width=True)

    if find_clicked:
        if not resume_text.strip():
            st.error("Please upload or paste a resume first.")
            return
        if not query.strip():
            st.error("Please provide a search query.")
            return

        try:
            if provider == "JSearch via RapidAPI":
                with st.spinner(f"Fetching live jobs for '{query}' from JSearch..."):
                    live_jobs = fetch_jsearch_jobs(
                        query,
                        rapidapi_key=rapidapi_key,
                        limit=int(result_limit),
                        country=country,
                        date_posted=date_posted,
                    )
            else:
                with st.spinner(f"Fetching live jobs for '{query}' from Remotive..."):
                    live_jobs = fetch_remotive_jobs(query, limit=int(result_limit))
        except JobSourceError as exc:
            st.error(f"Could not fetch live jobs: {exc}")
            return
        except Exception as exc:
            st.error(f"Could not connect to the job API: {exc}")
            return

        if not live_jobs:
            st.warning("No live jobs were found for this query. Try another query from the dropdown or enter your own.")
            return

        with st.spinner("Ranking live jobs against the uploaded resume..."):
            postings = live_jobs_to_postings(live_jobs)
            ranked = rank_jobs_for_resume(resume_text, postings, limit=int(top_n))

        if not ranked:
            st.warning("Jobs were fetched, but none had enough text for matching.")
            return

        job_by_url = {job.url: job for job in live_jobs}
        table_rows = []
        for item in ranked:
            live_job = job_by_url.get(str(item["url"]))
            table_rows.append(
                {
                    "rank": item["rank"],
                    "title": item["title"],
                    "company": item["company"],
                    "location": live_job.location if live_job else "",
                    "final_score": item["final_score"],
                    "semantic_score": item["semantic_score"],
                    "skill_score": item["skill_score"],
                    "scoring_method": item["analysis"]["scoring_method"],
                    "matched_skills": ", ".join(item["matched_skills"]),
                    "missing_skills": ", ".join(item["missing_skills"]),
                    "url": item["url"],
                }
            )

        st.markdown("### Best Matching Jobs")
        with st.expander("Show ranked table"):
            st.dataframe(table_rows, use_container_width=True, hide_index=True)

        for item in ranked:
            live_job = job_by_url.get(str(item["url"]))
            render_job_card(item, live_job)


if __name__ == "__main__":
    main()
