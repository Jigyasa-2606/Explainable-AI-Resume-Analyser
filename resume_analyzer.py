from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TRAINED_MODEL_PATH = Path(__file__).resolve().parent / "trained_match_model.joblib"

_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_PHONE_RE = re.compile(r"\+?\d[\d\s().-]{8,}\d")

_SKILL_PATTERN_CACHE: tuple[list[str], list[re.Pattern[str]]] | None = None


def sanitize_job_query_text(query: str) -> str:
    """Strip emails / phones so job APIs are not queried with contact strings."""
    q = _EMAIL_RE.sub(" ", query or "")
    q = _PHONE_RE.sub(" ", q)
    return re.sub(r"\s+", " ", q).strip()


def _line_is_bad_job_query_headline(line: str) -> bool:
    s = line.strip()
    if len(s) < 12 or len(s) > 140:
        return True
    if _EMAIL_RE.search(s):
        return True
    if _PHONE_RE.search(s):
        return True
    low = s.lower()
    if sum(ch.isalpha() for ch in s) < 8:
        return True
    noise = ("resume", "curriculum vitae", "cv", "phone", "email", "linkedin.com", "github.com")
    if any(term in low for term in noise) and len(s) < 40:
        return True
    return False


def _skill_patterns() -> tuple[list[str], list[re.Pattern[str]]]:
    global _SKILL_PATTERN_CACHE
    if _SKILL_PATTERN_CACHE is not None:
        return _SKILL_PATTERN_CACHE

    canonical = sorted(SKILL_LEXICON, key=len, reverse=True)
    patterns = []
    for skill in canonical:
        escaped = re.escape(skill.replace(" ", r"\s+"))
        patterns.append(re.compile(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", re.I))
    _SKILL_PATTERN_CACHE = (canonical, patterns)
    return _SKILL_PATTERN_CACHE


SKILL_LEXICON: frozenset[str] = frozenset(
    {
        "machine learning",
        "deep learning",
        "computer vision",
        "natural language processing",
        "nlp",
        "large language model",
        "llm",
        "generative ai",
        "data engineering",
        "data science",
        "data analyst",
        "business intelligence",
        "power bi",
        "tableau",
        "excel",
        "sql",
        "nosql",
        "postgresql",
        "mysql",
        "mongodb",
        "redis",
        "kafka",
        "spark",
        "hadoop",
        "airflow",
        "dbt",
        "snowflake",
        "bigquery",
        "etl",
        "python",
        "java",
        "kotlin",
        "scala",
        "go",
        "golang",
        "rust",
        "c++",
        "csharp",
        "c#",
        ".net",
        "dotnet",
        "ruby",
        "php",
        "swift",
        "android",
        "ios",
        "javascript",
        "typescript",
        "node",
        "nodejs",
        "react",
        "react native",
        "angular",
        "vue",
        "next.js",
        "nextjs",
        "express",
        "django",
        "flask",
        "fastapi",
        "spring boot",
        "spring",
        "graphql",
        "rest api",
        "microservices",
        "kubernetes",
        "k8s",
        "docker",
        "terraform",
        "ansible",
        "jenkins",
        "ci/cd",
        "github actions",
        "aws",
        "gcp",
        "azure",
        "linux",
        "bash",
        "shell scripting",
        "system design",
        "distributed systems",
        "tensorflow",
        "pytorch",
        "keras",
        "scikit-learn",
        "sklearn",
        "pandas",
        "numpy",
        "opencv",
        "mlops",
        "statistics",
        "a/b testing",
        "experimentation",
        "product management",
        "agile",
        "scrum",
        "git",
        "jira",
        "figma",
        "seo",
        "growth marketing",
        "salesforce",
        "stripe",
        "payment systems",
        "security",
        "penetration testing",
        "oauth",
        "jwt",
        "grpc",
        "websocket",
        "blockchain",
        "solidity",
    }
)


RELATED_SKILLS: dict[str, tuple[str, ...]] = {
    "python": ("django", "flask", "fastapi", "pandas", "numpy", "pytorch", "tensorflow"),
    "javascript": ("typescript", "react", "angular", "vue", "node"),
    "typescript": ("javascript", "react", "angular", "vue"),
    "react": ("javascript", "typescript", "next.js"),
    "java": ("spring boot", "spring", "kafka"),
    "sql": ("postgresql", "mysql", "snowflake", "bigquery"),
    "aws": ("docker", "kubernetes", "terraform"),
    "kubernetes": ("docker", "helm", "terraform"),
    "machine learning": ("python", "tensorflow", "pytorch", "pandas"),
    "nlp": ("python", "transformers", "deep learning"),
}


BIAS_TERMS_HIGH = (
    "young energetic",
    "digital native",
    "recent graduate preferred",
    "native english speaker only",
    "culture fit",
    "married",
    "single",
)

BIAS_TERMS_MEDIUM = (
    "rockstar",
    "ninja",
    "work hard play hard",
    "young team",
    "fast-paced environment",
)


@dataclass(frozen=True)
class JobPosting:
    title: str
    company: str
    description: str
    url: str = ""
    source_index: int = 0


@dataclass
class SkillMatch:
    matched: list[str]
    missing: list[str]
    partial: list[str]
    score: float
    graph_matches: list[str]
    evidence: dict[str, str]


_MODEL_ARTIFACT: dict[str, Any] | None = None


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _tokenize(text: str) -> list[str]:
    lowered = (text or "").lower()
    lowered = re.sub(r"[^a-z0-9+#.\s]+", " ", lowered)
    return [tok for tok in lowered.split() if len(tok) > 1]


def extract_skills(text: str) -> list[str]:
    body = (text or "").lower()
    canonical, patterns = _skill_patterns()
    found: list[str] = []
    seen: set[str] = set()
    for skill, pattern in zip(canonical, patterns, strict=False):
        if skill in seen:
            continue
        if pattern.search(body):
            found.append(skill)
            seen.add(skill)
            for child in RELATED_SKILLS.get(skill, ()):
                seen.add(child)
    return sorted(found, key=str.lower)


def _skill_evidence_label(resume_text: str, skill: str) -> str:
    lower = resume_text.lower()
    lines = lower.splitlines()
    hits = lower.count(skill.replace("+", r"\+"))
    section_boost = any(skill in line for line in lines if "skill" in line or "technical" in line)
    if hits >= 3 or section_boost:
        return "strong"
    if hits >= 1:
        return "weak"
    return "weak"


def _partial_skills(resume_skills: list[str], missing: list[str]) -> list[str]:
    resume_set = set(resume_skills)
    partial: list[str] = []
    for js in missing:
        if js in resume_set:
            continue
        for rs in resume_set:
            if len(js) >= 4 and (js in rs or rs in js):
                partial.append(js)
                break
    return sorted(set(partial), key=str.lower)


def build_skill_match(resume_text: str, job_text: str) -> SkillMatch:
    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_text)
    resume_set = set(resume_skills)
    job_set = set(job_skills)
    matched = sorted(job_set & resume_set, key=str.lower)
    missing = sorted(job_set - resume_set, key=str.lower)
    partial = _partial_skills(resume_skills, missing)

    denom = max(len(job_set), 1)
    score = min(100.0, (len(matched) + 0.45 * len(partial)) / denom * 100)

    graph_matches: list[str] = []
    for skill in matched:
        related = RELATED_SKILLS.get(skill, ())
        extras = sorted(set(related) & resume_set - {skill})
        if extras:
            graph_matches.append(f"{skill.title()} aligns with related strengths: {', '.join(extras)}.")

    evidence = {skill: _skill_evidence_label(resume_text, skill) for skill in matched}

    return SkillMatch(
        matched=list(matched),
        missing=list(missing),
        partial=list(partial),
        score=round(score, 2),
        graph_matches=graph_matches,
        evidence=evidence,
    )


def skill_graph_explanations(skill_match: SkillMatch) -> list[str]:
    lines = list(skill_match.graph_matches)
    if skill_match.partial:
        tail = ", ".join(skill_match.partial[:12])
        suffix = "..." if len(skill_match.partial) > 12 else ""
        lines.append(f"Partial overlaps suggest adjacent skills — strengthen wording around: {tail}{suffix}")
    return lines


def _pair_tfidf_cosine(resume: str, job: str) -> float:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore[import-not-found]
        from sklearn.metrics.pairwise import cosine_similarity  # type: ignore[import-not-found]
    except ImportError:
        return 0.0

    resume = resume or ""
    job = job or ""
    if len(resume.strip()) < 8 or len(job.strip()) < 8:
        return 0.0

    vectorizer = TfidfVectorizer(max_features=4096, stop_words="english", ngram_range=(1, 2))
    try:
        tfidf = vectorizer.fit_transform([resume, job])
        sim = float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])
        if math.isnan(sim) or math.isinf(sim):
            sim = 0.0
        return float(max(0.0, min(1.0, sim)))
    except ValueError:
        return 0.0


def extract_scoring_features(resume: str, job_description: str, use_transformer: bool = False) -> dict[str, float]:
    _ = use_transformer
    resume = resume or ""
    job = job_description or ""
    resume_tokens = _tokenize(resume)
    job_tokens = _tokenize(job)
    resume_counter = Counter(resume_tokens)
    job_counter = Counter(job_tokens)
    overlap = sum(min(resume_counter[tok], job_counter[tok]) for tok in job_counter)
    denom_overlap = max(sum(job_counter.values()), 1)

    resume_skill_set = set(extract_skills(resume))
    job_skill_set = set(extract_skills(job))
    union = resume_skill_set | job_skill_set
    jaccard = len(resume_skill_set & job_skill_set) / max(len(union), 1)

    bullets = resume.count("\n-") + resume.count("\n•") + resume.count("\n*")
    digits_resume = sum(ch.isdigit() for ch in resume)

    return {
        "semantic_tfidf": _pair_tfidf_cosine(resume, job),
        "token_overlap_density": overlap / denom_overlap,
        "resume_word_count": math.log1p(len(resume_tokens)),
        "job_word_count": math.log1p(len(job_tokens)),
        "resume_char_log": math.log1p(len(resume)),
        "job_char_log": math.log1p(len(job)),
        "skill_jaccard": jaccard,
        "resume_skill_count": float(len(resume_skill_set)),
        "job_skill_count": float(len(job_skill_set)),
        "skill_coverage": len(resume_skill_set & job_skill_set) / max(len(job_skill_set), 1),
        "bullet_estimate": math.log1p(bullets),
        "resume_digits_ratio": digits_resume / max(len(resume), 1),
        "unique_resume_tokens_ratio": len(resume_counter) / max(len(resume_tokens), 1),
        "unique_job_tokens_ratio": len(job_counter) / max(len(job_tokens), 1),
    }


def _bias_scan(job_text: str) -> dict[str, Any]:
    blob = job_text.lower()
    hits_high = [term for term in BIAS_TERMS_HIGH if term in blob]
    hits_med = [term for term in BIAS_TERMS_MEDIUM if term in blob]
    risk = "High" if hits_high else "Medium" if hits_med else "Low"
    return {
        "risk_level": risk,
        "signals": hits_high + hits_med,
        "notes": "Heuristic scan for exclusionary language in the job description.",
    }


def _ats_score(resume_text: str, job_text: str, skill_match: SkillMatch) -> tuple[float, list[str]]:
    suggestions: list[str] = []
    score = 55.0
    lines = [ln.strip() for ln in resume_text.splitlines() if ln.strip()]
    if len(lines) >= 10:
        score += 10
    else:
        suggestions.append("Expand structured sections so ATS parsers detect clear headings and bullets.")

    if re.search(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", resume_text, re.I):
        score += 5
    else:
        suggestions.append("Include a professional email address near the top.")

    if re.search(r"\+?\d[\d\s().-]{8,}\d", resume_text):
        score += 4

    bullets = resume_text.count("\n-") + resume_text.count("•")
    if bullets >= 3:
        score += 8
    else:
        suggestions.append("Use bullet points to highlight measurable achievements.")

    if skill_match.missing:
        suggestions.append(
            "Mirror critical keywords naturally where truthful: " + ", ".join(skill_match.missing[:6]) + "."
        )

    coverage = len(skill_match.matched) / max(len(skill_match.matched) + len(skill_match.missing), 1)
    score += coverage * 18

    jd_terms = set(_tokenize(job_text))
    resume_terms = set(_tokenize(resume_text))
    overlap_ratio = len(jd_terms & resume_terms) / max(len(jd_terms), 1)
    score += overlap_ratio * 10

    return float(min(100, max(15, score))), suggestions


def _section_feedback(resume_text: str, job_skills: list[str]) -> dict[str, dict[str, str]]:
    lower = resume_text.lower()

    def grade(hit: bool, detail: str) -> dict[str, str]:
        return {"rating": "Strong" if hit else "Needs work", "feedback": detail}

    education_hit = bool(
        re.search(r"\b(education|university|college|bachelor|master|phd|b\.?s\.?|m\.?s\.?)\b", lower)
    )
    experience_hit = bool(
        re.search(r"\b(experience|employment|internship|engineer|developer|analyst)\b", lower)
    )
    projects_hit = bool(re.search(r"\b(project|portfolio|github)\b", lower))
    skills_hit = bool(re.search(r"\b(skill|technologies|tools)\b", lower))

    overlap_hint = ""
    if job_skills:
        overlap_hint = f" Mirror truthful mentions of {', '.join(job_skills[:4])}, where relevant."

    return {
        "education": grade(
            education_hit,
            "Surface degree, major, and timeframe clearly." if not education_hit else "Education cues detected.",
        ),
        "experience": grade(
            experience_hit,
            (
                "Quantify scope, tech stack, and outcomes for each role." + overlap_hint
                if experience_hit
                else "Add concise impact bullets tied to role keywords." + overlap_hint
            ),
        ),
        "projects": grade(
            projects_hit,
            "Call out repos, datasets, metrics, or deployment details." if not projects_hit else "Projects section looks present.",
        ),
        "skills": grade(
            skills_hit,
            "Group skills by domain (languages, frameworks, data, cloud)." if not skills_hit else "Skills section detected.",
        ),
    }


def _weighted_section_score(sections: dict[str, dict[str, str]]) -> float:
    scores: list[float] = []
    for key in ("education", "experience", "projects", "skills"):
        rating = sections.get(key, {}).get("rating", "")
        scores.append(92.0 if rating == "Strong" else 38.0)
    return round(sum(scores) / max(len(scores), 1), 1)


def _impact_snippets(resume_text: str, limit: int = 3) -> list[str]:
    out: list[str] = []
    for line in resume_text.splitlines():
        t = line.strip()
        if len(t) < 14 or len(t) > 220:
            continue
        if re.search(r"\d|%|\$|\bkpi\b|\braised\b|\breduced\b|\bimproved\b|\bincreased\b", t, re.I):
            out.append(t[:180])
        if len(out) >= limit:
            break
    return out


def _resume_intelligence(
    resume_text: str,
    skill_match: SkillMatch,
    sections: dict[str, dict[str, str]],
) -> dict[str, Any]:
    lower = resume_text.lower()
    seniority = "unknown"
    if re.search(r"\b(principal|staff|lead|director|head)\b", lower):
        seniority = "leadership_track"
    elif re.search(r"\b(senior|sr\.)\b", lower):
        seniority = "senior"
    elif re.search(r"\b(junior|intern|graduate)\b", lower):
        seniority = "early_career"

    domains = []
    if any(term in lower for term in ("machine learning", "model", "nlp", "tensorflow", "pytorch")):
        domains.append("ML / AI")
    if any(term in lower for term in ("frontend", "react", "vue", "css")):
        domains.append("Frontend")
    if any(term in lower for term in ("backend", "api", "microservice", "database")):
        domains.append("Backend")
    if any(term in lower for term in ("data", "sql", "etl", "warehouse")):
        domains.append("Data")

    inferred_role = "General technologist"
    skills_top = extract_skills(resume_text)[:3]
    if skills_top:
        inferred_role = f"{skills_top[0].title()} practitioner"

    wscore = _weighted_section_score(sections)
    tool_depths = {
        skill: ("strong" if skill_match.evidence.get(skill) == "strong" else "moderate")
        for skill in skill_match.matched[:10]
    }

    return {
        "inferred_role": inferred_role,
        "seniority_estimate": seniority.replace("_", " ").title()
        if seniority != "unknown"
        else "Unknown",
        "candidate_stage": seniority,
        "domains": domains or ["General engineering"],
        "signal_skills": skill_match.matched[:12],
        "weighted_section_score": wscore,
        "project_impacts": _impact_snippets(resume_text),
        "tool_depths": tool_depths,
    }


def _load_model_bundle() -> dict[str, Any] | None:
    global _MODEL_ARTIFACT
    if _MODEL_ARTIFACT is not None:
        return _MODEL_ARTIFACT

    if not TRAINED_MODEL_PATH.exists():
        return None
    try:
        import joblib  # type: ignore[import-not-found]

        loaded = joblib.load(TRAINED_MODEL_PATH)
    except Exception:
        return None

    if isinstance(loaded, dict):
        _MODEL_ARTIFACT = loaded
        return loaded
    return None


def _predict_with_trained_model(features: dict[str, float]) -> tuple[float | None, str]:
    bundle = _load_model_bundle()
    if not bundle:
        return None, "formula_fallback"

    model = bundle.get("model")
    names: list[str] | None = bundle.get("feature_names")
    if model is None or not names:
        return None, "formula_fallback"

    vector = [[float(features.get(name, 0.0)) for name in names]]
    try:
        raw = float(model.predict(vector)[0])
        clipped = float(min(100, max(0, raw)))
        return clipped, "trained_model"
    except Exception:
        return None, "formula_fallback"


def analyze_resume(resume_text: str, job_text: str) -> dict[str, Any]:
    resume_text = _normalize_ws(resume_text)
    job_text = _normalize_ws(job_text)

    feats = extract_scoring_features(resume_text, job_text)
    tfidf_sem = feats["semantic_tfidf"] * 100
    overlap_sem = feats["token_overlap_density"] * 100
    jaccard_sem = feats["skill_jaccard"] * 100
    semantic_score = float(
        min(100.0, max(tfidf_sem, overlap_sem * 0.92, jaccard_sem * 0.65))
    )
    skill_match = build_skill_match(resume_text, job_text)
    skill_component = float(skill_match.score)

    ats_value, ats_suggestions = _ats_score(resume_text, job_text, skill_match)

    heuristic_final = float(
        min(100, max(0, 0.48 * semantic_score + 0.44 * skill_component + 0.08 * ats_value))
    )

    ml_score, scoring_method = _predict_with_trained_model(feats)
    base_final = heuristic_final
    final_score_float = heuristic_final
    if ml_score is not None:
        final_score_float = float(min(100, max(0, 0.72 * ml_score + 0.28 * heuristic_final)))

    bias = _bias_scan(job_text)

    improvements = [
        (
            f"Close skill gaps focusing on: {', '.join(skill_match.missing[:6])}."
            if skill_match.missing
            else "Skills align well — tighten outcomes wording."
        ),
        (
            "Weave JD keywords naturally into impact bullets."
            if skill_match.missing
            else "Highlight measurable wins with metrics."
        ),
        "Ensure LinkedIn/GitHub links are clickable if submitting HTML resumes.",
    ]

    sections = _section_feedback(resume_text, skill_match.matched + skill_match.missing)
    resume_intel = _resume_intelligence(resume_text, skill_match, sections)

    overall = (
        f"Fit blends semantic similarity ({semantic_score:.1f}/100), skill coverage ({skill_component:.1f}/100), "
        f"and ATS readiness ({ats_value:.1f}/100). Method: {scoring_method.replace('_', ' ')}."
    )

    return {
        "final_score": int(round(final_score_float)),
        "base_final_score": round(base_final, 2),
        "learning_to_rank_score": None,
        "semantic_score": int(round(semantic_score)),
        "skill_match": skill_match,
        "ats_score": int(round(ats_value)),
        "bias": bias,
        "overall_explanation": overall,
        "sections": sections,
        "ats_suggestions": ats_suggestions,
        "improvements": improvements,
        "scoring_method": scoring_method,
        "resume_intelligence": resume_intel,
    }


def format_report(result: dict[str, Any]) -> str:
    bias = result.get("bias") if isinstance(result.get("bias"), dict) else {}
    sections = result.get("sections") if isinstance(result.get("sections"), dict) else {}
    skill_match = result.get("skill_match")
    lines = [
        "### Match overview",
        f"- Final score: **{result.get('final_score')}**/100",
        f"- Semantic match: **{result.get('semantic_score')}%**",
        f"- ATS score: **{result.get('ats_score')}**/100",
        f"- Scoring method: `{result.get('scoring_method')}`",
        "",
        "### Narrative",
        str(result.get("overall_explanation", "")),
        "",
        "### Skills",
    ]
    if isinstance(skill_match, SkillMatch):
        lines.extend(
            [
                f"- Matched: {', '.join(skill_match.matched) or 'None'}",
                f"- Missing: {', '.join(skill_match.missing) or 'None'}",
                f"- Partial: {', '.join(skill_match.partial) or 'None'}",
                "",
                "#### Skill graph notes",
            ]
        )
        lines.extend(f"- {item}" for item in skill_graph_explanations(skill_match))
    lines.extend(["", "### Section feedback"])
    for key in ["education", "experience", "projects", "skills"]:
        block = sections.get(key, {})
        lines.append(f"- **{key.title()}**: {block.get('rating', 'n/a')} — {block.get('feedback', '')}")

    lines.extend(
        [
            "",
            "### Bias checklist",
            f"- Risk level: **{bias.get('risk_level', 'Unknown')}**",
        ]
    )
    signals = bias.get("signals") if isinstance(bias.get("signals"), list) else []
    if signals:
        lines.append("- Signals: " + ", ".join(str(s) for s in signals))

    lines.extend(["", "### ATS ideas", *[f"- {s}" for s in result.get("ats_suggestions", [])]])
    lines.extend(["", "### Improvements", *[f"- {s}" for s in result.get("improvements", [])]])
    return "\n".join(lines)


def infer_resume_job_queries(resume_text: str, max_queries: int = 6) -> list[str]:
    text = resume_text or ""
    skills = extract_skills(text)
    queries: list[str] = []
    seen_key: set[str] = set()

    def push(raw: str) -> None:
        q = sanitize_job_query_text(_normalize_ws(raw))
        if len(q) < 4:
            return
        key = q.casefold()
        if key in seen_key:
            return
        seen_key.add(key)
        queries.append(q)

    role_pattern = re.compile(
        r"\b(data scientist|data engineer|ml engineer|software engineer|python developer|java developer|"
        r"backend developer|frontend developer|full stack developer|full stack|devops engineer|web developer|"
        r"sde|machine learning engineer|ai engineer|cloud engineer|product manager|business analyst|intern)\b",
        re.I,
    )
    for match in role_pattern.finditer(text):
        push(match.group(0))

    for skill in skills[:5]:
        push(f"{skill} developer")
        push(f"{skill} engineer")

    for line in text.splitlines():
        stripped = line.strip()
        if _line_is_bad_job_query_headline(stripped):
            continue
        push(stripped[:88])

    if not queries:
        push("software engineer")

    return queries[:max_queries]


def rank_jobs_for_resume(resume_text: str, postings: list[JobPosting], limit: int = 10) -> list[dict[str, Any]]:
    limit = max(1, limit)
    scored: list[dict[str, Any]] = []
    resume_text = _normalize_ws(resume_text)

    for posting in postings:
        job_body = _normalize_ws("\n".join([posting.title, posting.company, posting.description]))
        if len(job_body) < 20:
            continue
        analysis = analyze_resume(resume_text, job_body)
        skill_match: SkillMatch = analysis["skill_match"]
        bias = analysis["bias"] if isinstance(analysis["bias"], dict) else {}

        job_skills = extract_skills(job_body)
        resume_skills = extract_skills(resume_text)
        overlap = len(set(job_skills) & set(resume_skills))
        experience_fit = {
            "shared_skills": overlap,
            "job_skill_target": len(job_skills),
            "coverage": round(overlap / max(len(job_skills), 1), 3),
        }

        summary = (
            f"{analysis['overall_explanation']} "
            f"Key matches: {', '.join(skill_match.matched[:6]) or 'general wording overlap'}."
        )

        scored.append(
            {
                "rank": 0,
                "source_index": posting.source_index,
                "title": posting.title or "Role",
                "company": posting.company or "",
                "url": posting.url or "",
                "final_score": analysis["final_score"],
                "base_final_score": analysis.get("base_final_score"),
                "learning_to_rank_score": analysis.get("learning_to_rank_score"),
                "semantic_score": analysis["semantic_score"],
                "skill_score": int(round(skill_match.score)),
                "ats_score": analysis["ats_score"],
                "bias_risk": bias.get("risk_level", "Unknown"),
                "experience_fit": experience_fit,
                "summary": summary,
                "matched_skills": skill_match.matched,
                "missing_skills": skill_match.missing,
                "partial_skills": skill_match.partial,
                "analysis": analysis,
            }
        )

    scored.sort(key=lambda item: item["final_score"], reverse=True)
    for idx, item in enumerate(scored[:limit], start=1):
        item["rank"] = idx
    return scored[:limit]


def trained_model_info() -> dict[str, Any]:
    path = TRAINED_MODEL_PATH
    if not path.exists():
        return {"available": False, "path": str(path)}
    bundle = _load_model_bundle()
    if not bundle:
        return {"available": False, "path": str(path)}
    return {
        "available": True,
        "path": str(path),
        "trained_rows": bundle.get("trained_rows"),
        "target_scale": bundle.get("target_scale"),
        "model_type": bundle.get("model_type"),
        "metrics": bundle.get("metrics", {}),
        "rank_metrics": bundle.get("rank_metrics", {}),
    }
