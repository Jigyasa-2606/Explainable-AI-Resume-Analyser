from __future__ import annotations

import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TRAINED_MODEL_PATH = Path(__file__).resolve().parent / "trained_match_model.joblib"
TRAINED_MODEL_ZIP_PATH = TRAINED_MODEL_PATH.parent / "trained_match_model.joblib.zip"

_MODEL_ARTIFACT: dict[str, Any] | None = None
_MODEL_LOAD_ATTEMPTED = False
_MODEL_LOAD_ERROR: str | None = None
_SBERT_MODEL: Any = None
_CROSS_ENCODER: Any = None

SBERT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_STAGE2_TOP_K = 30
DEFAULT_STAGE2_BLEND = 0.55
LTR_SHORTLIST_K = 40

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


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float, lo: float = 0.0, hi: float = 1.0) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(min(hi, max(lo, float(raw))))
    except ValueError:
        return default


def _env_int(name: str, default: int, lo: int = 1, hi: int = 500) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(min(hi, max(lo, int(raw))))
    except ValueError:
        return default


def _use_transformer_embeddings() -> bool:
    return _env_flag("USE_TRANSFORMER_EMBEDDINGS")


def _ml_heuristic_blend() -> float:
    """0 = pure ML when a trained model exists; 1 = pure heuristic formula."""
    return _env_float("ML_HEURISTIC_BLEND", 0.0)


def _use_ltr_rerank() -> bool:
    """LTR lost to RF on the current labels; keep it opt-in."""
    return _env_flag("USE_LTR_RERANK")


def _stage2_reranker_mode() -> str:
    raw = os.getenv("STAGE2_RERANKER", "off").strip().lower()
    if raw in {"off", "false", "0", "none"}:
        return "off"
    if raw in {"cross-encoder", "cross_encoder", "crossencoder"}:
        return "cross-encoder"
    if raw in {"sbert", "bi-encoder", "biencoder"}:
        return "sbert"
    return "auto"


def _stage2_top_k() -> int:
    return _env_int("STAGE2_TOP_K", DEFAULT_STAGE2_TOP_K, lo=5, hi=200)


def _stage2_blend() -> float:
    return _env_float("STAGE2_BLEND", DEFAULT_STAGE2_BLEND)


def _clip_for_embedding(text: str, limit: int = 2500) -> str:
    clipped = _normalize_ws(text)
    if len(clipped) <= limit:
        return clipped
    return clipped[:limit]


def _cosine_01(left: Any, right: Any) -> float:
    try:
        import numpy as np  # type: ignore[import-not-found]

        sim = float(np.dot(left, right))
    except Exception:
        sim = float(sum(float(a) * float(b) for a, b in zip(left, right)))
    if math.isnan(sim) or math.isinf(sim):
        return 0.0
    return float(max(0.0, min(1.0, sim)))


def _minmax_to_100(values: list[float]) -> list[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    span = hi - lo
    if span <= 1e-6:
        return [50.0 for _ in values]
    return [float((value - lo) / span * 100.0) for value in values]


def transformer_available() -> bool:
    try:
        import importlib.util

        return importlib.util.find_spec("sentence_transformers") is not None
    except Exception:
        return False


def _load_sbert() -> Any | None:
    global _SBERT_MODEL
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        if _SBERT_MODEL is None:
            _SBERT_MODEL = SentenceTransformer(SBERT_MODEL_NAME)
        return _SBERT_MODEL
    except Exception:
        return None


def _load_cross_encoder() -> Any | None:
    global _CROSS_ENCODER
    try:
        from sentence_transformers import CrossEncoder  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        if _CROSS_ENCODER is None:
            _CROSS_ENCODER = CrossEncoder(CROSS_ENCODER_MODEL_NAME)
        return _CROSS_ENCODER
    except Exception:
        return None


def sbert_pair_scores(resumes: list[str], jobs: list[str]) -> list[float] | None:
    """Batch bi-encoder cosine scores on a 0-100 scale. One encode call for all unique texts."""
    if len(resumes) != len(jobs) or not resumes:
        return None
    model = _load_sbert()
    if model is None:
        return None

    unique: list[str] = []
    index: dict[str, int] = {}
    for text in [*resumes, *jobs]:
        clipped = _clip_for_embedding(text)
        if clipped not in index:
            index[clipped] = len(unique)
            unique.append(clipped)
    try:
        vectors = model.encode(unique, normalize_embeddings=True, show_progress_bar=False, batch_size=32)
    except Exception:
        return None

    scores: list[float] = []
    for resume, job in zip(resumes, jobs):
        left = vectors[index[_clip_for_embedding(resume)]]
        right = vectors[index[_clip_for_embedding(job)]]
        scores.append(_cosine_01(left, right) * 100.0)
    return scores


def cross_encoder_pair_scores(resumes: list[str], jobs: list[str]) -> list[float] | None:
    """Small cross-encoder rerank scores, min-max scaled to 0-100 within the batch."""
    if len(resumes) != len(jobs) or not resumes:
        return None
    model = _load_cross_encoder()
    if model is None:
        return None
    pairs = [
        [_clip_for_embedding(resume, 1800), _clip_for_embedding(job, 1800)]
        for resume, job in zip(resumes, jobs)
    ]
    try:
        raw = [float(value) for value in model.predict(pairs, show_progress_bar=False)]
    except Exception:
        return None
    return _minmax_to_100(raw)


def stage2_pair_scores(
    resumes: list[str],
    jobs: list[str],
    preferred: str | None = None,
) -> tuple[list[float] | None, str]:
    mode = preferred or _stage2_reranker_mode()
    if mode == "off":
        return None, "off"
    if mode in {"cross-encoder", "auto"}:
        if mode == "cross-encoder":
            scores = cross_encoder_pair_scores(resumes, jobs)
            return (scores, "cross-encoder") if scores is not None else (None, "unavailable")
    if mode in {"sbert", "auto"}:
        scores = sbert_pair_scores(resumes, jobs)
        if scores is not None:
            return scores, "sbert"
    if mode == "auto":
        scores = cross_encoder_pair_scores(resumes, jobs)
        if scores is not None:
            return scores, "cross-encoder"
    return None, "unavailable"


def _pair_sbert_cosine(resume: str, job: str) -> float:
    scores = sbert_pair_scores([resume or ""], [job or ""])
    if not scores:
        return 0.0
    return float(max(0.0, min(1.0, scores[0] / 100.0)))

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
    use_sbert = use_transformer or _use_transformer_embeddings()
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

    semantic_sbert = _pair_sbert_cosine(resume, job) if use_sbert else 0.0

    return {
        "semantic_tfidf": _pair_tfidf_cosine(resume, job),
        "semantic_sbert": semantic_sbert,
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


def _ensure_trained_model_file() -> None:
    """Extract trained_match_model.joblib from the zip when only the archive is deployed."""
    if TRAINED_MODEL_PATH.exists():
        return
    if not TRAINED_MODEL_ZIP_PATH.exists():
        return
    try:
        import zipfile

        with zipfile.ZipFile(TRAINED_MODEL_ZIP_PATH) as archive:
            archive.extract("trained_match_model.joblib", TRAINED_MODEL_PATH.parent)
    except Exception:
        return


def _load_model_bundle() -> dict[str, Any] | None:
    global _MODEL_ARTIFACT, _MODEL_LOAD_ATTEMPTED, _MODEL_LOAD_ERROR
    if _MODEL_LOAD_ATTEMPTED:
        return _MODEL_ARTIFACT

    _MODEL_LOAD_ATTEMPTED = True
    _ensure_trained_model_file()

    if not TRAINED_MODEL_PATH.exists():
        _MODEL_LOAD_ERROR = f"missing {TRAINED_MODEL_PATH.name}"
        return None
    try:
        import joblib  # type: ignore[import-not-found]

        loaded = joblib.load(TRAINED_MODEL_PATH)
    except Exception as exc:
        _MODEL_LOAD_ERROR = str(exc)
        return None

    if isinstance(loaded, dict):
        _MODEL_ARTIFACT = loaded
        return loaded
    _MODEL_LOAD_ERROR = "model file is not a dict artifact"
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


def regression_scores(feature_rows: list[dict[str, float]]) -> list[float] | None:
    bundle = _load_model_bundle()
    if not bundle:
        return None

    model = bundle.get("model")
    names: list[str] | None = bundle.get("feature_names")
    if model is None or not names:
        return None

    matrix = [[float(row.get(name, 0.0)) for name in names] for row in feature_rows]
    try:
        preds = model.predict(matrix)
        return [float(min(100, max(0, float(score)))) for score in preds]
    except Exception:
        return None


def learning_to_rank_scores(feature_rows: list[dict[str, float]]) -> list[float] | None:
    """Pairwise ranker win-rate scores for a resume compared against multiple jobs."""
    bundle = _load_model_bundle()
    if not bundle:
        return None

    ranker = bundle.get("ranker")
    names: list[str] | None = bundle.get("feature_names")
    if ranker is None or not names or len(feature_rows) < 2:
        return None

    vectors = [[float(row.get(name, 0.0)) for name in names] for row in feature_rows]
    win_scores = [0.0] * len(vectors)

    for left in range(len(vectors)):
        for right in range(len(vectors)):
            if left == right:
                continue
            diff = [vectors[left][idx] - vectors[right][idx] for idx in range(len(vectors[left]))]
            try:
                proba = ranker.predict_proba([diff])[0]
                win_scores[left] += float(proba[1]) if len(proba) > 1 else float(proba[0])
            except Exception:
                win_scores[left] += 0.5

    span = max(win_scores) - min(win_scores)
    if span <= 1e-6:
        return [50.0 for _ in win_scores]
    return [float((score - min(win_scores)) / span * 100) for score in win_scores]


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
    blend = _ml_heuristic_blend()
    base_final = heuristic_final
    final_score_float = heuristic_final

    if ml_score is not None:
        final_score_float = float(
            min(100, max(0, (1.0 - blend) * ml_score + blend * heuristic_final))
        )
        if blend < 0.05:
            scoring_method = "trained_model"
        else:
            scoring_method = "trained_model_hybrid"

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

    if scoring_method.startswith("trained_model"):
        overall = (
            f"Random Forest match model predicts {final_score_float:.1f}/100 "
            f"(regression on engineered NLP features). "
            f"Supporting signals — semantic: {semantic_score:.1f}, skills: {skill_component:.1f}, ATS: {ats_value:.1f}."
        )
    else:
        overall = (
            f"Heuristic fallback score {final_score_float:.1f}/100 — train "
            f"`trained_match_model.joblib` for ML scoring. "
            f"Signals: semantic {semantic_score:.1f}, skills {skill_component:.1f}, ATS {ats_value:.1f}."
        )

    return {
        "final_score": int(round(final_score_float)),
        "base_final_score": round(ml_score if ml_score is not None else base_final, 2),
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
        "ml_regression_score": round(ml_score, 2) if ml_score is not None else None,
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


def heuristic_score_from_features(features: dict[str, float]) -> float:
    tfidf_sem = features.get("semantic_tfidf", 0.0) * 100
    overlap_sem = features.get("token_overlap_density", 0.0) * 100
    jaccard_sem = features.get("skill_jaccard", 0.0) * 100
    semantic = float(min(100.0, max(tfidf_sem, overlap_sem * 0.92, jaccard_sem * 0.65)))
    skill = float(features.get("skill_coverage", 0.0) * 100.0)
    ats = float(min(100.0, 55.0 + features.get("skill_coverage", 0.0) * 18.0))
    return float(min(100.0, max(0.0, 0.48 * semantic + 0.44 * skill + 0.08 * ats)))


def first_pass_scores_for_pairs(resumes: list[str], jobs: list[str]) -> tuple[list[float], str]:
    """Cheap TF-IDF + skills + Random Forest scores for many resume/job pairs."""
    if len(resumes) != len(jobs) or not resumes:
        return [], "empty"
    feature_rows = [
        extract_scoring_features(resume, job, use_transformer=False)
        for resume, job in zip(resumes, jobs)
    ]
    rf_scores = regression_scores(feature_rows)
    if rf_scores is not None:
        return rf_scores, "trained_model"
    return [heuristic_score_from_features(row) for row in feature_rows], "formula_fallback"


def apply_shortlist_rerank(
    first_pass: list[float],
    stage2: list[float],
    top_k: int,
    blend: float,
) -> tuple[list[float], list[int]]:
    """Blend stage-2 scores into the first-pass top-k, then keep the rest in first-pass order."""
    count = len(first_pass)
    if count == 0 or len(stage2) != count:
        return list(first_pass), list(range(len(first_pass)))
    top_k = max(1, min(top_k, count))
    order = sorted(range(count), key=lambda idx: first_pass[idx], reverse=True)
    head = order[:top_k]
    tail = order[top_k:]
    blended = list(first_pass)
    for idx in head:
        blended[idx] = float(min(100.0, max(0.0, (1.0 - blend) * first_pass[idx] + blend * stage2[idx])))
    head_sorted = sorted(head, key=lambda idx: blended[idx], reverse=True)
    return blended, head_sorted + tail


def two_stage_scores_for_pairs(
    resumes: list[str],
    jobs: list[str],
    top_k: int | None = None,
    blend: float | None = None,
    preferred_reranker: str | None = None,
) -> tuple[list[float], str, list[float | None]]:
    """
    Stage 1: RF / heuristic over all pairs.
    Stage 2: SBERT or cross-encoder on the first-pass top-k only.
    """
    first_pass, first_method = first_pass_scores_for_pairs(resumes, jobs)
    if not first_pass:
        return [], first_method, []

    rerank_k = top_k or _stage2_top_k()
    rerank_blend = _stage2_blend() if blend is None else blend
    stage2_values: list[float | None] = [None] * len(first_pass)
    order = sorted(range(len(first_pass)), key=lambda idx: first_pass[idx], reverse=True)
    head = order[: max(1, min(rerank_k, len(first_pass)))]

    head_resumes = [resumes[idx] for idx in head]
    head_jobs = [jobs[idx] for idx in head]
    semantic, reranker_name = stage2_pair_scores(head_resumes, head_jobs, preferred=preferred_reranker)
    if semantic is None:
        return first_pass, first_method, stage2_values

    dense_stage2 = list(first_pass)
    for idx, score in zip(head, semantic):
        dense_stage2[idx] = score
        stage2_values[idx] = round(score, 2)
    blended, _ = apply_shortlist_rerank(first_pass, dense_stage2, len(head), rerank_blend)
    return blended, f"two_stage_{reranker_name}", stage2_values


def semantic_similarity(resume: str, job: str, use_transformer: bool = False) -> float:
    if use_transformer:
        scores = sbert_pair_scores([resume], [job])
        if scores:
            return scores[0]
    return _pair_tfidf_cosine(resume, job) * 100.0


def compare_skills(resume: str, job: str) -> SkillMatch:
    return build_skill_match(resume, job)


def ats_score(resume: str, job: str, skills: SkillMatch) -> tuple[float, list[str]]:
    return _ats_score(resume, job, skills)


def analyze_sections(resume: str, job_skills: list[str] | None = None) -> dict[str, dict[str, str]]:
    return _section_feedback(resume, job_skills or [])


def final_score(semantic: float, skill: float, sections: Any = None, ats: float = 0.0) -> float:
    del sections
    return float(min(100.0, max(0.0, 0.48 * semantic + 0.44 * skill + 0.08 * ats)))


def rank_jobs_for_resume(resume_text: str, postings: list[JobPosting], limit: int = 10) -> list[dict[str, Any]]:
    limit = max(1, limit)
    resume_text = _normalize_ws(resume_text)
    candidates: list[dict[str, Any]] = []
    feature_rows: list[dict[str, float]] = []
    job_bodies: list[str] = []

    for posting in postings:
        job_body = _normalize_ws("\n".join([posting.title, posting.company, posting.description]))
        if len(job_body) < 20:
            continue

        analysis = analyze_resume(resume_text, job_body)
        skill_match: SkillMatch = analysis["skill_match"]
        bias = analysis["bias"] if isinstance(analysis["bias"], dict) else {}
        feature_rows.append(extract_scoring_features(resume_text, job_body, use_transformer=False))
        job_bodies.append(job_body)

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

        candidates.append(
            {
                "rank": 0,
                "source_index": posting.source_index,
                "title": posting.title or "Role",
                "company": posting.company or "",
                "url": posting.url or "",
                "final_score": analysis["final_score"],
                "base_final_score": analysis.get("base_final_score"),
                "first_pass_score": analysis.get("base_final_score"),
                "stage2_score": None,
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

    if not candidates:
        return []

    rf_scores = regression_scores(feature_rows)
    if rf_scores is not None:
        first_pass = rf_scores
        scoring_method = "trained_model"
    else:
        first_pass = [float(item["final_score"]) for item in candidates]
        scoring_method = str(candidates[0]["analysis"].get("scoring_method") or "formula_fallback")

    for item, score in zip(candidates, first_pass):
        rounded = round(score, 2)
        item["first_pass_score"] = rounded
        item["final_score"] = int(round(score))
        item["analysis"]["first_pass_score"] = rounded
        item["analysis"]["ml_regression_score"] = item["analysis"].get("ml_regression_score") or rounded

    rerank_k = _stage2_top_k()
    order = sorted(range(len(candidates)), key=lambda idx: first_pass[idx], reverse=True)
    stage2_head = order[: max(1, min(rerank_k, len(candidates)))]
    semantic, reranker_name = stage2_pair_scores(
        [resume_text] * len(stage2_head),
        [job_bodies[idx] for idx in stage2_head],
    )

    if semantic is None:
        if _use_ltr_rerank():
            ltr_window = order[: max(2, min(LTR_SHORTLIST_K, len(candidates)))]
            if len(ltr_window) >= 2:
                ltr_scores = learning_to_rank_scores([feature_rows[idx] for idx in ltr_window])
                if ltr_scores and len(ltr_scores) == len(ltr_window):
                    scoring_method = "learning_to_rank"
                    for idx, ltr_score in zip(ltr_window, ltr_scores):
                        rounded_ltr = round(ltr_score, 2)
                        candidates[idx]["learning_to_rank_score"] = rounded_ltr
                        candidates[idx]["final_score"] = int(round(ltr_score))
                        candidates[idx]["analysis"]["learning_to_rank_score"] = rounded_ltr
                        candidates[idx]["analysis"]["final_score"] = int(round(ltr_score))
                        candidates[idx]["analysis"]["scoring_method"] = "learning_to_rank"
                        candidates[idx]["summary"] = (
                            f"Learning-to-rank model score {rounded_ltr:.1f}/100. "
                            f"Random Forest first pass {candidates[idx]['first_pass_score']}. "
                            f"Key matches: {', '.join(candidates[idx]['matched_skills'][:6]) or 'general wording overlap'}."
                        )
                    ranked = [candidates[idx] for idx in sorted(ltr_window, key=lambda i: candidates[i]["final_score"], reverse=True)]
                    ranked.extend(candidates[idx] for idx in order if idx not in set(ltr_window))
                    for idx, item in enumerate(ranked[:limit], start=1):
                        item["rank"] = idx
                    return ranked[:limit]
        for item in candidates:
            item["analysis"]["scoring_method"] = scoring_method
            item["summary"] = (
                f"Random Forest match score {item['first_pass_score']:.1f}/100. "
                f"Key matches: {', '.join(item['matched_skills'][:6]) or 'general wording overlap'}."
            )
            item["analysis"]["overall_explanation"] = item["summary"]
        candidates.sort(key=lambda item: item["final_score"], reverse=True)
    else:
        dense_stage2 = list(first_pass)
        for idx, score in zip(stage2_head, semantic):
            dense_stage2[idx] = score
            candidates[idx]["stage2_score"] = round(score, 2)
            candidates[idx]["analysis"]["stage2_score"] = round(score, 2)
        blended, ranked_order = apply_shortlist_rerank(first_pass, dense_stage2, len(stage2_head), _stage2_blend())
        scoring_method = f"two_stage_{reranker_name}"
        reranked_ids = set(stage2_head)
        for idx, score in enumerate(blended):
            rounded = round(score, 2)
            candidates[idx]["final_score"] = int(round(score))
            candidates[idx]["analysis"]["final_score"] = int(round(score))
            candidates[idx]["analysis"]["scoring_method"] = scoring_method
            if idx in reranked_ids:
                label = "SBERT" if reranker_name == "sbert" else "cross-encoder"
                candidates[idx]["summary"] = (
                    f"Two-stage ranker: {label} rerank {rounded:.1f}/100 "
                    f"(Random Forest first pass {candidates[idx]['first_pass_score']}, "
                    f"{label} {candidates[idx]['stage2_score']}). "
                    f"Key matches: {', '.join(candidates[idx]['matched_skills'][:6]) or 'general wording overlap'}."
                )
                candidates[idx]["analysis"]["overall_explanation"] = candidates[idx]["summary"]
        candidates = [candidates[idx] for idx in ranked_order]

    for idx, item in enumerate(candidates[:limit], start=1):
        item["rank"] = idx
        item["analysis"]["scoring_method"] = scoring_method
    return candidates[:limit]


def trained_model_info() -> dict[str, Any]:
    path = TRAINED_MODEL_PATH
    bundle = _load_model_bundle() if path.exists() else None
    return {
        "available": bool(bundle),
        "path": str(path),
        "trained_rows": bundle.get("trained_rows") if bundle else None,
        "target_scale": bundle.get("target_scale") if bundle else None,
        "model_type": bundle.get("model_type") if bundle else None,
        "metrics": bundle.get("metrics", {}) if bundle else {},
        "rank_metrics": bundle.get("rank_metrics", {}) if bundle else {},
        "load_error": _MODEL_LOAD_ERROR,
        "stage2": {
            "reranker": _stage2_reranker_mode(),
            "top_k": _stage2_top_k(),
            "blend": _stage2_blend(),
            "ltr_rerank": _use_ltr_rerank(),
            "sentence_transformers": transformer_available(),
        },
    }
