import React, { useEffect, useMemo, useState, useRef } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const API_BASE = import.meta.env.VITE_API_BASE || "";
const API_BASE_FALLBACK = API_BASE.includes("localhost")
  ? API_BASE.replace("localhost", "127.0.0.1")
  : "";

const ROLE_OPTIONS = [
  "Software Engineer", "Frontend Developer", "Backend Developer", "Full Stack Developer",
  "Python Developer", "Java Developer", "React Developer", "Node.js Developer",
  "Mobile App Developer", "DevOps Engineer", "Cloud Engineer",
  "Data Analyst", "Data Scientist", "Machine Learning Engineer", "AI Engineer",
  "Data Engineer", "QA Engineer", "Product Manager", "UI UX Designer",
  "Software Engineer Intern", "Data Analyst Intern", "Machine Learning Intern",
];

function UploadIcon() {
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="17 8 12 3 7 8" />
      <line x1="12" y1="3" x2="12" y2="15" />
    </svg>
  );
}

function ExternalLinkIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
      <polyline points="15 3 21 3 21 9" />
      <line x1="10" y1="14" x2="21" y2="3" />
    </svg>
  );
}

function MapPinIcon() {
  return (
    <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z" />
      <circle cx="12" cy="10" r="3" />
    </svg>
  );
}

function BriefcaseIcon() {
  return (
    <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="2" y="7" width="20" height="14" rx="2" ry="2" />
      <path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16" />
    </svg>
  );
}

function SkeletonCard() {
  return (
    <div className="skeleton-card">
      <div style={{ display: "flex", gap: 14, marginBottom: 18 }}>
        <div className="skel" style={{ width: 36, height: 36, borderRadius: 8, flexShrink: 0 }} />
        <div style={{ flex: 1 }}>
          <div className="skel skel-h1" />
          <div className="skel skel-h2" />
        </div>
      </div>
      <div className="skel-metrics">
        <div className="skel skel-metric" />
        <div className="skel skel-metric" />
        <div className="skel skel-metric" />
      </div>
      <div className="skel skel-p" />
      <div className="skel skel-p" />
      <div className="skel skel-p" style={{ width: "68%" }} />
    </div>
  );
}

function LoadingState({ count = 5 }) {
  return (
    <div className="skeleton-grid">
      {Array.from({ length: count }, (_, i) => (
        <SkeletonCard key={i} />
      ))}
    </div>
  );
}

function App() {
  const [resumeFile, setResumeFile] = useState(null);
  const [resumeText, setResumeText] = useState("");
  const [query, setQuery] = useState("");
  const [selectedRoles, setSelectedRoles] = useState([]);
  const [country, setCountry] = useState("in");
  const [datePosted, setDatePosted] = useState("month");
  const [opportunityType, setOpportunityType] = useState("jobs");
  const [loading, setLoading] = useState(false);
  const [statusMessage, setStatusMessage] = useState("");
  const [backendReady, setBackendReady] = useState(!API_BASE);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const resultsRef = useRef(null);
  const resumeTextAreaRef = useRef(null);

  useEffect(() => {
    if (!API_BASE) return;
    let cancelled = false;
    (async () => {
      setStatusMessage("Connecting…");
      const ready = await wakeBackend(API_BASE, (attempt) => {
        if (!cancelled) setStatusMessage(`Starting the matcher… ${attempt}/6`);
      });
      if (!cancelled) {
        setBackendReady(ready);
        setStatusMessage(ready ? "" : "The matcher is waking up. Wait a minute and try again.");
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const suggestedQuery = useMemo(() => {
    if (!result?.suggested_queries?.length) return "roles that fit your resume";
    return result.suggested_queries[0];
  }, [result]);

  function formatApiError(payload) {
    const d = payload?.detail;
    if (typeof d === "string") return d;
    if (Array.isArray(d)) {
      return d
        .map((item) => (typeof item === "object" && item?.msg ? `${item.loc?.join(".")}: ${item.msg}` : String(item)))
        .filter(Boolean)
        .join("; ");
    }
    if (d && typeof d === "object") return JSON.stringify(d);
    return "";
  }

  function toggleRole(role) {
    setSelectedRoles((current) => {
      if (current.includes(role)) return current.filter((item) => item !== role);
      if (current.length >= 5) return current;
      return [...current, role];
    });
  }

  async function findJobs(event) {
    event.preventDefault();
    setLoading(true);
    setError("");
    setStatusMessage("");
    setResult(null);

    const pasted = (resumeTextAreaRef.current?.value ?? resumeText).trim();

    if (!pasted && !resumeFile) {
      setError("Add your resume first — upload a PDF, DOCX, or TXT, or paste the text.");
      setLoading(false);
      return;
    }

    if (resumeFile && resumeFile.size === 0) {
      setError("That file looks empty. Pick another file or paste your resume.");
      setLoading(false);
      return;
    }

    const formData = new FormData();
    if (resumeFile) formData.append("resume_file", resumeFile);
    formData.append("resume_text", pasted);
    formData.append("query", query);
    formData.append("selected_roles", JSON.stringify(selectedRoles));
    formData.append("provider", "all");
    formData.append("country", country);
    formData.append("date_posted", datePosted);
    formData.append("opportunity_type", opportunityType);
    formData.append("result_limit", "25");
    formData.append("top_n", "10");

    try {
      if (API_BASE && !backendReady) {
        setStatusMessage("Starting the matcher…");
        const ready = await wakeBackend(API_BASE, (attempt) => {
          setStatusMessage(`Starting the matcher… ${attempt}/6`);
        });
        setBackendReady(ready);
        if (!ready) throw new Error(backendConnectionError());
      }

      setStatusMessage("Searching live jobs and ranking them…");
      const response = await apiFetch("/api/live-jobs", {
        method: "POST",
        body: formData,
      }, setStatusMessage);
      const payload = await readApiResponse(response);
      if (!response.ok) throw new Error(formatApiError(payload) || "Could not find matching jobs.");
      setResult(payload);
      if (!query && payload.query_used) setQuery(payload.query_used);
      setTimeout(() => resultsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 100);
    } catch (err) {
      setError(err.message || backendConnectionError());
      setBackendReady(false);
    } finally {
      setLoading(false);
      setStatusMessage("");
    }
  }

  function backendConnectionError() {
    return "Could not reach the matcher. Wait a minute and try again.";
  }

  async function readApiResponse(response) {
    const text = await response.text();
    if (!text) return {};
    try {
      return JSON.parse(text);
    } catch {
      return { detail: text };
    }
  }

  async function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  async function wakeBackend(base, onAttempt) {
    for (let attempt = 1; attempt <= 6; attempt += 1) {
      onAttempt?.(attempt);
      try {
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), 90000);
        const response = await fetch(`${base}/health`, {
          method: "GET",
          mode: "cors",
          cache: "no-store",
          signal: controller.signal,
        });
        clearTimeout(timer);
        if (response.ok) return true;
      } catch {
        // Free-tier hosts can take a minute to wake.
      }
      await sleep(Math.min(attempt * 5000, 20000));
    }
    return false;
  }

  async function apiFetch(path, options, onStatus, retries = 3) {
    const bases = API_BASE ? [API_BASE, API_BASE_FALLBACK].filter(Boolean) : [""];
    let lastError;

    for (const base of bases) {
      for (let attempt = 1; attempt <= retries; attempt += 1) {
        try {
          onStatus?.(`Searching… ${attempt}/${retries}`);
          const controller = new AbortController();
          const timer = setTimeout(() => controller.abort(), 180000);
          const response = await fetch(`${base}${path}`, {
            ...options,
            mode: "cors",
            cache: "no-store",
            signal: controller.signal,
          });
          clearTimeout(timer);
          return response;
        } catch (err) {
          lastError = err;
          if (attempt < retries) {
            onStatus?.(`Still searching… ${attempt + 1}/${retries}`);
            await sleep(attempt * 6000);
          }
        }
      }
    }

    throw new Error(`${backendConnectionError()} ${lastError?.message || ""}`.trim());
  }

  const hasPartialSources = Boolean(result?.provider_warnings?.length);

  return (
    <div className="page">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">RA</span>
          <span>Resume Analyser</span>
        </div>
        <p className="topbar-note">/ live job ranking</p>
      </header>

      <main>
        <section className="hero">
          <div className="hero-text">
            <p className="hero-kicker">Think your resume can find the right job?</p>
            <h1>Try it.</h1>
            <p className="hero-subtitle">
              Upload a resume. We pull live roles, rank them against you, and show why each one fits.
            </p>
            <div className="hero-meta">
              <span>/ PDF, DOCX or text</span>
              <span>/ Ranked matches</span>
              <span>/ Skills, ATS, and fit</span>
            </div>
          </div>
        </section>

        <form className="layout" onSubmit={findJobs}>
          <section className="panel">
            <div className="section-heading">
              <div className="step-badge">1</div>
              <div>
                <h2>Your resume</h2>
                <p>PDF, DOCX, TXT, or paste the text. That’s enough to start.</p>
              </div>
            </div>

            <label className={`upload-box${resumeFile ? " has-file" : ""}`}>
              <input
                type="file"
                accept=".pdf,.docx,.txt"
                onChange={(e) => setResumeFile(e.target.files?.[0] || null)}
              />
              <div className="upload-icon">
                <UploadIcon />
              </div>
              <strong>{resumeFile ? resumeFile.name : "Drop or click to upload"}</strong>
              <small>PDF, DOCX, or TXT</small>
            </label>

            <div className="or-divider"><span>or paste</span></div>

            <textarea
              ref={resumeTextAreaRef}
              value={resumeText}
              onChange={(e) => setResumeText(e.target.value)}
              placeholder="Paste your resume here…"
            />
          </section>

          <aside className="panel">
            <div className="section-heading">
              <div className="step-badge">2</div>
              <div>
                <h2>Preferences</h2>
                <p>We’ll search every connected job board for you.</p>
              </div>
            </div>

            <div className="two-col">
              <label>
                Looking for
                <select value={opportunityType} onChange={(e) => setOpportunityType(e.target.value)}>
                  <option value="jobs">Jobs</option>
                  <option value="internships">Internships</option>
                  <option value="both">Both</option>
                </select>
              </label>
              <label>
                Country
                <select value={country} onChange={(e) => setCountry(e.target.value)}>
                  <option value="in">India</option>
                  <option value="us">United States</option>
                  <option value="gb">United Kingdom</option>
                  <option value="ca">Canada</option>
                  <option value="au">Australia</option>
                  <option value="sg">Singapore</option>
                </select>
              </label>
            </div>

            <label>
              Posted
              <select value={datePosted} onChange={(e) => setDatePosted(e.target.value)}>
                <option value="all">Any time</option>
                <option value="today">Today</option>
                <option value="3days">Last 3 days</option>
                <option value="week">This week</option>
                <option value="month">This month</option>
              </select>
            </label>

            <div className="role-picker">
              <div className="role-picker-label">Roles to emphasise <small>up to 5, optional</small></div>
              <div className="role-chip-grid">
                {ROLE_OPTIONS.map((role) => {
                  const on = selectedRoles.includes(role);
                  return (
                    <button
                      type="button"
                      key={role}
                      className={`role-chip${on ? " on" : ""}`}
                      onClick={() => toggleRole(role)}
                    >
                      {role}
                    </button>
                  );
                })}
              </div>
            </div>

            <label>
              Extra keywords
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder={selectedRoles.length ? selectedRoles.join(", ") : suggestedQuery}
              />
            </label>

            <button type="submit" className={`btn-primary${loading ? " loading" : ""}`} disabled={loading}>
              {loading ? (statusMessage || "Finding matches…") : "Find matching jobs"}
            </button>

            {!loading && statusMessage && <div className="status-note">{statusMessage}</div>}

            {error && <div className="error">⚠ {error}</div>}
          </aside>
        </form>

        {loading && <LoadingState count={5} />}

        {result && !loading && (
          <section className="results" ref={resultsRef}>
            <div className="results-header">
              <div>
                <span className="eyebrow">Ranked for you</span>
                <h2>Best matches{result.query_used ? ` for “${result.query_used}”` : ""}</h2>
                <p>
                  Scanned {result.jobs_fetched} live roles · showing the top {result.jobs.length}
                </p>
              </div>
            </div>

            {hasPartialSources && (
              <div className="provider-warnings">
                A few job boards didn’t respond this time. You’re still seeing ranked matches from the ones that did.
              </div>
            )}

            <div className="job-grid">
              {result.jobs.map((job) => (
                <JobCard key={`${job.rank}-${job.url || job.title}`} job={job} />
              ))}
            </div>
          </section>
        )}
      </main>
      <NightClock />
    </div>
  );
}

function NightClock() {
  const [now, setNow] = useState(() => new Date());
  useEffect(() => {
    const id = setInterval(() => setNow(new Date()), 30000);
    return () => clearInterval(id);
  }, []);
  const hh = String(now.getHours()).padStart(2, "0");
  const mm = String(now.getMinutes()).padStart(2, "0");
  return <div className="night-clock" aria-hidden="true">{hh}:{mm}</div>;
}

function JobCard({ job }) {
  const isTop = job.rank <= 3;

  return (
    <article className="job-card">
      <div className="job-top">
        <div className={`rank-badge${isTop ? " top" : ""}`}>#{job.rank}</div>
        <div className="job-title-row" style={{ flex: 1 }}>
          <h3>{job.title}</h3>
          <div className="job-meta">
            {job.company && (
              <span className="meta-pill">
                <BriefcaseIcon />
                {job.company}
              </span>
            )}
            {job.location && (
              <span className="meta-pill">
                <MapPinIcon />
                {job.location}
              </span>
            )}
            {job.source && <span className="meta-pill">{job.source}</span>}
            {job.job_type && <span className="meta-pill">{job.job_type}</span>}
          </div>
        </div>
      </div>

      <div className="score-bar-row">
        <div>
          <div className="fit-score">{job.final_score}</div>
          <div className="fit-label">/ 100 fit</div>
        </div>
        <div style={{ flex: 1 }}>
          <div className="bar-track">
            <div className="bar-fill" style={{ width: `${job.final_score}%` }} />
          </div>
        </div>
      </div>

      <div className="metrics">
        <Metric label="Skills" value={`${job.skill_score}%`} />
        <Metric label="Wording" value={`${job.semantic_score}%`} />
        <Metric label="ATS" value={`${job.ats_score}`} />
      </div>

      <p className="summary">{job.summary}</p>

      {job.experience_fit?.warning && (
        <div className="warning">⚠ {job.experience_fit.warning}</div>
      )}

      <SkillRow title="Matched skills" values={job.matched_skills} tone="good" />
      <SkillRow title="Missing skills" values={job.missing_skills} tone="bad" />

      <details>
        <summary>Why this score, and how to improve</summary>
        <div className="details-body">
          <p>{job.analysis?.overall_explanation}</p>
          {job.analysis?.improvements?.length > 0 && (
            <>
              <strong>Improvements</strong>
              <ul>
                {job.analysis.improvements.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </>
          )}
          {job.analysis?.ats_suggestions?.length > 0 && (
            <>
              <strong>ATS suggestions</strong>
              <ul>
                {job.analysis.ats_suggestions.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </>
          )}
        </div>
      </details>

      {job.url && (
        <a className="apply" href={job.url} target="_blank" rel="noreferrer">
          Open job posting <ExternalLinkIcon />
        </a>
      )}
    </article>
  );
}

function Metric({ label, value }) {
  return (
    <div className="metric">
      <strong>{value}</strong>
      <span>{label}</span>
    </div>
  );
}

function SkillRow({ title, values, tone }) {
  if (!values?.length) return null;
  return (
    <div className="skills">
      <div className="skills-label">{title}</div>
      <div>
        {values.slice(0, 8).map((item) => (
          <span className={`chip ${tone}`} key={item}>{item}</span>
        ))}
      </div>
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);
