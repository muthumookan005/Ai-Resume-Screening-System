import re
import io
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st
import fitz
from docx import Document
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# PAGE CONFIGURATION & STYLING
# -----------------------------
st.set_page_config(
    page_title="TalentLens | AI Resume Screening",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: #f5f7fb; }
    [data-testid="stSidebar"] { background: #111827; }
    [data-testid="stSidebar"] * { color: #eef2ff !important; }
    .hero {
        background: linear-gradient(120deg, #111827 0%, #243b67 65%, #315b9a 100%);
        padding: 28px 32px; border-radius: 18px; color: white; margin-bottom: 22px;
    }
    .hero h1 { color: white; font-family: 'Manrope', sans-serif; font-size: 30px; margin: 0; }
    .hero p { color: #dbeafe; margin: 8px 0 0 0; font-size: 14px; }
    .section-title { font-family: 'Manrope', sans-serif; color: #172033; font-size: 20px; font-weight: 800; margin: 8px 0 12px; }
    .metric-card {
        background: white; border: 1px solid #e7ebf2; border-radius: 14px;
        padding: 17px 18px; box-shadow: 0 3px 12px rgba(15,23,42,.035);
    }
    .metric-label { color: #64748b; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: .06em; }
    .metric-value { color: #172033; font-family: 'Manrope', sans-serif; font-size: 27px; font-weight: 800; margin-top: 5px; }
    .metric-note { color: #64748b; font-size: 12px; margin-top: 2px; }
    .panel {
        background: white; border: 1px solid #e7ebf2; border-radius: 14px;
        padding: 20px; margin-bottom: 14px;
    }
    .pill { display:inline-block; padding:4px 9px; border-radius:999px; background:#eaf2ff; color:#2456a6; font-size:11px; font-weight:700; margin:2px 4px 2px 0; }
    .muted { color:#64748b; font-size:13px; }
    div.stButton > button[kind="primary"] {
        background: #315bdb; border: 0; border-radius: 10px; font-weight: 700; padding: .65rem 1.2rem;
    }
    div[data-testid="stFileUploader"] { background: #f8faff; border: 1px dashed #9bb4e8; border-radius: 12px; padding: 10px; }
    h2, h3 { font-family: 'Manrope', sans-serif !important; color: #172033; }
</style>
""", unsafe_allow_html=True)


# -----------------------------
# MODEL AND DOCUMENT HELPERS
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def read_pdf(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> str:
    try:
        with fitz.open(stream=uploaded_file.getvalue(), filetype="pdf") as pdf:
            return "\n".join(page.get_text() for page in pdf)
    except Exception as exc:
        raise ValueError(f"Unable to read PDF: {exc}") from exc


def read_docx(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> str:
    try:
        document = Document(io.BytesIO(uploaded_file.getvalue()))
        parts = [paragraph.text for paragraph in document.paragraphs]
        for table in document.tables:
            for row in table.rows:
                parts.append(" ".join(cell.text for cell in row.cells))
        return "\n".join(parts)
    except Exception as exc:
        raise ValueError(f"Unable to read DOCX: {exc}") from exc


def read_resume(uploaded_file) -> str:
    filename = uploaded_file.name.lower()
    if filename.endswith(".pdf"):
        return read_pdf(uploaded_file)
    if filename.endswith(".docx"):
        return read_docx(uploaded_file)
    return ""


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


SKILLS = [
    "python", "java", "c++", "javascript", "html", "css", "sql", "mysql",
    "mongodb", "machine learning", "deep learning", "artificial intelligence",
    "natural language processing", "nlp", "tensorflow", "pytorch", "keras",
    "pandas", "numpy", "data science", "data analysis", "power bi", "tableau",
    "aws", "azure", "docker", "kubernetes", "git", "github", "django",
    "flask", "fastapi", "scikit-learn", "streamlit", "excel", "linux",
]


def find_skills(text: str) -> List[str]:
    normalized = clean_text(text)
    found = []
    for skill in SKILLS:
        pattern = r"(?<!\w)" + re.escape(skill) + r"(?!\w)"
        if re.search(pattern, normalized):
            found.append(skill)
    return found


def find_experience(text: str) -> float:
    normalized = text.lower()
    patterns = [
        r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)\b",
        r"(\d+(?:\.\d+)?)\s*(?:year|yr)\s+of\s+experience",
    ]
    values = []
    for pattern in patterns:
        values.extend(re.findall(pattern, normalized))
    return max((float(value) for value in values), default=0.0)


def get_similarity(resume: str, job: str, model) -> float:
    vectors = model.encode([resume, job], convert_to_numpy=True)
    return float(cosine_similarity([vectors[0]], [vectors[1]])[0][0])


def analyze_candidate(resume: str, job: str, required_skills: List[str],
                      required_experience: float, model) -> Dict:
    cleaned_resume = clean_text(resume)
    candidate_skills = find_skills(cleaned_resume)
    matched = [skill for skill in required_skills if skill in candidate_skills]
    missing = [skill for skill in required_skills if skill not in candidate_skills]

    skill_score = (len(matched) / len(required_skills) * 100) if required_skills else 0.0
    candidate_experience = find_experience(cleaned_resume)
    experience_score = (
        min(candidate_experience / required_experience, 1.0) * 100
        if required_experience > 0 else 100.0
    )
    similarity_score = max(0.0, get_similarity(cleaned_resume, clean_text(job), model) * 100)

    # Retains the weighting from the supplied project.
    final_score = skill_score * 0.50 + experience_score * 0.20 + similarity_score * 0.30
    recommendation = (
        "Strong Match" if final_score >= 80
        else "Moderate Match" if final_score >= 60
        else "Low Match"
    )
    return {
        "Skill Score": round(skill_score, 2),
        "Experience Score": round(experience_score, 2),
        "AI Similarity": round(similarity_score, 2),
        "Final Score": round(final_score, 2),
        "Experience": candidate_experience,
        "Matched Skills": matched,
        "Missing Skills": missing,
        "Recommendation": recommendation,
    }


# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.markdown("## ◈ TalentLens")
    st.caption("AI-assisted recruitment workspace")
    st.divider()
    page = st.radio("NAVIGATION", ["Screening workspace", "About this project"], label_visibility="visible")
    st.divider()
    st.markdown("**Screening model**")
    st.caption("all-MiniLM-L6-v2")
    st.caption("Skill + experience + semantic similarity")
    st.divider()
    st.caption("Portfolio project • Educational use")


# -----------------------------
# HEADER
# -----------------------------
st.markdown("""
<div class="hero">
  <h1>AI Resume Screening System</h1>
  <p>Review resumes against a job description, understand skill alignment, and compare candidates in one workspace.</p>
</div>
""", unsafe_allow_html=True)

if page == "About this project":
    st.markdown('<div class="section-title">About TalentLens</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="panel">
      <h3>What this application does</h3>
      <p>It extracts text from PDF and DOCX resumes, identifies skills from a predefined vocabulary,
      estimates stated experience, and compares resume text with a job description using sentence embeddings.</p>
      <h3>Scoring approach</h3>
      <ul>
        <li>Skill score: 50%</li>
        <li>Experience score: 20%</li>
        <li>Semantic similarity: 30%</li>
      </ul>
      <p class="muted">Scores are experimental decision-support signals, not validated hiring assessments.
      They can be affected by resume formatting, missing text, keyword vocabulary and model limitations.
      Do not use this tool as the sole basis for hiring decisions.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# -----------------------------
# INPUT WORKSPACE
# -----------------------------
left, right = st.columns([1.05, 0.95], gap="large")
with left:
    st.markdown('<div class="section-title">01 · Job requirements</div>', unsafe_allow_html=True)
    with st.container(border=True):
        job_description = st.text_area(
            "Paste the job description",
            height=230,
            placeholder="Example: We are looking for a Python developer with experience in machine learning, SQL and NLP...",
            label_visibility="collapsed",
        )
        st.caption("The system detects skills and stated years of experience from this text.")

with right:
    st.markdown('<div class="section-title">02 · Candidate resumes</div>', unsafe_allow_html=True)
    with st.container(border=True):
        resumes = st.file_uploader(
            "Drop PDF or DOCX resumes here",
            type=["pdf", "docx"],
            accept_multiple_files=True,
            help="Use fictional or anonymized resumes for demonstrations. Avoid uploading sensitive personal information.",
        )
        if resumes:
            st.success(f"{len(resumes)} resume(s) ready for analysis")
        else:
            st.info("Upload one or more PDF or DOCX files to begin.")

run_screening = st.button("✦  Analyze candidates", type="primary", use_container_width=True)

if run_screening:
    if not job_description.strip():
        st.error("Please enter a job description before screening.")
        st.stop()
    if not resumes:
        st.error("Please upload at least one PDF or DOCX resume.")
        st.stop()

    required_skills = find_skills(job_description)
    required_experience = find_experience(job_description)

    with st.spinner("Loading language model. The first run may take a little longer..."):
        try:
            model = load_model()
        except Exception as exc:
            st.error(f"Could not load the Sentence Transformers model: {exc}")
            st.info("Check your internet connection and installed PyTorch / sentence-transformers packages.")
            st.stop()

    results = []
    progress = st.progress(0, text="Reading resumes...")
    for index, resume_file in enumerate(resumes):
        try:
            resume_text = read_resume(resume_file)
            if not resume_text.strip():
                st.warning(f"No selectable text found in {resume_file.name}. Scanned PDFs may need OCR.")
                continue
            result = analyze_candidate(
                resume_text, job_description, required_skills, required_experience, model
            )
            result["Candidate"] = resume_file.name
            results.append(result)
        except Exception as exc:
            st.warning(f"Could not process {resume_file.name}: {exc}")
        progress.progress((index + 1) / len(resumes), text=f"Processed {index + 1} of {len(resumes)} resumes")
    progress.empty()

    if not results:
        st.error("No readable resumes could be analyzed.")
        st.stop()

    results.sort(key=lambda item: item["Final Score"], reverse=True)
    st.session_state["screening_results"] = results
    st.session_state["required_skills"] = required_skills
    st.session_state["required_experience"] = required_experience


# -----------------------------
# RESULTS DASHBOARD
# -----------------------------
if "screening_results" in st.session_state:
    results = st.session_state["screening_results"]
    required_skills = st.session_state["required_skills"]
    required_experience = st.session_state["required_experience"]

    st.markdown("---")
    st.markdown('<div class="section-title">03 · Screening overview</div>', unsafe_allow_html=True)

    strong_count = sum(row["Recommendation"] == "Strong Match" for row in results)
    avg_score = sum(row["Final Score"] for row in results) / len(results)
    avg_similarity = sum(row["AI Similarity"] for row in results) / len(results)

    m1, m2, m3, m4 = st.columns(4)
    metrics = [
        (m1, "RESUMES REVIEWED", str(len(results)), "Documents processed"),
        (m2, "STRONG MATCHES", str(strong_count), "Score of 80 or above"),
        (m3, "AVERAGE MATCH", f"{avg_score:.1f}%", "Weighted experimental score"),
        (m4, "AVG. AI SIMILARITY", f"{avg_similarity:.1f}%", "Semantic text similarity"),
    ]
    for col, label, value, note in metrics:
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">{label}</div>'
                f'<div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',
                unsafe_allow_html=True,
            )

    st.write("")
    tab_rank, tab_detail, tab_requirements = st.tabs(["Candidate ranking", "Candidate details", "Job analysis"])

    table_rows = []
    for rank, item in enumerate(results, start=1):
        table_rows.append({
            "Rank": rank,
            "Candidate file": item["Candidate"],
            "Final score": item["Final Score"],
            "Skill score": item["Skill Score"],
            "Experience score": item["Experience Score"],
            "AI similarity": item["AI Similarity"],
            "Recommendation": item["Recommendation"],
        })
    result_df = pd.DataFrame(table_rows)

    with tab_rank:
        st.markdown("#### Candidate leaderboard")
        st.caption("Rankings are based on the experimental weighted score. Review each resume manually.")
        st.dataframe(
            result_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Final score": st.column_config.ProgressColumn("Final score", min_value=0, max_value=100, format="%.1f%%"),
                "Skill score": st.column_config.NumberColumn("Skill score", format="%.1f%%"),
                "Experience score": st.column_config.NumberColumn("Experience score", format="%.1f%%"),
                "AI similarity": st.column_config.NumberColumn("AI similarity", format="%.1f%%"),
            },
        )
        csv_bytes = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "↓  Download candidate ranking (CSV)",
            data=csv_bytes,
            file_name="candidate_ranking.csv",
            mime="text/csv",
        )
        chart_data = result_df.set_index("Candidate file")[["Final score"]]
        st.markdown("#### Candidate score comparison")
        st.bar_chart(chart_data)

    with tab_detail:
        st.markdown("#### Review individual candidates")
        selected_candidate = st.selectbox("Select a candidate", [item["Candidate"] for item in results])
        selected = next(item for item in results if item["Candidate"] == selected_candidate)
        c1, c2 = st.columns([1, 1])
        with c1:
            st.metric("Overall experimental score", f'{selected["Final Score"]:.1f}%')
            st.metric("AI semantic similarity", f'{selected["AI Similarity"]:.1f}%')
        with c2:
            st.metric("Skill alignment", f'{selected["Skill Score"]:.1f}%')
            st.metric("Experience stated in resume", f'{selected["Experience"]:.1f} years')
        st.info(f'Recommendation label: **{selected["Recommendation"]}** — use this only as a review aid.')
        matched_col, missing_col = st.columns(2)
        with matched_col:
            st.markdown("**Detected required skills**")
            if selected["Matched Skills"]:
                st.success(" · ".join(selected["Matched Skills"]))
            else:
                st.write("No required skills detected as matched.")
        with missing_col:
            st.markdown("**Required skills not detected**")
            if selected["Missing Skills"]:
                st.warning(" · ".join(selected["Missing Skills"]))
            else:
                st.write("No missing required skills detected.")

    with tab_requirements:
        st.markdown("#### Extracted job requirements")
        st.write(f"**Detected skills ({len(required_skills)}):**")
        if required_skills:
            st.write(" · ".join(required_skills))
        else:
            st.warning("No skills from the built-in vocabulary were detected. Consider adding explicit skill terms to the job description.")
        st.write(f"**Required experience detected:** {required_experience:g} years")
        st.caption("The skill vocabulary and experience parser are rule-based and may miss equivalent terms or unusual wording.")

    st.markdown("""
    <div class="panel">
      <strong>Responsible-use note</strong>
      <p class="muted">This prototype provides experimental text-matching indicators. It does not assess a person's ability,
      potential, personality, or fairness-related attributes. Resume parsing can miss information, and similarity scores
      are not probabilities of job performance. A human reviewer must assess candidates using consistent, job-related criteria.</p>
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown("---")
    st.markdown('<div class="section-title">How the screening workflow works</div>', unsafe_allow_html=True)
    steps = st.columns(3)
    for col, number, title, description in [
        (steps[0], "01", "Add job details", "Paste the job description to identify required skills and experience."),
        (steps[1], "02", "Upload resumes", "Add PDF or DOCX resumes. Use demo or anonymized files."),
        (steps[2], "03", "Review insights", "Compare extracted skills, semantic similarity and candidate details."),
    ]:
        with col:
            st.markdown(
                f'<div class="panel"><div class="metric-label">STEP {number}</div>'
                f'<h3>{title}</h3><p class="muted">{description}</p></div>',
                unsafe_allow_html=True,
            )
    st.caption("Tip: For a portfolio demonstration, prepare two or three fictional resumes with different skill profiles.")
