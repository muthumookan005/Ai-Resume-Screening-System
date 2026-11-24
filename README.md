# AI Resume Screening System

A Streamlit portfolio project that extracts text from PDF/DOCX resumes and compares resume content with a job description using skill matching and sentence-embedding similarity.

## Features
- Multiple PDF and DOCX resume uploads
- Job description input
- Skill detection using a predefined vocabulary
- Basic experience extraction
- Semantic similarity using `all-MiniLM-L6-v2`
- Candidate ranking, per-candidate breakdown and CSV export

## Tech stack
Python, Streamlit, Pandas, PyMuPDF, python-docx, Sentence Transformers, scikit-learn.

## Run locally
1. Create and activate a virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the app:
   ```bash
   streamlit run app.py
   ```

The first model load may take time and requires an internet connection to download the model if it is not cached.

## Responsible use
This is an educational prototype, not a validated hiring assessment. Scores are experimental text-matching indicators and must not be used as the sole basis for employment decisions. Resume parsing may miss information, and the skill vocabulary is limited. Use fictional or anonymized resumes for demos; do not publish personal candidate data.
