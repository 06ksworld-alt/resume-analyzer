import streamlit as st
import pdfplumber
from sentence_transformers import SentenceTransformer, util

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Skills & roles
skills_list = [
    "python", "machine learning", "data science", "sql",
    "deep learning", "nlp", "pandas", "numpy", "tensorflow"
]

job_roles = {
    "Data Scientist": "python machine learning data science pandas numpy",
    "ML Engineer": "python machine learning tensorflow deep learning",
    "Data Analyst": "sql python pandas numpy data analysis"
}

# Extract text
def extract_text(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            if page.extract_text():
                text += page.extract_text()
    return text.lower()

# Similarity
def get_similarity(resume_text, job_desc):
    emb1 = model.encode(resume_text, convert_to_tensor=True)
    emb2 = model.encode(job_desc, convert_to_tensor=True)
    score = util.cos_sim(emb1, emb2)
    return float(score[0][0]) * 100

# UI
st.title("🚀 AI Resume Analyzer Pro")
st.markdown("### Get instant insights on your resume 💼")

uploaded_file = st.file_uploader("Upload your Resume (PDF)", type=["pdf"])

if uploaded_file:
    resume_text = extract_text(uploaded_file)

    st.subheader("📊 Analysis Result")

    # Skills
    found_skills = [skill for skill in skills_list if skill in resume_text]
    st.success(f"Skills Found: {', '.join(found_skills)}")

    # Job match
    best_role = None
    best_score = 0

    for role, desc in job_roles.items():
        score = get_similarity(resume_text, desc)
        if score > best_score:
            best_score = score
            best_role = role

    st.write(f"🎯 Best Role: {best_role}")
    st.write(f"📈 Match Score: {round(best_score, 2)}%")
    resume_score = min(int(best_score), 100)
    st.metric("Resume Score", f"{resume_score}/100")


    # Missing skills
    missing = list(set(skills_list) - set(found_skills))
    st.write("⚠️ Missing Skills:", missing[:5])




