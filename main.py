from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import os
import shutil
import json
import numpy as np
import re

from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
API_KEY = os.getenv("GEMINI_API_KEY")
os.environ["GOOGLE_API_KEY"] =API_KEY 
model = init_chat_model("google_genai:gemini-2.5-flash-lite")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)
def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
def keyword_score(resume, job_desc):
    resume_words = re.findall(r"\w+", resume.lower())
    jd_words = re.findall(r"\w+", job_desc.lower())
    if not jd_words:
        return 0
    match_count = sum(1 for word in jd_words if word in resume_words)
    return match_count / len(jd_words)
def semantic_score(resume, job_desc):
    emb1 = embeddings.embed_query(resume[:2000])
    emb2 = embeddings.embed_query(job_desc[:2000])
    return cosine_similarity(emb1, emb2)
def extract_resume_text(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text = " ".join([doc.page_content for doc in documents])
    return clean_text(text)
def generate_job_description(resume):
    prompt = f"""
    Identify best matching job role from this resume.

    Resume:
    {resume[:2000]}

    Return STRICT format:

    Role: <role name>
    Skills: <comma separated skills>
    Responsibilities: <short description>
    """

    return model.invoke(prompt).content
def extract_skills(text):
    words = re.findall(r"[a-zA-Z+#.]+", text.lower())
    return list(set(words))
@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    resume = extract_resume_text(file_path)
    job_description = generate_job_description(resume)
    key_score = keyword_score(resume, job_description)
    sem_score = semantic_score(resume, job_description)

    final_score = int((0.6 * key_score + 0.4 * sem_score) * 100)
    final_score = min(final_score, 95)  
    prompt = f"""
    You are a strict ATS analyzer.

    RULES:
    - Do NOT give 100% unless perfect match
    - Missing skills must be real
    - Role fit must be % value

    Resume:
    {resume[:3000]}

    Job Description:
    {job_description}

    Return ONLY JSON:

    {{
      "detected_role": "",
      "extracted_skills": [],
      "missing_skills": [],
      "role_fit": "",
      "career_level": "",
      "improvements": [],
      "project_suggestions": [],
      "ats_tips": []
    }}
    """

    llm_output = model.invoke(prompt).content
    default_analysis = {
        "detected_role": "",
        "extracted_skills": [],
        "missing_skills": [],
        "role_fit": "0%",
        "career_level": "",
        "improvements": [],
        "project_suggestions": [],
        "ats_tips": []
    }
    try:
        llm_output = llm_output.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(llm_output)
    except:
        parsed = {}
    for key in default_analysis:
        if key not in parsed or parsed[key] is None:
            parsed[key] = default_analysis[key]
    for key in ["extracted_skills", "missing_skills", "improvements", "project_suggestions", "ats_tips"]:
        if not isinstance(parsed[key], list):
            parsed[key] = []
    if isinstance(parsed["role_fit"], (int, float)):
        parsed["role_fit"] = f"{int(parsed['role_fit'])}%"

    if parsed["role_fit"] == "":
        parsed["role_fit"] = f"{final_score}%"
    resume_skills = set(extract_skills(resume))
    jd_skills = set(extract_skills(job_description))
    missing = list(jd_skills - resume_skills)
    parsed["missing_skills"] = list(set(parsed["missing_skills"] + missing))[:10]
    parsed["extracted_skills"] = parsed["extracted_skills"][:15]
    parsed["improvements"] = parsed["improvements"][:5]
    parsed["project_suggestions"] = parsed["project_suggestions"][:5]
    parsed["ats_tips"] = parsed["ats_tips"][:5]
    return JSONResponse({
        "ats_score": final_score,
        "keyword_score": round(key_score * 100, 2),
        "semantic_score": round(sem_score * 100, 2),
        "job_description": job_description,
        "analysis": parsed
    })
