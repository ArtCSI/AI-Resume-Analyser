# AI Powered Resume Analyser with Personalized Feedback

## 📌 Project Overview
This project is an AI-powered Resume Analyser that:
- Extracts text from resumes (PDF/DOCX).
- Compares resume with a given Job Description (JD).
- Calculates a Match % score.
- Provides personalized recruiter-style feedback using AI.

## 🚀 Features
- Resume text extraction using `pdfplumber` & `docx2txt`.
- Semantic similarity scoring with `sentence-transformers`.
- Personalized recruiter-style feedback using OpenAI GPT.
- Simple web app interface with **Streamlit**.

## 🛠️ Installation
```bash
git clone https://github.com/yourusername/resume_analyser.git
cd resume_analyser
python -m venv venv
venv\Scripts\activate   # (Windows)
source venv/bin/activate  # (Mac/Linux)
pip install -r requirements.txt
