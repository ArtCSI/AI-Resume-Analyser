# AI Resume Analyzer - Setup Guide

## 📋 Project Overview
An intelligent resume analysis system using a **custom-trained ANN (Artificial Neural Network)** to match resumes against job descriptions with automatic skill extraction, AI-powered feedback, and interactive web interface.

---

## ⚙️ Requirements
- **Python**: 3.10, 3.11, 3.12, or 3.13
- **RAM**: 4GB minimum
- **Storage**: 3GB free space

---

## 🚀 Complete Setup Commands - Follow these commands in vscode terminal and if any errors are found then , debug using commands mentioned after setup commands

### Windows (Command Prompt)
```cmd
cd AI-Resume-Analyzer
python --version
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

### Windows (PowerShell)
```powershell
cd AI-Resume-Analyzer
python --version
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

### macOS / Linux
```bash
cd AI-Resume-Analyzer
python3 --version
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

**✅ Success:** Browser opens automatically to `http://localhost:8501`

**⏱️ Installation Time:** 5-10 minutes (one-time setup)

---

## 🧪 Quick Test

1. **Upload Resume**: Click "Choose your resume file" → Select any PDF/DOCX

2. **Paste Job Description**:
   ```
   Python Developer position requiring:
   - Python, Django, Flask
   - SQL, PostgreSQL
   - RESTful APIs
   - Git, Docker, AWS
   - 3+ years experience
   ```

3. **Click "🚀 Analyze Resume"** → Wait 10-15 seconds

4. **Verify Results**:
   - ✅ Match Score percentage
   - ✅ Matched Skills (green tags)
   - ✅ Missing Skills (red tags)
   - ✅ Detailed feedback sections

---

## 🐛 Troubleshooting

### "python: command not found"
```bash
python3 --version
python3 -m venv venv
source venv/bin/activate
```

### "No module named 'streamlit'"
```bash
# Make sure (venv) appears in your prompt
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac/Linux

pip install -r requirements.txt
```

### TensorFlow installation fails
```bash
pip install --no-cache-dir tensorflow==2.20.0
pip install -r requirements.txt
```

### Port 8501 already in use
```bash
streamlit run app.py --server.port 8502
```

### PowerShell execution policy error
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\venv\Scripts\Activate.ps1
```

### Model files not found
```bash
# Check files exist:
dir *.keras *.pkl        # Windows
ls *.keras *.pkl         # Mac/Linux

# Should show: resume_score_ann.keras, resume_scaler.pkl
```

---

## 📁 Project Structure
```
AI-Resume-Analyzer/
├── app.py                      # Main application
├── matcher.py                  # ANN model logic
├── feedback.py                 # Feedback generation
├── resume_parser.py            # Document parsing
├── resume_score_ann.keras     # Pre-trained model
├── resume_scaler.pkl          # Feature scaler
├── requirements.txt           # Dependencies
└── .env                        # API credentials
└── train_ann_real_data.py      # training file                
```

---

## 🧠 ANN Model Details

- **Input Features**: 4 (skill match ratio, semantic similarity, resume/JD skills count)
- **Architecture**: Dense(64) → Dense(32) → Dense(16) → Output(1)
- **Training**: 3000 samples from real resumes
- **Output**: Match score (0-100%)

**Pre-trained model included** - No training required to run the application.

---

## 📊 Score Interpretation

| Score | Meaning |
|-------|---------|
| 85-100% | Excellent match |
| 70-84% | Strong match |
| 60-69% | Good match |
| 50-59% | Moderate match |
| <50% | Limited match |

---

## ✅ Verification Checklist

- [ ] Python 3.10+ installed
- [ ] Virtual environment activated (see `(venv)` in prompt)
- [ ] Dependencies installed successfully
- [ ] App opens at http://localhost:8501
- [ ] Can upload resume and paste job description
- [ ] Analysis completes with results displayed

---

## 📝 Quick Reference

**First Time Setup:**
```bash
cd AI-Resume-Analyzer
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac/Linux
pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

**Every Subsequent Run:**
```bash
cd AI-Resume-Analyzer
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac/Linux
streamlit run app.py
```

**Stop App:** Press `Ctrl+C`

**Deactivate venv:** Type `deactivate`

---

## ⚠️ Important Notes

- Always activate virtual environment before running (see `(venv)` in prompt)
- First run downloads NLP models (~100MB, 1-2 minutes)
- TensorFlow/NumPy warnings are normal and can be ignored
- Analysis takes 10-15 seconds per resume
- API credentials included in `.env` file

---

## 🎯 Technologies

- Streamlit 1.39.0 (UI)
- TensorFlow 2.20.0 (ANN)
- Sentence Transformers 3.3.1 (NLP)
- LangChain + Groq API (AI feedback)
- pdfplumber, docx2txt (Parsing)

---

## ⏱️ Time Estimates

| Task | Time |
|------|------|
| Setup | 5-10 min |
| First launch | 2 min |
| Analysis | 10-15 sec |
| **Total** | **10-15 min** |

---

**Tested on:** Windows 11, macOS 14, Ubuntu 22.04  
**Python:** 3.10.11, 3.11.5, 3.12.0, 3.13.0

---

## 🚀 TL;DR

```bash
cd AI-Resume-Analyzer
python -m venv venv
venv\Scripts\activate                    # Windows
source venv/bin/activate                 # Mac/Linux
pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

**Open browser at http://localhost:8501 and test! ✅**