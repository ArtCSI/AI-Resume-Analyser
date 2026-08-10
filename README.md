# 🤖 AI Resume Analyzer

An AI-powered resume analysis and job-matching application that evaluates how well a resume aligns with a given job description using **NLP, semantic similarity, skill extraction, and a custom Artificial Neural Network (ANN)**.

The system accepts PDF/DOCX resumes, extracts relevant information and skills, compares them against a job description, generates an ANN-based match score, identifies matched and missing skills, and provides both rule-based and LLM-powered feedback.

---

## ✨ Features

### 📄 Resume Parsing & Validation

* Upload resumes in **PDF or DOCX** format
* Extract resume text automatically
* Detect common resume sections
* Analyze document statistics such as word, character, and line counts
* Identify potential formatting issues and provide suggestions

### 🎯 Skill Matching

* Extract technical and professional skills from resumes and job descriptions
* Identify:

  * ✅ Matched skills
  * ⚠️ Missing skills
* Calculate a skill-match ratio between the resume and job requirements

### 🧠 Semantic Similarity

* Uses **Sentence Transformers (`all-MiniLM-L6-v2`)** to generate text embeddings
* Measures semantic similarity between resume content and the job description
* Goes beyond simple keyword matching by considering contextual similarity

### 🤖 ANN-Based Resume Matching

The project uses a custom-trained Artificial Neural Network to generate a resume–job match score.

The model uses four features:

* Skill match ratio
* Semantic similarity
* Number of skills identified in the resume
* Number of skills identified in the job description

Current architecture:

```text
Input Features (4)
        ↓
Dense Layer (64)
        ↓
Dense Layer (32)
        ↓
Dense Layer (16)
        ↓
Output Layer
        ↓
Match Score
```

The training pipeline is provided in `train_ann_real_data.py`.

### 💡 Automated Feedback

The system provides two forms of feedback:

**Rule-based feedback**

* Highlights missing skills
* Suggests relevant improvements
* Recommends stronger alignment with the job description

**AI-powered feedback**

* Uses a LangChain/Groq-based LLM workflow
* Generates additional resume improvement suggestions
* Provides role-specific feedback based on the uploaded resume and job description

### 📊 Explainable Results

The application displays:

* Overall AI match score
* ANN feature breakdown
* Matched skills
* Missing skills
* Detailed analysis
* AI-powered insights
* Extracted resume text
* Actionable recommendations

---

## 🏗️ System Architecture

```text
                 ┌─────────────────────┐
                 │   Resume PDF/DOCX   │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │   Resume Parser     │
                 │ PDF / DOCX / Text   │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │  Skill Extraction   │
                 └──────────┬──────────┘
                            │
                ┌───────────┴───────────┐
                ▼                       ▼
       ┌────────────────┐      ┌─────────────────┐
       │ Skill Matching │      │ Sentence        │
       │                │      │ Transformer     │
       └───────┬────────┘      └────────┬────────┘
               │                        │
               └───────────┬────────────┘
                           ▼
                ┌─────────────────────┐
                │ ANN Feature Vector  │
                └──────────┬──────────┘
                           ▼
                ┌─────────────────────┐
                │   ANN Model         │
                │ Resume-JD Scoring   │
                └──────────┬──────────┘
                           │
                ┌──────────┴──────────┐
                ▼                     ▼
       ┌────────────────┐    ┌─────────────────┐
       │ Rule-Based     │    │ LLM-Based       │
       │ Feedback       │    │ Feedback        │
       └────────┬───────┘    └────────┬────────┘
                │                     │
                └──────────┬──────────┘
                           ▼
                ┌─────────────────────┐
                │ Interactive Results │
                │ Score + Skills +    │
                │ Feedback + Actions  │
                └─────────────────────┘
```

---

## 🛠️ Tech Stack

| Component             | Technology            |
| --------------------- | --------------------- |
| Frontend / UI         | Streamlit             |
| Programming Language  | Python                |
| Machine Learning      | TensorFlow / Keras    |
| NLP                   | Sentence Transformers |
| Embeddings            | `all-MiniLM-L6-v2`    |
| Traditional ML        | Scikit-learn          |
| LLM Integration       | LangChain + Groq      |
| PDF Parsing           | pdfplumber            |
| DOCX Parsing          | docx2txt              |
| Data Processing       | NumPy, Pandas         |
| Visualization         | Matplotlib, Seaborn   |
| Environment Variables | python-dotenv         |

---

## 📁 Project Structure

```text
AI-Resume-Analyser/
│
├── app.py
│   └── Main Streamlit application and user interface
│
├── matcher.py
│   └── Skill extraction, semantic similarity and ANN matching logic
│
├── resume_parser.py
│   └── PDF/DOCX parsing and resume validation
│
├── feedback.py
│   └── Rule-based and AI-powered feedback generation
│
├── train_ann_real_data.py
│   └── ANN training pipeline
│
├── requirements.txt
│   └── Python dependencies
│
├── .gitignore
│   └── Excludes secrets, environments and generated model/data files
│
└── README.md
```

> Trained model artifacts and datasets are intentionally excluded from the repository. The training pipeline is provided separately in `train_ann_real_data.py`.

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/ArtCSI/AI-Resume-Analyser.git
cd AI-Resume-Analyser
```

### 2. Create a virtual environment

#### Windows

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

#### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root.

```text
GROQ_API_KEY=your_api_key_here
```

**Do not commit `.env` or API keys to GitHub.**

### 5. Run the application

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

---

## 🧪 Using the Application

### Step 1 — Upload Resume

Upload a resume in:

* PDF
* DOCX

The application parses the document and can perform basic file-quality analysis.

### Step 2 — Add Job Description

Paste the complete job description into the job-description field.

### Step 3 — Analyze

Click:

```text
🚀 Analyze Resume
```

The system performs the following pipeline:

```text
1. Resume parsing
2. Skill extraction
3. ANN-based similarity computation
4. ANN feature analysis
5. Rule-based feedback generation
6. AI-powered feedback generation
```

### Step 4 — Review Results

The application displays:

* AI Resume Match Score
* ANN feature breakdown
* Matched skills
* Missing skills
* Detailed analysis
* AI-powered insights
* Quick action items

---

## 🧠 Machine Learning Pipeline

The matching system combines multiple signals rather than relying exclusively on keyword overlap.

### Feature 1 — Skill Match Ratio

Measures the proportion of job-description skills that are also identified in the resume.

### Feature 2 — Semantic Similarity

Resume and job-description text are converted into embeddings using a Sentence Transformer and compared using semantic similarity.

### Feature 3 — Resume Skill Count

Number of recognized skills extracted from the resume.

### Feature 4 — Job Description Skill Count

Number of recognized skills extracted from the job description.

These features are passed into the ANN to produce the final match score.

---

## 📈 Current Model

The ANN currently uses:

```text
4 input features
      ↓
Dense(64)
      ↓
Dense(32)
      ↓
Dense(16)
      ↓
Output(1)
```

The output is converted into a **0–100 resume–job match score**.

The model-training workflow is available in:

```text
train_ann_real_data.py
```

---

## 🔍 Why This Approach?

A resume can contain relevant experience even when the exact wording of a job description is different.

For example:

```text
Resume:
"Built REST APIs using Flask"

Job Description:
"Experience developing backend web services"
```

A pure keyword matcher may miss part of this relationship.

Semantic embeddings provide contextual similarity, while explicit skill extraction provides an interpretable view of which requirements are directly matched or missing.

The ANN combines these signals into a single matching score.

---

## ⚠️ Current Limitations

This project is an evolving AI/ML application and currently has several limitations:

* Skill extraction relies heavily on a predefined skill vocabulary.
* Synonyms and less common technical terminology may not always be detected.
* Resume/job-description matching can be affected by document length and formatting.
* The current ANN training pipeline uses engineered features and generated training targets rather than a large professionally labeled recruitment dataset.
* Match scores should be interpreted as an analytical estimate rather than an actual hiring probability.
* LLM-generated feedback depends on external API availability and model behavior.

These limitations provide opportunities for future experimentation and improvement.

---

## 🔬 Planned Improvements

### Model & Evaluation

* Build a manually labeled resume–job-description evaluation dataset
* Compare against keyword and TF-IDF baselines
* Benchmark embedding-only versus ANN-based approaches
* Evaluate using MAE, RMSE and correlation metrics
* Analyze model failure cases

### NLP Improvements

* Improve skill normalization and synonym detection
* Expand contextual skill extraction
* Improve handling of long resumes and job descriptions
* Introduce section-aware resume analysis

### Explainability

* Provide a clearer breakdown of why a match score was generated
* Identify the contribution of individual matching features
* Highlight evidence from relevant resume sections

### Product Improvements

* Generate downloadable analysis reports
* Add improved visualization of matching results
* Support additional document formats
* Add job-role-specific analysis

---

## 🎯 Project Goals

The long-term goal is to develop the system into a more robust and explainable resume intelligence platform that can help candidates:

* Understand their compatibility with a job description
* Identify missing skills
* Improve resume-job alignment
* Receive actionable feedback
* Better understand how different resume features influence matching

---

## 📚 Key Learning Areas

This project brings together:

* Natural Language Processing
* Sentence embeddings
* Semantic similarity
* Artificial Neural Networks
* Feature engineering
* Resume/document parsing
* Information extraction
* LLM integration
* Explainable AI concepts
* Streamlit application development
* Machine-learning experimentation

## ⭐ Future Direction

The project is being developed incrementally, with emphasis on moving from a functional prototype toward a more rigorously evaluated and explainable resume–job matching system.
