# TalentIQ: The AI-Powered Talent Hub 🧠

**TalentIQ** helps recruiters find top talent faster using AI. This Streamlit application parses resumes, performs **semantic job matching** with vector search (FAISS & Sentence-BERT), enriches profiles using the **Gemini API**, and includes a mini **Applicant Tracking System (ATS)**.

---
🔗 **Live Demo:** [https://gauravvbisht18-talentiq-app-app-vmnpvo.streamlit.app/](https://gauravvbisht18-talentiq-app-app-vmnpvo.streamlit.app/)

## 🚀 Key Features

### 📄 Smart Resume Parsing
- Extracts structured data from resumes (PDF/DOCX).  
- Uses NLP models to identify skills, education, experience, and contact details.

### 💡 Custom NLP Models
- Fine-tuned transformer models for semantic entity extraction and classification.  
- Detects candidate expertise, seniority, and intent automatically.

### 🔍 Semantic Job Matching
- Utilizes **vector search** (FAISS or Pinecone) for meaning-based job matching.  
- Matches candidates beyond keywords using deep semantic similarity.

### 🤖 Generative AI Enrichment
- Uses LLMs to auto-generate professional summaries of candidates.  
- Suggests top-fit candidates for open roles with explanation highlights.

### ⚙️ Mini ATS (Applicant Tracking System)
- Manage, shortlist, and track candidates easily.  
- Includes tagging, filtering, and search functionalities for recruiters.

### 🧠 AI-Powered Job Recommendations
- Suggests best job opportunities based on skills and experience.  
- Personalized recommendations for both candidates and recruiters.

---

## 🧩 Architecture Overview

1. User uploads resume in PDF/DOCX format.  
2. Backend parses and extracts resume data using NLP pipeline.  
3. Extracted data is stored in a structured database.  
4. Job descriptions are vectorized and compared using semantic similarity.  
5. AI models recommend top candidates and generate professional summaries.  
6. Recruiters view insights through an intuitive frontend dashboard.

---

## 🧮 Machine Learning Pipeline

1. **Resume Parsing**
   - Extracts raw text and entities using spaCy and regex.  
   - Uses fine-tuned BERT model for accurate entity tagging.

2. **Feature Extraction**
   - Converts resumes and job descriptions into embeddings.

3. **Vector Search**
   - Performs cosine similarity search via FAISS/Pinecone index.

4. **Ranking & Scoring**
   - Ranks candidates by relevance, skills, and experience context.

5. **Profile Enrichment**
   - Generates summaries using LLMs to enhance recruiter visibility.

---

## 📊 Example Workflow

1. Recruiter uploads multiple resumes and a job description.  
2. System parses and indexes all resumes automatically.  
3. Semantic similarity identifies best-fit candidates.  
4. Recruiter receives:  
   - Ranked candidate list  
   - Auto-generated AI summaries  
   - Skill-based recommendations

---

## 🧠 AI Models Used

| Model | Purpose |
|--------|----------|
| **BERT / RoBERTa** | Embedding generation for resumes and job descriptions |
| **spaCy NER Model** | Entity extraction (skills, organizations, roles) |
| **GPT-based LLM** | Summarization and recommendation generation |
| **TF-IDF + Logistic Regression** | Initial matching baseline for traditional scoring |

---

## 🔒 Security & Data Privacy

- Implements **JWT Authentication** for secure access.  
- All sensitive data (resumes, profiles) encrypted at rest and in transit.  
- Anonymized resume data used only for model improvement.

---

## 💼 Future Enhancements

- 🌍 Multi-language parsing and matching  
- 🧩 Chrome extension for LinkedIn profile imports  
- 📈 Recruiter analytics and dashboard insights  
- 🔔 AI job alerts for candidates  
- 💬 Chat-based AI recruiter assistant  

---

## 👨‍💻 Developed By

**Gaurav Bisht**  
📍 Nainital, Uttarakhand  
💡 Passionate about building AI-driven tools that empower people and transform recruitment.

---

> *“TalentIQ isn’t just an app — it’s a vision to bridge the gap between recruiters and the right talent using AI.”*
