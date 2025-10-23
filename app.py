"""
===============================================================================
TalentIQ: THE AI-POWERED RECRUITMENT PLATFORM
===============================================================================

A sophisticated AI-powered recruitment tool featuring:
- Custom NER model for entity extraction
- Generative AI profile enrichment (Gemini API)
- Semantic vector search (Sentence-BERT + FAISS)
- Hybrid skill-gap analysis
- ATS Status Tracking
- Ability to delete profiles
===============================================================================
"""

import streamlit as st
import spacy
from spacy.matcher import Matcher
import sqlite3
import json
import pandas as pd
from pathlib import Path
import fitz  # PyMuPDF
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os
import google.generativeai as genai


st.set_page_config(
    page_title="TalentIQ",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Database configuration
DB_NAME = "profiles.db"
MODEL_DIR = "./custom-ner-model"
STATUS_OPTIONS = ["New", "Screening", "Interviewing", "Hired", "Rejected"]


# Initialize session state for caching
if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = None
if 'vector_index' not in st.session_state:
    st.session_state.vector_index = None
if 'profile_ids' not in st.session_state:
    st.session_state.profile_ids = []

# (NEW) Initialize session state for Job Matcher page
if 'match_results' not in st.session_state:
    st.session_state.match_results = None
if 'jd_text' not in st.session_state:
    st.session_state.jd_text = ""


# ============================================================================
# DATABASE FUNCTIONS
# ============================================================================

def init_database():
    """Initialize SQLite database with profiles table"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT,
            phone TEXT,
            skills_json TEXT,
            entities_json TEXT,
            full_text TEXT,
            ai_summary TEXT,
            seniority TEXT,
            key_projects TEXT,
            embedding BLOB,
            status TEXT DEFAULT 'New'
        )
    """)
    
    conn.commit()
    conn.close()

def migrate_db():
    """
    Checks if the 'status' column exists in the profiles table
    and adds it if it doesn't.
    """
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        cursor.execute("PRAGMA table_info(profiles)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if 'status' not in columns:
            st.toast("Database schema out of date. Adding 'status' column...")
            cursor.execute("ALTER TABLE profiles ADD COLUMN status TEXT DEFAULT 'New'")
            conn.commit()
            st.toast("Database migration complete!")
            
    except sqlite3.Error as e:
        st.error(f"Database migration error: {e}")
    finally:
        if conn:
            conn.close()

def insert_profile(name, email, phone, skills, entities, full_text, ai_summary="", seniority="", key_projects="", embedding=None):
    """Insert a new profile into the database"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    embedding_bytes = embedding.tobytes() if embedding is not None else None
    
    cursor.execute("""
        INSERT INTO profiles (name, email, phone, skills_json, entities_json, full_text, 
                              ai_summary, seniority, key_projects, embedding)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (name, email, phone, json.dumps(skills), json.dumps(entities), full_text,
          ai_summary, seniority, key_projects, embedding_bytes))
    
    conn.commit()
    profile_id = cursor.lastrowid
    conn.close()
    return profile_id

def get_all_profiles():
    """Retrieve all profiles from the database"""
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT * FROM profiles", conn)
    conn.close()
    return df

def get_profile_embeddings():
    """Retrieve profile IDs and embeddings"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT id, embedding FROM profiles WHERE embedding IS NOT NULL")
    results = cursor.fetchall()
    conn.close()
    
    if not results:
        return [], None
    
    profile_ids = [r[0] for r in results]
    embeddings = [np.frombuffer(r[1], dtype=np.float32) for r in results]
    
    return profile_ids, np.array(embeddings)

def delete_profile(profile_id):
    """Deletes a profile from the database by ID."""
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM profiles WHERE id = ?", (profile_id,))
        conn.commit()
        st.toast(f"Deleted profile ID: {profile_id}")
    except sqlite3.Error as e:
        st.error(f"Error deleting profile: {e}")
    finally:
        if conn:
            conn.close()

def update_profile_status(profile_id, new_status):
    """Updates the status of a profile in the database."""
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("UPDATE profiles SET status = ? WHERE id = ?", (new_status, profile_id))
        conn.commit()
    except sqlite3.Error as e:
        st.error(f"Error updating status: {e}")
    finally:
        if conn:
            conn.close()

# ============================================================================
# TEXT EXTRACTION FUNCTIONS
# ============================================================================

def extract_text_from_pdf(file):
    """Extract text from PDF file using PyMuPDF"""
    try:
        pdf_bytes = file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text.strip()
    except Exception as e:
        st.error(f"Error extracting PDF: {e}")
        return ""

def extract_text_from_docx(file):
    """Extract text from DOCX file using python-docx"""
    try:
        doc = Document(file)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text.strip()
    except Exception as e:
        st.error(f"Error extracting DOCX: {e}")
        return ""

# ============================================================================
# NER & ENTITY EXTRACTION
# ============================================================================

@st.cache_resource
def load_custom_model():
    """Load the trained custom NER model"""
    if not Path(MODEL_DIR).exists():
        st.error(f"❌ Custom model not found at {MODEL_DIR}")
        st.error("Please run 'python train_model.py' first!")
        st.stop()
    return spacy.load(MODEL_DIR)

@st.cache_resource
def load_base_spacy_model():
    """Load base spaCy model for rule-based matching"""
    return spacy.load("en_core_web_sm")

def extract_contact_info(text, nlp_base):
    """Extract email and phone using rule-based matching"""
    doc = nlp_base(text)
    matcher = Matcher(nlp_base.vocab)
    
    email_pattern = [{"LIKE_EMAIL": True}]
    matcher.add("EMAIL", [email_pattern])
    
    phone_pattern = [
        {"TEXT": {"REGEX": r"(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"}}
    ]
    matcher.add("PHONE", [phone_pattern])
    
    matches = matcher(doc)
    
    email = None
    phone = None
    
    for match_id, start, end in matches:
        span = doc[start:end]
        if nlp_base.vocab.strings[match_id] == "EMAIL":
            email = span.text
        elif nlp_base.vocab.strings[match_id] == "PHONE":
            phone = span.text
    
    if not email:
        import re
        email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        if email_match:
            email = email_match.group()
    
    if not phone:
        import re
        phone_match = re.search(r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', text)
        if phone_match:
            phone = phone_match.group()
    
    return email, phone

def extract_entities(text, nlp_custom):
    """Extract custom entities using trained NER model"""
    doc = nlp_custom(text)
    
    entities = {
        "SKILL": [],
        "DEGREE": [],
        "JOB_TITLE": []
    }
    
    for ent in doc.ents:
        if ent.label_ in entities:
            entities[ent.label_].append(ent.text)
    
    for key in entities:
        entities[key] = list(set(entities[key]))
    
    return entities

# ============================================================================
# GENERATIVE AI ENRICHMENT (GEMINI)
# ============================================================================

def enrich_profile_with_ai(resume_text):
    """Use Gemini API to generate profile insights"""
    
    api_key = st.secrets.get("GEMINI_API_KEY")
    
    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY")
        
    if not api_key:
        return {
            "summary": "AI enrichment unavailable (API key not configured)",
            "seniority": "Unknown",
            "key_projects": []
        }
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash-preview-09-2025')
        
        prompt = f"""You are an expert technical recruiter. Analyze this resume and provide a JSON response with:
1. "summary": A concise 2-3 sentence professional summary
2. "seniority": One of ["Entry-level", "Junior", "Mid-level", "Senior", "Lead", "Executive"]
3. "key_projects": List of 1-3 most impressive projects or achievements (brief descriptions)

Resume:
{resume_text[:3000]}

Return ONLY valid JSON with no additional text."""
        
        response = model.generate_content(prompt)
        
        response_text = response.text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        
        result = json.loads(response_text.strip())
        return result
        
    except Exception as e:
        st.warning(f"AI enrichment failed: {e}")
        return {
            "summary": "AI analysis unavailable",
            "seniority": "Unknown",
            "key_projects": []
        }

# ============================================================================
# SEMANTIC SEARCH & VECTOR DATABASE
# ============================================================================

@st.cache_resource
def load_embedding_model():
    """Load Sentence-BERT model for embeddings"""
    return SentenceTransformer('all-MiniLM-L6-v2')

def create_vector_index(embeddings):
    """Create FAISS index from embeddings"""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    return index

def semantic_search(query_text, top_k=10):
    """Perform semantic search using vector similarity"""
    
    if st.session_state.embedding_model is None:
        st.session_state.embedding_model = load_embedding_model()
    
    profile_ids, profile_embeddings = get_profile_embeddings()
    
    if len(profile_ids) == 0:
        return pd.DataFrame()
    
    if st.session_state.vector_index is None or st.session_state.profile_ids != profile_ids:
        st.session_state.vector_index = create_vector_index(profile_embeddings)
        st.session_state.profile_ids = profile_ids
    
    query_embedding = st.session_state.embedding_model.encode([query_text])
    
    distances, indices = st.session_state.vector_index.search(query_embedding.astype('float32'), min(top_k, len(profile_ids)))
    
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        if idx < len(profile_ids):
            similarity = max(0, 100 * (1 - dist / 10))  # Normalized
            results.append({
                'profile_id': profile_ids[idx],
                'semantic_score': similarity
            })
    
    return pd.DataFrame(results)

# ============================================================================
# HYBRID MATCHING
# ============================================================================

def hybrid_job_matching(job_description, nlp_custom):
    """Perform hybrid matching: semantic search + skill gap analysis"""
    
    semantic_results = semantic_search(job_description, top_k=50)
    
    if semantic_results.empty:
        return pd.DataFrame()
    
    job_entities = extract_entities(job_description, nlp_custom)
    required_skills = set([s.lower() for s in job_entities["SKILL"]])
    
    conn = sqlite3.connect(DB_NAME)
    
    results = []
    for _, row in semantic_results.iterrows():
        profile_id = int(row['profile_id'])
        
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, name, email, phone, skills_json, ai_summary, seniority, status 
            FROM profiles WHERE id = ?
        """, (profile_id,))
        
        profile_data = cursor.fetchone()
        
        if profile_data:
            pid, name, email, phone, skills_json, ai_summary, seniority, status = profile_data
            candidate_skills = set([s.lower() for s in json.loads(skills_json)])
            
            matching_skills = list(required_skills & candidate_skills)
            missing_skills = list(required_skills - candidate_skills)
            
            results.append({
                "id": pid,
                "Candidate Name": name or "Unknown",
                "Semantic Match Score (%)": round(row['semantic_score'], 1),
                "Seniority": seniority or "N/A",
                "Matching Skills": matching_skills,
                "Missing Skills": missing_skills,
                "Skill Match %": round(100 * len(matching_skills) / max(len(required_skills), 1), 1) if required_skills else 0,
                "Email": email or "N/A",
                "Phone": phone or "N/A",
                "AI Summary": ai_summary or "N/A",
                "status": status or "New"
            })
    
    conn.close()
    
    df = pd.DataFrame(results)
    if not df.empty:
        df['Combined Score'] = (df['Semantic Match Score (%)'] * 0.6 + df['Skill Match %'] * 0.4)
        df = df.sort_values('Combined Score', ascending=False)
    
    return df

# ============================================================================
# STREAMLIT UI
# ============================================================================

def main():
    
    # (BRANDING) 2. This is the second change for your main page title.
    # It goes right at the start of the 'main' function.
    st.title("TalentIQ: The AI-Powered Talent Hub 🧠")
    st.markdown("---")
    
    # Initialize and migrate the database
    init_database()
    migrate_db() 
    
    st.sidebar.title("🎯 Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["📊 Profile Database Manager", "🔍 Job Matcher & Skill-Gap Analysis"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🚀 Features")
    st.sidebar.markdown("""
    - **Custom NER Model**
    - **AI-Powered Insights** (Gemini)
    - **Semantic Search** (Sentence-BERT)
    - **Hybrid Ranking**
    - **Skill-Gap Analysis**
    - **ATS Status Tracking**
    - **Delete Profiles**
    """)
    
    # ========================================================================
    # PAGE 1: PROFILE DATABASE MANAGER
    # ========================================================================
    
    if page == "📊 Profile Database Manager":
        st.header("📊 Profile Database Manager") # Changed from st.title to st.header
        st.markdown("Upload resumes to parse and store candidate profiles with AI-powered enrichment.")
        
        # Wrap uploader and button in a container for better spacing
        with st.container(border=True):
            uploaded_files = st.file_uploader(
                "Upload Resume Files (PDF or DOCX)",
                type=["pdf", "docx"],
                accept_multiple_files=True
            )
            
            col1, col2 = st.columns([1, 4])
            with col1:
                parse_button = st.button("🚀 Parse and Save", type="primary", use_container_width=True)
            
            if parse_button and uploaded_files:
                nlp_custom = load_custom_model()
                nlp_base = load_base_spacy_model()
                
                if st.session_state.embedding_model is None:
                    with st.spinner("Loading embedding model..."):
                        st.session_state.embedding_model = load_embedding_model()
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, file in enumerate(uploaded_files):
                    status_text.text(f"Processing {file.name}...")
                    
                    if file.name.endswith(".pdf"):
                        text = extract_text_from_pdf(file)
                    else:
                        text = extract_text_from_docx(file)
                    
                    if not text:
                        st.warning(f"⚠️ Could not extract text from {file.name}")
                        continue
                    
                    entities = extract_entities(text, nlp_custom)
                    email, phone = extract_contact_info(text, nlp_base)
                    
                    name = text.split("\n")[0][:50] if text else "Unknown"
                    if email and (name == "Unknown" or len(name) > 30):
                        name = email.split("@")[0]
                    
                    ai_insights = enrich_profile_with_ai(text)
                    embedding = st.session_state.embedding_model.encode(text)
                    
                    profile_id = insert_profile(
                        name=name,
                        email=email,
                        phone=phone,
                        skills=entities["SKILL"],
                        entities=entities,
                        full_text=text,
                        ai_summary=ai_insights.get("summary", ""),
                        seniority=ai_insights.get("seniority", ""),
                        key_projects=json.dumps(ai_insights.get("key_projects", [])),
                        embedding=embedding
                    )
                    
                    st.session_state.vector_index = None
                    st.session_state.profile_ids = []
                    
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                
                status_text.text("✅ All files processed successfully!")
                st.success(f"✅ Successfully parsed and saved {len(uploaded_files)} resumes!")
                st.balloons()
                
                st.session_state.match_results = None
                st.session_state.jd_text = ""

        
        st.subheader("📁 Current Database")
        
        df = get_all_profiles()
        
        if df.empty:
            st.info("No profiles in database yet. Upload some resumes to get started!")
        else:
            st.metric("Total Profiles", len(df))
            
            with st.container(border=True):
                # Header for the profile list
                cols = st.columns([1, 3, 3, 2, 2, 1])
                cols[0].write("**ID**")
                cols[1].write("**Name**")
                cols[2].write("**Email**")
                cols[3].write("**Seniority**")
                cols[4].write("**ATS Status**")
                cols[5].write("**Actions**")
                st.markdown("---")

                # List each profile
                for _, row in df.iterrows():
                    cols = st.columns([1, 3, 3, 2, 2, 1])
                    with cols[0]:
                        st.write(row['id'])
                    with cols[1]:
                        st.write(row['name'])
                    with cols[2]:
                        st.write(row['email'])
                    with cols[3]:
                        st.write(row['seniority'])
                    with cols[4]:
                        st.write(row.get('status', 'New')) 
                    with cols[5]:
                        if st.button("Delete", key=f"del_{row['id']}", type="secondary"):
                            delete_profile(row['id'])
                            st.session_state.vector_index = None
                            st.session_state.profile_ids = []
                            st.session_state.match_results = None
                            st.session_state.jd_text = ""
                            st.rerun() 
            
    # ========================================================================
    # PAGE 2: JOB MATCHER
    # ========================================================================
    
    elif page == "🔍 Job Matcher & Skill-Gap Analysis":
        st.header("🔍 Interactive Job-Matching Dashboard") # Changed from st.title to st.header
        st.markdown("Paste a job description to find the best matching candidates using AI-powered semantic search and skill-gap analysis.")
        
        df = get_all_profiles()
        if df.empty:
            st.warning("⚠️ No profiles in database. Please add some resumes first!")
            return
        
        st.info(f"📊 Searching across {len(df)} candidate profiles")
        
        with st.container(border=True):
            st.text_area(
                "Job Description",
                height=250,
                placeholder="Paste the full job description here...",
                key="jd_text" 
            )
            
            col1, col2, col3 = st.columns([1, 1, 3])
            with col1:
                match_button = st.button("🎯 Find Best Matches", type="primary", use_container_width=True)
            with col2:
                top_k = st.selectbox("Top Candidates", [10, 20, 30, 50], index=0)
        
        if match_button:
            if st.session_state.jd_text.strip():
                with st.spinner("🤖 Running AI-powered matching algorithm..."):
                    nlp_custom = load_custom_model()
                    st.session_state.match_results = hybrid_job_matching(st.session_state.jd_text, nlp_custom)
            else:
                st.warning("⚠️ Please enter a job description first!")
                st.session_state.match_results = None 

        if st.session_state.match_results is not None:
            
            results_df = st.session_state.match_results.head(top_k) 
            
            if results_df.empty:
                st.error("No matching candidates found.")
            else:
                st.success(f"✅ Found {len(results_df)} matching candidates (showing top {top_k})!")
                
                with st.container(border=True):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        avg_semantic = results_df['Semantic Match Score (%)'].mean()
                        st.metric("Avg Semantic Match", f"{avg_semantic:.1f}%")
                    with col2:
                        avg_skill = results_df['Skill Match %'].mean()
                        st.metric("Avg Skill Match", f"{avg_skill:.1f}%")
                    with col3:
                        top_score = results_df['Combined Score'].max()
                        st.metric("Top Combined Score", f"{top_score:.1f}%")
                    with col4:
                        perfect_matches = len(results_df[results_df['Skill Match %'] == 100])
                        st.metric("Perfect Skill Matches", perfect_matches)
                
                st.subheader("🏆 Ranked Candidates")
                
                display_df = results_df.copy()
                display_df['Matching Skills'] = display_df['Matching Skills'].apply(lambda x: ', '.join(x) if x else 'None')
                display_df['Missing Skills'] = display_df['Missing Skills'].apply(lambda x: ', '.join(x) if x else 'None')
                
                def color_score(val):
                    if val >= 80: return 'background-color: #d4edda'
                    elif val >= 60: return 'background-color: #fff3cd'
                    else: return 'background-color: #f8d7da'
                
                styled_df = display_df.style.applymap(
                    color_score,
                    subset=['Semantic Match Score (%)', 'Skill Match %', 'Combined Score']
                ).format({
                    'Semantic Match Score (%)': '{:.1f}',
                    'Skill Match %': '{:.1f}',
                    'Combined Score': '{:.1f}'
                })
                
                st.dataframe(styled_df, use_container_width=True, height=600)
                
                st.subheader("📋 Detailed Candidate Profiles")
                
                for idx, row in results_df.head(5).iterrows():
                    with st.expander(f"🌟 {row['Candidate Name']} - Combined Score: {row['Combined Score']:.1f}%"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"**📧 Email:** {row['Email']}")
                            st.markdown(f"**📱 Phone:** {row['Phone']}")
                            st.markdown(f"**💼 Seniority:** {row['Seniority']}")
                            st.markdown(f"**🎯 Semantic Match:** {row['Semantic Match Score (%)']:.1f}%")
                            st.markdown(f"**⚙️ Skill Match:** {row['Skill Match %']:.1f}%")
                        
                        with col2:
                            st.markdown("**✅ Matching Skills:**")
                            if row['Matching Skills']:
                                for skill in row['Matching Skills']:
                                    st.markdown(f"- {skill}")
                            else:
                                st.markdown("*None*")
                            
                            st.markdown("**⚠️ Missing Skills:**")
                            if row['Missing Skills']:
                                for skill in row['Missing Skills']:
                                    st.markdown(f"- {skill}")
                            else:
                                st.markdown("*None - Perfect match!*")
                        
                        st.markdown("**🤖 AI-Generated Summary:**")
                        st.info(row['AI Summary'])

                        st.markdown("---")
                        
                        current_status = row['status']
                        try:
                            current_index = STATUS_OPTIONS.index(current_status)
                        except ValueError:
                            current_index = 0
                        
                        new_status = st.selectbox(
                            "Update Candidate Status:",
                            STATUS_OPTIONS,
                            index=current_index,
                            key=f"status_{row['id']}"
                        )
                        
                        if new_status != current_status:
                            update_profile_status(row['id'], new_status)
                            st.session_state.match_results.loc[
                                st.session_state.match_results['id'] == row['id'], 'status'
                            ] = new_status
                            st.rerun() 
                
                st.markdown("---")
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name="candidate_matches.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    main()

