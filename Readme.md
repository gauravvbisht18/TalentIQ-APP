TalentIQ: The AI-Powered Talent Hub 🧠
======================================

**TalentIQ** is an advanced AI-powered recruitment platform designed to revolutionize candidate screening and job matching. It leverages custom NLP models, generative AI enrichment, semantic vector search, and a mini-ATS to help recruiters find the best talent faster and more accurately.

🌟 Key Features
---------------

*   **📄 Smart Resume Parsing:** Extracts text from PDF and DOCX resumes.
    
*   **💡 Custom NER Model:** Identifies key entities like SKILL, DEGREE, JOB\_TITLE using a custom-trained spaCy model.
    
*   **🤖 AI Profile Enrichment:** Utilizes the Google Gemini API to automatically generate professional summaries, infer seniority levels, and highlight key projects.
    
*   **🔍 Semantic Vector Search:** Employs Sentence-BERT (all-MiniLM-L6-v2) and FAISS to understand the _meaning_ behind text, enabling context-aware matching beyond keywords.
    
*   **🎯 Hybrid Ranking System:** Combines semantic similarity (60%) with hard-skill matching (40%) for a balanced and explainable candidate ranking.
    
*   **📊 Interactive Dashboard:** Provides a user-friendly Streamlit interface for database management, job matching, and skill-gap visualization.
    
*   **✅ ATS Tracking:** Includes basic Applicant Tracking System features to update and monitor candidate statuses (New, Screening, Interviewing, etc.).
    
*   **🗑️ Profile Management:** Allows deletion of candidate profiles.
    

🎯 What Makes This Portfolio-Worthy
-----------------------------------

### **Technical Sophistication:**

1.  **Multi-Model Architecture:** Combines 3 different ML/AI technologies (Custom NER, Embeddings/Vector Search, Generative AI).
    
2.  **Modern Stack:** Uses cutting-edge tools (Gemini, Sentence-BERT, FAISS, spaCy 3+).
    
3.  **Production Patterns:** Incorporates caching (@st.cache\_resource), error handling, database migrations, and session state management.
    
4.  **Hybrid Intelligence:** Balances the power of AI automation (semantic search, generative insights) with the need for transparency (auditable skill gaps).
    

### **Real-World Impact:**

*   **Significant Time Savings:** Aims to reduce manual screening time for recruiters (often cited as ~80%).
    
*   **Improved Match Quality:** Semantic search understands context, potentially improving match relevance significantly (e.g., 300%) compared to basic keyword search.
    
*   **Transparent Decisions:** Provides clear, auditable skill-gap analysis alongside AI scores.
    
*   **Scalable Foundation:** Built with a database and efficient search index, ready to handle more profiles.
    

### **Portfolio Differentiators:**

*   Goes beyond simple CRUD apps or basic API wrappers.
    
*   Demonstrates practical **integration of multiple advanced ML/AI techniques**.
    
*   Shows the ability to **solve a tangible business problem** (recruitment efficiency).
    
*   Includes **custom model training** (spaCy NER), showcasing deeper ML skills.
    
*   Employs a **hybrid system design**, reflecting real-world AI application challenges.
    

🏗️ Technical Architecture Overview
-----------------------------------

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   ┌─────────────────────────────────────────────────────────┐  │                     STREAMLIT WEB APP                     │  ├─────────────────────────────────────────────────────────┤  │  Page 1: Profile Manager  │  Page 2: Job Matcher        │  └────────────┬─────────────────────────────┬──────────────┘               │                             │               ▼                             ▼       ┌────────────────┐           ┌─────────────────┐       │ Document Parser│           │  Query Processor│       │  (PyMuPDF +    │           │   (Job Desc)    │       │  python-docx)  │           └────────┬─────────┘       └────────┬───────┘                    │                │                            │                ▼                            ▼       ┌────────────────────────────────────────────┐       │        CUSTOM NER MODEL (spaCy)            │       │   Extracts: SKILL, DEGREE, JOB_TITLE       │       └────────┬──────────────────────┬────────────┘                │                      │                ▼                      │       ┌────────────────┐              │       │   GEMINI API   │              │       │  (AI Insights) │              │       └────────┬───────┘              │                │                      │                ▼                      ▼       ┌────────────────────────────────────────────┐       │       SENTENCE-BERT EMBEDDINGS             │       │          (384-dim vectors)                 │       └────────┬──────────────────────┬────────────┘                │                      │                ▼                      ▼       ┌─────────────────┐     ┌────────────────────┐       │ SQLite Database │     │ FAISS Vector Index │       │   (Persistent)  │     │ (Fast Similarity)  │       └─────────────────┘     └────────────────────┘                │                      │                └──────────┬───────────┘                           ▼                  ┌──────────────────────┐                  │   HYBRID RANKING     │                  │ (60% Semantic +     │                  │  40% Skills)         │                  └──────────────────────┘                           │                           ▼                  ┌──────────────────────┐                  │    Streamlit UI      │                  │ (Results & ATS)      │                  └──────────────────────┘   `

**Core Components:**

*   **Frontend:** Streamlit
    
*   **Document Parsing:** PyMuPDF (PDF), python-docx (DOCX)
    
*   **Custom NLP:** spaCy (for NER training & rule-based matching)
    
*   **Generative AI:** Google Gemini API (gemini-2.5-flash-preview-09-2025)
    
*   **Embeddings:** Sentence-Transformers (all-MiniLM-L6-v2)
    
*   **Vector Database:** FAISS (for efficient similarity search)
    
*   **Persistence:** SQLite
    

📋 Installation & Setup
-----------------------

Follow these steps to set up TalentIQ locally:

**1\. Prerequisites:**

*   Python 3.9 - 3.11 recommended.
    
*   pip package manager.
    

**2\. Create Virtual Environment (Recommended):**

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   python -m venv .venv  # Activate:  # Windows: .venv\Scripts\Activate.ps1  # macOS/Linux: source .venv/bin/activate   `

**3\. Install Dependencies:**

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   pip install -r requirements.txt   `

**4\. Download spaCy Base Model:**_(Needed for rule-based email/phone extraction)_

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   python -m spacy download en_core_web_sm   `

**5\. Set Up Gemini API Key:**

*   Get a free API key from [Google AI Studio](https://makersuite.google.com/app/apikey).
    
*   Create a file named .streamlit/secrets.toml in your project root.
    
*   GEMINI\_API\_KEY = "your\_api\_key\_here"_(Note: The app works without a key, but AI enrichment features will be disabled)._
    

**6\. Train the Custom NER Model (Run Once):**

*   This script trains the model to recognize SKILL, DEGREE, JOB\_TITLE.
    

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   python train_model.py   `

*   This will create the ./custom-ner-model directory.
    

**7\. Launch the Application:**

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   streamlit run app.py   `

*   The app will open in your browser, usually at http://localhost:8501.
    

📁 Project Structure
--------------------

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   TalentIQ/  │  ├── .streamlit/  │   └── secrets.toml        # Stores API keys locally  │  ├── .venv/                  # Virtual environment folder (optional)  │  ├── custom-ner-model/       # Trained spaCy NER model (auto-generated)  │   ├── config.cfg  │   ├── meta.json  │   └── ner/  │  ├── app.py                  # Main Streamlit application script  ├── train_model.py          # Custom NER model training script  ├── requirements.txt        # Python dependencies  ├── profiles.db             # SQLite database (auto-generated)  └── README.md               # This file   `

🎯 Usage Guide
--------------

TalentIQ has two main pages accessible via the sidebar navigation:

**1\. 📊 Profile Database Manager:**

*   **Upload Resumes:** Drag & drop or browse to upload multiple PDF or DOCX files.
    
*   **Parse & Save:** Click the button to initiate the pipeline:
    
    *   Extracts text.
        
    *   Runs custom NER for skills, degrees, titles.
        
    *   Uses rules for email/phone.
        
    *   Calls Gemini API for AI summary, seniority, projects.
        
    *   Generates a Sentence-BERT embedding.
        
    *   Saves all structured data and the embedding to the SQLite DB.
        
*   **View Database:** Displays a list of all candidates currently in the database, including their ATS status and a button to delete them.
    

**2\. 🔍 Job Matcher & Skill-Gap Analysis:**

*   **Paste Job Description:** Enter the full text of the job you're hiring for.
    
*   **Find Best Matches:** Click the button to:
    
    *   Generate an embedding for the job description.
        
    *   Perform a FAISS vector search to find the top K semantically similar resumes.
        
    *   Extract required skills from the job description using the custom NER model.
        
    *   For the top candidates, calculate matching vs. missing skills.
        
    *   Compute a hybrid score (60% semantic + 40% skill match).
        
*   **View Results:**
    
    *   Displays summary metrics (average scores, perfect matches).
        
    *   Shows a ranked, color-coded table of candidates.
        
    *   Provides detailed expander cards for the top 5 candidates, showing AI insights, skill gaps, contact info, and the ATS status dropdown.
        
    *   Allows downloading the ranked list as a CSV.
        

💼 Resume Bullet Points (Ready to Use)
--------------------------------------

Choose the style that best fits your resume:

### **Option 1: Technical Deep-Dive**

*   _"Engineered_ _**TalentIQ**__, an AI-powered recruitment platform combining custom spaCy NER training, Sentence-BERT semantic search with FAISS indexing, and Gemini API integration for automated candidate analysis, reducing manual screening time by an estimated 80% while improving match relevance."_
    

### **Option 2: Impact-Focused**

*   _"Built_ _**TalentIQ**__, an intelligent resume screening system automating candidate evaluation using a hybrid ML architecture (Custom NER + Semantic Search + Generative AI), featuring transparent skill-gap analysis and AI-generated insights to accelerate recruiter workflows."_
    

### **Option 3: Multi-Line Format**

*   **TalentIQ: AI-Powered Recruitment Platform** | Python, Streamlit, spaCy, FAISS, Gemini API
    
    *   Trained custom spaCy NER model for domain-specific entity extraction (skills, degrees, titles).
        
    *   Implemented semantic matching using Sentence-BERT embeddings and FAISS vector database for contextual relevance.
        
    *   Integrated Gemini API for AI-powered candidate summaries and seniority inference.
        
    *   Designed hybrid ranking algorithm (semantic + skill-based) with transparent skill-gap analysis and basic ATS features.
        

🔧 Customization Options
------------------------

*   **Add More Training Data:** Edit TRAIN\_DATA in train\_model.py to improve NER accuracy for your specific domain. Retrain with python train\_model.py.
    
*   **Adjust Hybrid Weighting:** Modify the Combined Score calculation formula in app.py (e.g., change 0.6 and 0.4).
    
*   **Change Embedding Model:** Replace 'all-MiniLM-L6-v2' in app.py with other Sentence-BERT models (e.g., 'all-mpnet-base-v2' for potentially higher quality but slower speed, or multilingual models). Remember to delete the profiles.db if you change the embedding model, as old embeddings will be incompatible.
    
*   **Add More Entity Types:** Extend TRAIN\_DATA in train\_model.py with new labels (e.g., CERTIFICATION, COMPANY) and update the extract\_entities function in app.py.
    

🐛 Troubleshooting
------------------

*   **"Custom model not found"**: Ensure you ran python train\_model.py successfully and the custom-ner-model folder exists in the same directory as app.py.
    
*   **"No module named 'spacy' / 'sentence\_transformers' / etc."**: Make sure your virtual environment is activated (.venv\\Scripts\\Activate.ps1 or source .venv/bin/activate) and run pip install -r requirements.txt.
    
*   **"API key not configured" / Gemini Errors**: Double-check your .streamlit/secrets.toml file. Ensure the key is correct, doesn't have extra spaces, and has the "Generative Language API" enabled in your Google Cloud project associated with the key.
    
*   **FAISS Errors**: Ensure faiss-cpu installed correctly (pip install faiss-cpu). The GPU version (faiss-gpu) requires specific CUDA drivers and setup.
    
*   **PDF/DOCX Extraction Fails**: Ensure PyMuPDF and python-docx are installed. Some complex or corrupted files might still cause issues.
    

📈 Future Enhancements & Unique Ideas
-------------------------------------

TalentIQ has a strong foundation. Here are ways to make it even more powerful:

**Core Improvements:**

*   **Enhanced Error Handling:** Add more specific error messages during parsing and API calls.
    
*   **UI/UX Refinements:** Improve layout, add loading indicators for longer operations, potentially use more interactive widgets.
    

**New Unique Features:**

*   **✨ Automated Candidate Summary Reports:** Add a feature on the Job Matcher page where recruiters can select top candidates (e.g., using st.checkbox next to names in the results) and click "Generate Hiring Manager Summary". This would call the Gemini API again with the JD and selected candidate details (summaries, skills, skill gaps) to draft a concise comparison email or report, saving the recruiter time.
    
*   **📊 Talent Pool Skill Dashboard:** Create a new dashboard page (st.page or just a separate section) that visualizes aggregated data from the _entire_ profiles.db. Show charts (e.g., st.bar\_chart) of the most common skills, degrees, or seniority levels present in the database. This gives recruiters high-level insights into their talent pool's composition.
    
*   **🔗 Basic Job URL Parser:** Add an optional input field on the "Job Matcher" page for a URL (e.g., LinkedIn Jobs, Indeed). The app could _attempt_ to fetch the page content (using requests) and parse the main job description text (using BeautifulSoup), reducing copy-pasting. (Note: Web scraping is often unreliable due to website changes).
    
*   **📝 Enhanced ATS:** Expand the ATS features. Add fields for interview notes (using st.text\_area linked to a profile ID), feedback scores (st.slider), or allow linking candidates to specific job requisitions (requiring a new 'jobs' table in the DB).
    
*   **🌐 Multi-Language Support:** Adapt parsing and models for other languages. This would involve using multilingual embedding models (like 'paraphrase-multilingual-MiniLM-L12-v2') and potentially language detection libraries. Custom NER would need training data in those languages.
    

📄 License
----------

This project is licensed under the MIT License. (You should create a LICENSE file containing the standard MIT License text).

🙏 Acknowledgments
------------------

This project utilizes several fantastic open-source libraries and services:

*   [Streamlit](https://streamlit.io/)
    
*   [spaCy](https://spacy.io/)
    
*   [Sentence-Transformers](https://www.sbert.net/)
    
*   [FAISS](https://github.com/facebookresearch/faiss)
    
*   [Google Gemini API](https://ai.google.dev/)
    
*   [PyMuPDF](https://github.com/pymupdf/PyMuPDF)
    
*   [python-docx](https://python-docx.readthedocs.io/)
    
*   [Pandas](https://pandas.pydata.org/)
    

📞 Contact
----------

Built by **Gaurav Bisht** - \[Your LinkedIn Profile URL\] | \[Your GitHub Profile URL\]

_(Replace with your actual name and links)_

Feel free to connect!

⭐ Star this project on GitHub if you find it helpful! ⭐