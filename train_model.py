"""
Custom spaCy NER Model Training Script
=======================================
This script trains a custom Named Entity Recognition model to extract:
- SKILL: Technical skills (e.g., Python, React, Machine Learning)
- DEGREE: Educational qualifications (e.g., B.Tech, Bachelor of Science)
- JOB_TITLE: Professional roles (e.g., Software Engineer, Data Analyst)

Run this script ONCE before launching the Streamlit app to generate the custom model.
"""

import spacy
from spacy.training import Example
import random
from pathlib import Path

# Training data in spaCy format
# Format: (text, {"entities": [(start, end, label), ...]})
TRAIN_DATA = [
    ("I am proficient in Python and JavaScript for web development.", {
        "entities": [(22, 28, "SKILL"), (33, 43, "SKILL")]
    }),
    ("My expertise includes Machine Learning and Data Science using TensorFlow.", {
        "entities": [(22, 40, "SKILL"), (45, 57, "SKILL"), (64, 74, "SKILL")]
    }),
    ("I have a B.Tech in Computer Science and a Master of Science in AI.", {
        "entities": [(9, 15, "DEGREE"), (50, 67, "DEGREE")]
    }),
    ("Worked as a Software Engineer at Google for 3 years.", {
        "entities": [(12, 29, "JOB_TITLE")]
    }),
    ("My role as Data Analyst involved analyzing customer behavior.", {
        "entities": [(11, 23, "JOB_TITLE")]
    }),
    ("Skills: React, Node.js, MongoDB, Docker, and Kubernetes for cloud deployment.", {
        "entities": [(8, 13, "SKILL"), (15, 22, "SKILL"), (24, 31, "SKILL"), 
                     (33, 39, "SKILL"), (45, 55, "SKILL")]
    }),
    ("I hold a Bachelor of Engineering degree and completed certification in AWS.", {
        "entities": [(9, 35, "DEGREE"), (72, 75, "SKILL")]
    }),
    ("Previously worked as Senior Data Scientist building ML models.", {
        "entities": [(21, 42, "JOB_TITLE"), (52, 54, "SKILL")]
    }),
    ("Technical skills include Java, C++, SQL, Git, and Agile methodologies.", {
        "entities": [(25, 29, "SKILL"), (31, 34, "SKILL"), (36, 39, "SKILL"), 
                     (41, 44, "SKILL"), (50, 55, "SKILL")]
    }),
    ("Experience as Full Stack Developer with expertise in Django and Flask.", {
        "entities": [(14, 35, "JOB_TITLE"), (58, 64, "SKILL"), (69, 74, "SKILL")]
    }),
    ("Certified in Deep Learning and Natural Language Processing techniques.", {
        "entities": [(13, 27, "SKILL"), (32, 59, "SKILL")]
    }),
    ("I completed my PhD in Artificial Intelligence from MIT.", {
        "entities": [(15, 18, "DEGREE")]
    }),
    ("Working as Machine Learning Engineer specializing in Computer Vision.", {
        "entities": [(11, 37, "JOB_TITLE"), (53, 68, "SKILL")]
    }),
    ("Core competencies: DevOps, CI/CD, Jenkins, Terraform, and Ansible.", {
        "entities": [(19, 25, "SKILL"), (27, 32, "SKILL"), (34, 41, "SKILL"), 
                     (43, 52, "SKILL"), (58, 65, "SKILL")]
    }),
    ("I am a Product Manager with 5 years of experience in SaaS.", {
        "entities": [(7, 22, "JOB_TITLE")]
    }),
    ("Educational background: Bachelor of Computer Applications and MBA.", {
        "entities": [(24, 57, "DEGREE"), (62, 65, "DEGREE")]
    }),
    ("Skilled in Ruby on Rails, GraphQL, REST APIs, and microservices architecture.", {
        "entities": [(11, 25, "SKILL"), (27, 34, "SKILL"), (36, 45, "SKILL"), 
                     (51, 64, "SKILL")]
    }),
    ("Position: Frontend Developer proficient in HTML, CSS, TypeScript, and Vue.js.", {
        "entities": [(10, 28, "JOB_TITLE"), (43, 47, "SKILL"), (49, 52, "SKILL"), 
                     (54, 64, "SKILL"), (70, 76, "SKILL")]
    }),
    ("I have expertise in Blockchain, Solidity, Ethereum, and Smart Contracts.", {
        "entities": [(20, 30, "SKILL"), (32, 40, "SKILL"), (42, 50, "SKILL"), 
                     (56, 71, "SKILL")]
    }),
    ("Degree: Master of Business Administration specializing in Finance.", {
        "entities": [(8, 42, "DEGREE")]
    }),
    ("Currently working as DevOps Engineer managing cloud infrastructure on Azure.", {
        "entities": [(21, 36, "JOB_TITLE"), (70, 75, "SKILL")]
    }),
    ("Proficient in Tableau, Power BI, Excel, and data visualization techniques.", {
        "entities": [(14, 21, "SKILL"), (23, 31, "SKILL"), (33, 38, "SKILL"), 
                     (44, 63, "SKILL")]
    }),
    ("Role: Business Analyst with strong analytical and problem-solving skills.", {
        "entities": [(6, 22, "JOB_TITLE")]
    }),
    ("I possess a Bachelor of Arts in Economics and Statistics.", {
        "entities": [(12, 29, "DEGREE")]
    }),
    ("Experienced QA Engineer skilled in Selenium, Cypress, automated testing.", {
        "entities": [(12, 23, "JOB_TITLE"), (35, 43, "SKILL"), (45, 52, "SKILL"), 
                     (54, 70, "SKILL")]
    }),
    ("Technical expertise: Scala, Apache Spark, Hadoop, big data processing.", {
        "entities": [(21, 26, "SKILL"), (28, 40, "SKILL"), (42, 48, "SKILL"), 
                     (50, 68, "SKILL")]
    }),
    ("Working as UI/UX Designer creating wireframes and prototypes in Figma.", {
        "entities": [(11, 25, "JOB_TITLE"), (64, 69, "SKILL")]
    }),
    ("Qualifications: Associate Degree in Network Administration.", {
        "entities": [(16, 32, "DEGREE")]
    }),
    ("I am a Cybersecurity Specialist expert in penetration testing and ethical hacking.", {
        "entities": [(7, 31, "JOB_TITLE"), (42, 61, "SKILL"), (66, 81, "SKILL")]
    }),
    ("Skilled in SAP, Oracle, ERP systems, supply chain management software.", {
        "entities": [(11, 14, "SKILL"), (16, 22, "SKILL"), (24, 35, "SKILL"), 
                     (37, 62, "SKILL")]
    })
]

def train_ner_model(train_data, n_iter=30, output_dir="./custom-ner-model"):
    """
    Train a custom spaCy NER model from scratch.
    
    Args:
        train_data: List of training examples
        n_iter: Number of training iterations
        output_dir: Directory to save the trained model
    """
    # Create a blank English model
    print("Creating blank English model...")
    nlp = spacy.blank("en")
    
    # Create the NER pipeline component
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")
    
    # Add labels to the NER component
    for _, annotations in train_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])
    
    # Disable other pipeline components during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    
    print(f"Training model for {n_iter} iterations...")
    print(f"Labels: {ner.labels}")
    
    # Training loop
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()
        
        for iteration in range(n_iter):
            random.shuffle(train_data)
            losses = {}
            
            # Batch training
            for text, annotations in train_data:
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                nlp.update([example], drop=0.5, losses=losses, sgd=optimizer)
            
            if (iteration + 1) % 5 == 0:
                print(f"Iteration {iteration + 1}/{n_iter} - Loss: {losses['ner']:.4f}")
    
    # Save the trained model
    output_path = Path(output_dir)
    if not output_path.exists():
        output_path.mkdir(parents=True)
    
    nlp.to_disk(output_path)
    print(f"\n✓ Model saved to {output_dir}")
    
    # Test the model
    print("\nTesting the trained model:")
    test_text = "I am a Senior Software Engineer with expertise in Python, Machine Learning, and AWS. I hold a Master of Science degree."
    doc = nlp(test_text)
    
    print(f"\nTest text: {test_text}")
    print("\nExtracted entities:")
    for ent in doc.ents:
        print(f"  - {ent.text:30} | {ent.label_}")
    
    return nlp

if __name__ == "__main__":
    print("=" * 60)
    print("Starting Custom NER Model Training")
    print("=" * 60)
    
    trained_model = train_ner_model(TRAIN_DATA, n_iter=30)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print("\nNext step: Run 'streamlit run app.py' to launch the application.")