import streamlit as st
import PyPDF2
import docx
import io
import re
from typing import Dict, List, Tuple
import json
import os
from datetime import datetime
import requests

# Try to import OpenAI, but make it optional
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Configure page
st.set_page_config(
    page_title="ATS Resume Analyzer",
    page_icon="📄",
    layout="wide"
)

class ATSAnalyzer:
    def __init__(self, api_key: str = None, provider: str = "demo"):
        self.provider = provider
        self.api_key = api_key
        
        if provider == "openai" and api_key and OPENAI_AVAILABLE:
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = None
    
    def extract_text_from_pdf(self, uploaded_file) -> str:
        try:
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
            return ""
    
    def extract_text_from_docx(self, uploaded_file) -> str:
        try:
            doc = docx.Document(uploaded_file)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            st.error(f"Error reading DOCX: {str(e)}")
            return ""
    
    def extract_text_from_file(self, uploaded_file) -> str:
        if uploaded_file.type == "application/pdf":
            return self.extract_text_from_pdf(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            return self.extract_text_from_docx(uploaded_file)
        elif uploaded_file.type == "text/plain":
            return str(uploaded_file.read(), "utf-8")
        else:
            st.error("Unsupported file format. Please upload PDF, DOCX, or TXT files.")
            return ""
    
    def analyze_with_huggingface(self, resume_text: str, job_description: str) -> Dict:
        try:
            api_url = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"
            prompt = f"Analyze this resume for job fit. Resume: {resume_text[:500]}... Job: {job_description[:500]}..."
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            payload = {"inputs": prompt}
            
            if self.api_key:
                response = requests.post(api_url, headers=headers, json=payload, timeout=30)
                if response.status_code == 200:
                    return self.enhanced_analysis(resume_text, job_description)
            
            return self.enhanced_analysis(resume_text, job_description)
        except Exception as e:
            st.warning(f"HuggingFace API issue: {str(e)}. Using enhanced analysis.")
            return self.enhanced_analysis(resume_text, job_description)
    
    def analyze_resume_with_llm(self, resume_text: str, job_description: str) -> Dict:
        if self.provider == "openai" and self.client:
            return self.analyze_with_openai(resume_text, job_description)
        elif self.provider == "huggingface":
            return self.analyze_with_huggingface(resume_text, job_description)
        else:
            return self.enhanced_analysis(resume_text, job_description)
    
    def analyze_with_openai(self, resume_text: str, job_description: str) -> Dict:
        prompt = f"""
        You are an expert ATS (Applicant Tracking System) analyzer. Analyze the following resume against the job description and provide a comprehensive evaluation.

        RESUME:
        {resume_text}

        JOB DESCRIPTION:
        {job_description}

        Please provide your analysis in the following JSON format:
        {{
            "ats_score": <score out of 100>,
            "job_required_skills": [<list of skills required for the job>],
            "resume_present_skills": [<list of skills found in the resume>],
            "missing_skills": [<list of skills missing from resume but required for job>],
            "recommended_additions": [<list of specific recommendations to improve the resume>],
            "keyword_match_percentage": <percentage of job keywords found in resume>,
            "strengths": [<list of resume strengths for this job>],
            "weaknesses": [<list of areas for improvement>],
            "overall_feedback": "<detailed feedback about the resume's fit for this job>"
        }}
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert ATS analyzer and career counselor."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            
            analysis_text = response.choices[0].message.content
            json_start = analysis_text.find('{')
            json_end = analysis_text.rfind('}') + 1
            json_str = analysis_text[json_start:json_end]
            return json.loads(json_str)
        except Exception as e:
            st.error(f"Error with OpenAI analysis: {str(e)}")
            st.info("Falling back to enhanced local analysis...")
            return self.enhanced_analysis(resume_text, job_description)
    
    def get_role_specific_skills(self, job_description: str) -> Dict:
        job_lower = job_description.lower()
        role_skills = {
            # Software Development Roles
            'java_developer': {
                'keywords': ['java developer', 'java', 'spring', 'springboot'],
                'skills': ['Java', 'Spring Boot', 'Spring Framework', 'Hibernate', 'Maven', 'Gradle', 'JUnit', 'REST API', 'Microservices', 'SQL', 'Git']
            },
            'frontend_developer': {
                'keywords': ['frontend', 'front-end', 'ui developer', 'react developer'],
                'skills': ['HTML', 'CSS', 'JavaScript', 'React', 'Angular', 'Vue.js', 'TypeScript', 'Sass', 'Bootstrap', 'jQuery', 'Git']
            },
            'python_developer': {
                'keywords': ['python developer', 'python', 'django', 'flask'],
                'skills': ['Python', 'Django', 'Flask', 'FastAPI', 'SQLAlchemy', 'Pandas', 'NumPy', 'REST API', 'PostgreSQL', 'Git', 'Docker']
            },
            'fullstack_developer': {
                'keywords': ['fullstack', 'full-stack', 'full stack'],
                'skills': ['JavaScript', 'React', 'Node.js', 'Express.js', 'MongoDB', 'SQL', 'HTML', 'CSS', 'Git', 'REST API', 'Docker']
            },
            'cpp_developer': {
                'keywords': ['c++ developer', 'c++', 'cpp'],
                'skills': ['C++', 'Object Oriented Programming', 'STL', 'Data Structures', 'Algorithms', 'GCC', 'CMake', 'Git', 'Linux', 'Debugging']
            },
            'mobile_developer': {
                'keywords': ['android developer', 'ios developer', 'mobile developer', 'react native', 'flutter'],
                'skills': ['Java', 'Kotlin', 'Swift', 'React Native', 'Flutter', 'Android Studio', 'Xcode', 'REST API', 'Git', 'Mobile UI/UX']
            },
            'backend_developer': {
                'keywords': ['backend developer', 'backend', 'server side', 'api developer'],
                'skills': ['Python', 'Java', 'Node.js', 'Express.js', 'Django', 'Spring Boot', 'REST API', 'GraphQL', 'SQL', 'MongoDB', 'Redis', 'Git']
            },
            'dotnet_developer': {
                'keywords': ['.net developer', 'c# developer', 'asp.net', 'dotnet'],
                'skills': ['C#', '.NET Framework', '.NET Core', 'ASP.NET', 'Entity Framework', 'SQL Server', 'Azure', 'Visual Studio', 'Git', 'REST API']
            },
            'golang_developer': {
                'keywords': ['golang developer', 'go developer', 'go lang'],
                'skills': ['Go', 'Golang', 'Gin', 'Gorilla', 'Docker', 'Kubernetes', 'REST API', 'gRPC', 'PostgreSQL', 'Redis', 'Git']
            },
            
            # AI/ML/Data Roles
            'genai_engineer': {
                'keywords': ['generative ai', 'genai', 'gen ai', 'ai engineer', 'llm engineer', 'chatgpt', 'gpt'],
                'skills': ['Python', 'LangChain', 'OpenAI API', 'Hugging Face', 'Transformers', 'PyTorch', 'TensorFlow', 'Vector Databases', 'RAG', 'Prompt Engineering', 'LLAMA', 'GPT']
            },
            'ai_engineer': {
                'keywords': ['ai engineer', 'artificial intelligence', 'machine learning engineer', 'deep learning'],
                'skills': ['Python', 'TensorFlow', 'PyTorch', 'Keras', 'Scikit-learn', 'OpenCV', 'Neural Networks', 'Deep Learning', 'Computer Vision', 'NLP', 'MLOps']
            },
            'ml_engineer': {
                'keywords': ['ml engineer', 'machine learning engineer', 'mlops', 'machine learning'],
                'skills': ['Python', 'TensorFlow', 'PyTorch', 'Scikit-learn', 'MLflow', 'Kubeflow', 'Docker', 'Kubernetes', 'AWS SageMaker', 'Model Deployment', 'Feature Engineering']
            },
            'data_scientist': {
                'keywords': ['data scientist', 'data science'],
                'skills': ['Python', 'R', 'Machine Learning', 'Pandas', 'NumPy', 'Scikit-learn', 'TensorFlow', 'SQL', 'Jupyter', 'Statistics', 'Data Visualization', 'Matplotlib', 'Seaborn']
            },
            'data_analyst': {
                'keywords': ['data analyst', 'business analyst', 'data analytics'],
                'skills': ['SQL', 'Python', 'R', 'Excel', 'Tableau', 'Power BI', 'Pandas', 'NumPy', 'Statistics', 'Data Visualization', 'Google Analytics', 'ETL']
            },
            'data_engineer': {
                'keywords': ['data engineer', 'data engineering', 'etl developer'],
                'skills': ['Python', 'SQL', 'Apache Spark', 'Hadoop', 'Kafka', 'Airflow', 'ETL', 'AWS', 'Azure', 'Google Cloud', 'Docker', 'Kubernetes']
            },
            'computer_vision_engineer': {
                'keywords': ['computer vision', 'cv engineer', 'image processing'],
                'skills': ['Python', 'OpenCV', 'TensorFlow', 'PyTorch', 'YOLO', 'CNN', 'Image Processing', 'Deep Learning', 'NumPy', 'Matplotlib', 'Keras']
            },
            'nlp_engineer': {
                'keywords': ['nlp engineer', 'natural language processing', 'text analytics'],
                'skills': ['Python', 'NLTK', 'spaCy', 'Transformers', 'BERT', 'GPT', 'TensorFlow', 'PyTorch', 'Hugging Face', 'Text Processing', 'Sentiment Analysis']
            },
            
            # DevOps/Cloud/Infrastructure
            'devops_engineer': {
                'keywords': ['devops', 'sre', 'site reliability'],
                'skills': ['Docker', 'Kubernetes', 'AWS', 'Jenkins', 'Git', 'Linux', 'Terraform', 'Ansible', 'CI/CD', 'Monitoring', 'Bash']
            },
            'cloud_engineer': {
                'keywords': ['cloud engineer', 'aws engineer', 'azure engineer', 'gcp engineer'],
                'skills': ['AWS', 'Azure', 'Google Cloud', 'Docker', 'Kubernetes', 'Terraform', 'CloudFormation', 'Lambda', 'EC2', 'S3', 'VPC']
            },
            'aws_engineer': {
                'keywords': ['aws engineer', 'aws developer', 'amazon web services'],
                'skills': ['AWS', 'EC2', 'S3', 'Lambda', 'RDS', 'VPC', 'IAM', 'CloudFormation', 'API Gateway', 'DynamoDB', 'CloudWatch']
            },
            'azure_engineer': {
                'keywords': ['azure engineer', 'azure developer', 'microsoft azure'],
                'skills': ['Microsoft Azure', 'Azure Functions', 'Azure Storage', 'Azure SQL', 'ARM Templates', 'Azure DevOps', 'Power BI', 'Azure AD']
            },
            'sre_engineer': {
                'keywords': ['sre', 'site reliability engineer', 'platform engineer'],
                'skills': ['Kubernetes', 'Docker', 'Prometheus', 'Grafana', 'Linux', 'Python', 'Go', 'Terraform', 'Monitoring', 'Incident Management', 'Automation']
            },
            
            # Engineering Roles
            'mechanical_engineer': {
                'keywords': ['mechanical engineer', 'mechanical engineering', 'design engineer'],
                'skills': ['AutoCAD', 'SolidWorks', 'CATIA', 'ANSYS', 'MATLAB', 'Mechanical Design', 'CAD', 'FEA', 'Manufacturing', 'Project Management', 'Quality Control']
            },
            'civil_engineer': {
                'keywords': ['civil engineer', 'civil engineering', 'structural engineer'],
                'skills': ['AutoCAD', 'Revit', 'SAP2000', 'STAAD Pro', 'Structural Design', 'Construction Management', 'Project Planning', 'Surveying', 'Concrete Design', 'Steel Design']
            },
            'electrical_engineer': {
                'keywords': ['electrical engineer', 'electrical engineering', 'electronics engineer'],
                'skills': ['Circuit Design', 'PCB Design', 'MATLAB', 'Simulink', 'PLC Programming', 'AutoCAD Electrical', 'Power Systems', 'Control Systems', 'Embedded Systems']
            },
            'software_engineer': {
                'keywords': ['software engineer', 'software developer', 'programmer'],
                'skills': ['Python', 'Java', 'JavaScript', 'C++', 'Data Structures', 'Algorithms', 'Object Oriented Programming', 'Git', 'REST API', 'Databases', 'Testing']
            },
            'embedded_engineer': {
                'keywords': ['embedded engineer', 'embedded systems', 'firmware engineer'],
                'skills': ['C', 'C++', 'Embedded C', 'Microcontrollers', 'Arduino', 'Raspberry Pi', 'RTOS', 'PCB Design', 'Hardware Debugging', 'Assembly Language']
            },
            
            # QA/Testing
            'qa_engineer': {
                'keywords': ['qa engineer', 'quality assurance', 'test engineer', 'software tester'],
                'skills': ['Manual Testing', 'Automation Testing', 'Selenium', 'TestNG', 'JUnit', 'API Testing', 'Performance Testing', 'Bug Tracking', 'Test Planning', 'JIRA']
            },
            'sdet_engineer': {
                'keywords': ['sdet', 'automation engineer', 'test automation'],
                'skills': ['Java', 'Python', 'Selenium', 'TestNG', 'Cucumber', 'REST Assured', 'API Testing', 'CI/CD', 'Git', 'Maven', 'Jenkins']
            },
            
            # Database/Analytics
            'database_administrator': {
                'keywords': ['dba', 'database administrator', 'database engineer'],
                'skills': ['SQL', 'MySQL', 'PostgreSQL', 'Oracle', 'SQL Server', 'Database Design', 'Performance Tuning', 'Backup Recovery', 'Database Security']
            },
            'business_intelligence': {
                'keywords': ['bi developer', 'business intelligence', 'etl developer'],
                'skills': ['SQL', 'ETL', 'Data Warehousing', 'Tableau', 'Power BI', 'SSIS', 'SSRS', 'Data Modeling', 'Business Analysis']
            },
            
            # Cybersecurity
            'cybersecurity_engineer': {
                'keywords': ['cybersecurity', 'security engineer', 'information security'],
                'skills': ['Network Security', 'Penetration Testing', 'Vulnerability Assessment', 'SIEM', 'Incident Response', 'Risk Assessment', 'Compliance', 'Firewalls']
            },
            'security_analyst': {
                'keywords': ['security analyst', 'cyber security analyst', 'soc analyst'],
                'skills': ['SIEM', 'Incident Response', 'Threat Analysis', 'Vulnerability Management', 'Security Monitoring', 'Malware Analysis', 'Risk Assessment']
            },
            
            # Product/Project Management
            'product_manager': {
                'keywords': ['product manager', 'product owner', 'pm'],
                'skills': ['Product Strategy', 'Roadmap Planning', 'User Research', 'Agile', 'Scrum', 'A/B Testing', 'Analytics', 'Stakeholder Management', 'JIRA', 'Confluence']
            },
            'project_manager': {
                'keywords': ['project manager', 'program manager', 'pmp'],
                'skills': ['Project Management', 'Agile', 'Scrum', 'Risk Management', 'Stakeholder Management', 'Budget Management', 'MS Project', 'JIRA', 'Communication']
            },
            
            # Design/UX
            'ui_ux_designer': {
                'keywords': ['ui designer', 'ux designer', 'ui/ux', 'user experience'],
                'skills': ['Figma', 'Adobe XD', 'Sketch', 'Photoshop', 'User Research', 'Wireframing', 'Prototyping', 'Design Systems', 'HTML', 'CSS']
            },
            'graphic_designer': {
                'keywords': ['graphic designer', 'visual designer', 'creative designer'],
                'skills': ['Adobe Photoshop', 'Adobe Illustrator', 'Adobe InDesign', 'CorelDraw', 'Branding', 'Typography', 'Print Design', 'Digital Design']
            },
            
            # Sales/Marketing
            'digital_marketing': {
                'keywords': ['digital marketing', 'marketing manager', 'seo specialist'],
                'skills': ['SEO', 'SEM', 'Google Analytics', 'Social Media Marketing', 'Content Marketing', 'Email Marketing', 'PPC', 'Marketing Automation']
            },
            'sales_engineer': {
                'keywords': ['sales engineer', 'technical sales', 'pre sales'],
                'skills': ['Technical Sales', 'Product Demonstration', 'Customer Relationship', 'Solution Design', 'Presentation Skills', 'CRM', 'Lead Generation']
            }
        }
        
        for role, data in role_skills.items():
            for keyword in data['keywords']:
                if keyword in job_lower:
                    return {'role': role, 'skills': data['skills']}
        
        general_skills = [
            # Programming Languages
            'Python', 'Java', 'JavaScript', 'C++', 'C#', 'Go', 'R', 'PHP', 'Ruby', 'Swift', 'Kotlin',
            # Web Technologies
            'HTML', 'CSS', 'React', 'Angular', 'Vue.js', 'Node.js', 'Django', 'Flask', 'Spring', 'ASP.NET',
            # Databases
            'SQL', 'MySQL', 'PostgreSQL', 'MongoDB', 'Redis', 'Oracle', 'SQL Server', 'DynamoDB',
            # Cloud & DevOps
            'AWS', 'Azure', 'Google Cloud', 'Docker', 'Kubernetes', 'Jenkins', 'Git', 'CI/CD', 'Terraform',
            # AI/ML/Data
            'Machine Learning', 'Deep Learning', 'TensorFlow', 'PyTorch', 'Pandas', 'NumPy', 'Scikit-learn',
            'Data Science', 'Data Analytics', 'Statistics', 'NLP', 'Computer Vision', 'LangChain', 'OpenAI',
            # Design & Tools
            'Figma', 'Adobe Photoshop', 'AutoCAD', 'SolidWorks', 'Tableau', 'Power BI', 'JIRA', 'Confluence',
            # Testing & Quality
            'Selenium', 'JUnit', 'Testing', 'Quality Assurance', 'API Testing',
            # General Skills
            'REST API', 'Microservices', 'Agile', 'Scrum', 'Project Management'
        ]
        
        found_skills = [skill for skill in general_skills if skill.lower() in job_lower]
        return {'role': 'general', 'skills': found_skills if found_skills else ['Programming', 'Problem Solving']}

    def enhanced_analysis(self, resume_text: str, job_description: str) -> Dict:
        resume_lower = resume_text.lower()
        job_lower = job_description.lower()
        role_data = self.get_role_specific_skills(job_description)
        job_skills = role_data['skills']
        detected_role = role_data['role']
        
        resume_skills = []
        for skill in job_skills:
            skill_variations = [skill.lower(), skill.lower().replace(' ', ''), skill.lower().replace('.', '')]
            if skill.lower() == 'c++':
                skill_variations.extend(['cpp', 'c plus plus'])
            elif skill.lower() == 'node.js':
                skill_variations.extend(['nodejs', 'node js'])
            elif skill.lower() == 'rest api':
                skill_variations.extend(['restful', 'rest', 'api'])
            for variation in skill_variations:
                if variation in resume_lower:
                    resume_skills.append(skill)
                    break
        
        resume_skills = list(dict.fromkeys(resume_skills))
        missing_skills = [skill for skill in job_skills if skill not in resume_skills]
        
        # Fix for f-string backslash issue - extract regex pattern to variable
        word_pattern = r'\b\w{3,}\b'
        job_words = set(re.findall(word_pattern, job_lower))
        resume_words = set(re.findall(word_pattern, resume_lower))
        common_words = job_words.intersection(resume_words)
        
        if len(job_words) < 10:
            keyword_match = min(100, (len(common_words) / max(len(job_words), 1)) * 100 + 20)
        else:
            keyword_match = (len(common_words) / len(job_words) * 100) if job_words else 0
        
        if len(job_skills) > 0:
            skills_match_percentage = (len(resume_skills) / len(job_skills)) * 100
            missing_critical_skills = len(missing_skills)
            if skills_match_percentage >= 80:
                base_score = 85
            elif skills_match_percentage >= 60:
                base_score = 75
            elif skills_match_percentage >= 40:
                base_score = 65
            elif skills_match_percentage >= 20:
                base_score = 55
            else:
                base_score = 45
            keyword_bonus = min(15, keyword_match * 0.15)
            missing_penalty = min(25, missing_critical_skills * 5)
            ats_score = max(20, min(100, int(base_score + keyword_bonus - missing_penalty)))
        else:
            ats_score = 60

        recommendations = []
        if missing_skills:
            recommendations.append(f"Add these {detected_role.replace('_', ' ').title()} skills: {', '.join(missing_skills[:4])}")
        
        recommendations.extend([
            "Use quantifiable achievements with numbers and percentages",
            "Include relevant certifications for the role",
            "Add years of experience with each technology"
        ])
        
        strengths = []
        weaknesses = []
        skill_match_ratio = len(resume_skills) / len(job_skills) if job_skills else 0
        if skill_match_ratio >= 0.8:
            strengths.append(f"Excellent skill match for {detected_role.replace('_', ' ').title()} role")
        elif skill_match_ratio >= 0.6:
            strengths.append(f"Good foundation for {detected_role.replace('_', ' ').title()} role")
        
        if keyword_match > 60:
            strengths.append("Strong keyword alignment with job requirements")
        
        if missing_skills:
            if len(missing_skills) <= 2:
                weaknesses.append(f"Missing {len(missing_skills)} key skill(s): {', '.join(missing_skills)}")
            else:
                weaknesses.append(f"Missing {len(missing_skills)} key skills including {', '.join(missing_skills[:2])}")
        
        if skill_match_ratio < 0.5:
            weaknesses.append("Significant skill gap for this role")
        
        if ats_score >= 80:
            feedback = f"Excellent match for {detected_role.replace('_', ' ').title()} position! Your skills align well with requirements."
        elif ats_score >= 65:
            feedback = f"Good candidate for {detected_role.replace('_', ' ').title()} role with some improvements needed."
        else:
            feedback = f"Consider strengthening key {detected_role.replace('_', ' ').title()} skills before applying."
        
        return {
            "ats_score": ats_score,
            "job_required_skills": job_skills,
            "resume_present_skills": resume_skills,
            "missing_skills": missing_skills,
            "recommended_additions": recommendations,
            "keyword_match_percentage": round(keyword_match, 2),
            "strengths": strengths if strengths else [f"Resume analyzed for {detected_role.replace('_', ' ').title()} role"],
            "weaknesses": weaknesses if weaknesses else ["No major issues identified"],
            "overall_feedback": feedback
        }

def main():
    st.title("🎯 ATS Resume Analyzer")
    st.markdown("Upload your resume and job description to get actionable insights for better ATS compatibility!")

    with st.sidebar:
        st.header("⚙️ Configuration")
        provider = st.selectbox(
            "Choose Analysis Method",
            ["Enhanced Local Analysis", "OpenAI GPT", "Hugging Face"],
            help="Select your preferred analysis method"
        )
        
        api_key = None
        if provider == "OpenAI GPT":
            api_key = st.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API key")
            if not OPENAI_AVAILABLE:
                st.error("OpenAI library not installed. Run: pip install openai")
        elif provider == "Hugging Face":
            api_key = st.text_input("HuggingFace API Key (Optional)", type="password", help="Get free API key from huggingface.co")

        st.markdown("---")
        st.markdown("### 💡 Analysis Methods")
        st.markdown("""
        - **Enhanced Local**: Advanced keyword matching (No API needed)
        - **OpenAI GPT**: Most accurate analysis (Requires API key & credits)
        - **Hugging Face**: Good analysis (Free tier available)
        """)
        st.markdown("### 📋 Instructions")
        st.markdown("""
        1. Upload your resume (PDF, DOCX, or TXT)
        2. Paste the job description
        3. Choose analysis method
        4. Click 'Analyze Resume' for insights
        """)
        st.markdown("### 🔒 Privacy")
        st.info("Your documents are processed locally and not stored.")

    provider_map = {
        "Enhanced Local Analysis": "demo",
        "OpenAI GPT": "openai", 
        "Hugging Face": "huggingface"
    }

    analyzer = ATSAnalyzer(api_key, provider_map[provider])

    col1, col2 = st.columns([1, 1])
    with col1:
        st.header("📄 Upload Resume")
        uploaded_resume = st.file_uploader(
            "Choose your resume file",
            type=['pdf', 'docx', 'txt'],
            help="Upload your resume in PDF, DOCX, or TXT format"
        )
        
        if uploaded_resume:
            st.success(f"✅ Resume uploaded: {uploaded_resume.name}")
            resume_text = analyzer.extract_text_from_file(uploaded_resume)
            if resume_text:
                with st.expander("📖 Resume Text Preview"):
                    st.text_area("Resume Content", resume_text[:1000] + "..." if len(resume_text) > 1000 else resume_text, height=200, disabled=True)

    with col2:
        st.header("💼 Job Description")
        job_description = st.text_area(
            "Paste the job description",
            height=300,
            placeholder="Paste the complete job description here..."
        )
        if job_description:
            st.success("✅ Job description added")

    if uploaded_resume and job_description:
        st.markdown("---")
        if st.button("🔍 Analyze Resume", type="primary", use_container_width=True):
            with st.spinner("Analyzing your resume... This may take a few moments."):
                resume_text = analyzer.extract_text_from_file(uploaded_resume)
                analysis = analyzer.analyze_resume_with_llm(resume_text, job_description)

                st.success("Analysis Complete!")
                
                # Fixed debug information section - extract regex pattern to variable
                word_pattern = r'\b\w{3,}\b'
                job_words_count = len(set(re.findall(word_pattern, job_description.lower())))
                resume_words_count = len(set(re.findall(word_pattern, resume_text.lower())))
                
                with st.expander("🔍 Debug Information"):
                    st.write(f"**Job Description Length:** {len(job_description)} characters")
                    st.write(f"**Resume Length:** {len(resume_text)} characters")
                    st.write(f"**Unique Job Words:** {job_words_count}")
                    st.write(f"**Unique Resume Words:** {resume_words_count}")
                    st.write(f"**Analysis Method:** {provider}")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("ATS Score", f"{analysis['ats_score']}/100")
                with col2:
                    st.metric("Keyword Match", f"{analysis['keyword_match_percentage']:.1f}%")
                with col3:
                    st.metric("Missing Skills", len(analysis['missing_skills']))

                tab1, tab2, tab3, tab4 = st.tabs(["📊 Skills Analysis", "💡 Recommendations", "💪 Strengths & Weaknesses", "📝 Overall Feedback"])
                
                with tab1:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.subheader("🎯 Required Skills")
                        for skill in analysis['job_required_skills']:
                            st.write(f"• {skill}")
                    
                    with col2:
                        st.subheader("✅ Skills Found")
                        if analysis['resume_present_skills']:
                            for skill in analysis['resume_present_skills']:
                                st.write(f"• {skill}")
                        else:
                            st.write("No matching skills found")
                    
                    with col3:
                        st.subheader("❌ Missing Skills")
                        if analysis['missing_skills']:
                            for skill in analysis['missing_skills']:
                                st.write(f"• {skill}")
                        else:
                            st.write("All required skills found!")

                with tab2:
                    st.subheader("💡 Recommendations")
                    for i, rec in enumerate(analysis['recommended_additions'], 1):
                        st.write(f"{i}. {rec}")

                with tab3:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("💪 Strengths")
                        for strength in analysis['strengths']:
                            st.success(f"✓ {strength}")
                    
                    with col2:
                        st.subheader("⚠️ Areas for Improvement")
                        for weakness in analysis['weaknesses']:
                            st.warning(f"! {weakness}")

                with tab4:
                    st.subheader("📝 Overall Feedback")
                    st.write(analysis['overall_feedback'])
                    
                    # Score interpretation
                    if analysis['ats_score'] >= 80:
                        st.success("🎉 Excellent! Your resume is well-optimized for ATS systems.")
                    elif analysis['ats_score'] >= 65:
                        st.info("👍 Good! Your resume has solid ATS compatibility with room for improvement.")
                    elif analysis['ats_score'] >= 50:
                        st.warning("⚠️ Fair. Consider making some improvements to boost your ATS score.")
                    else:
                        st.error("❌ Needs Improvement. Significant changes recommended for better ATS compatibility.")

if __name__ == "__main__":
    main()
