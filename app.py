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
        """Initialize the ATS analyzer with different LLM providers"""
        self.provider = provider
        self.api_key = api_key
        
        if provider == "openai" and api_key and OPENAI_AVAILABLE:
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = None
    
    def extract_text_from_pdf(self, uploaded_file) -> str:
        """Extract text from PDF file"""
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
        """Extract text from DOCX file"""
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
        """Extract text based on file type"""
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
        """Use Hugging Face free API for analysis"""
        try:
            # Use a simple approach with Hugging Face Inference API (free tier)
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
        """Analyze resume against job description using selected LLM provider"""
        if self.provider == "openai" and self.client:
            return self.analyze_with_openai(resume_text, job_description)
        elif self.provider == "huggingface":
            return self.analyze_with_huggingface(resume_text, job_description)
        else:
            return self.enhanced_analysis(resume_text, job_description)
    
    def analyze_with_openai(self, resume_text: str, job_description: str) -> Dict:
        """OpenAI analysis method"""
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

        Be specific and actionable in your recommendations. Focus on ATS optimization, keyword matching, and relevant skills alignment.
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
            
            # Parse the JSON response
            analysis_text = response.choices[0].message.content
            # Extract JSON from the response
            json_start = analysis_text.find('{')
            json_end = analysis_text.rfind('}') + 1
            json_str = analysis_text[json_start:json_end]
            
            return json.loads(json_str)
            
        except Exception as e:
            st.error(f"Error with OpenAI analysis: {str(e)}")
            st.info("Falling back to enhanced local analysis...")
            return self.enhanced_analysis(resume_text, job_description)
    
    def get_role_specific_skills(self, job_description: str) -> Dict:
        """Get role-specific required skills based on job description keywords"""
        job_lower = job_description.lower()
        
        # Define role-specific skill sets
        role_skills = {
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
            'data_scientist': {
                'keywords': ['data scientist', 'data science', 'machine learning'],
                'skills': ['Python', 'R', 'Machine Learning', 'Pandas', 'NumPy', 'Scikit-learn', 'TensorFlow', 'SQL', 'Jupyter', 'Statistics', 'Data Visualization']
            },
            'devops_engineer': {
                'keywords': ['devops', 'sre', 'site reliability'],
                'skills': ['Docker', 'Kubernetes', 'AWS', 'Jenkins', 'Git', 'Linux', 'Terraform', 'Ansible', 'CI/CD', 'Monitoring', 'Bash']
            },
            'cpp_developer': {
                'keywords': ['c++ developer', 'c++', 'cpp'],
                'skills': ['C++', 'Object Oriented Programming', 'STL', 'Data Structures', 'Algorithms', 'GCC', 'CMake', 'Git', 'Linux', 'Debugging']
            },
            'mobile_developer': {
                'keywords': ['android developer', 'ios developer', 'mobile developer', 'react native', 'flutter'],
                'skills': ['Java', 'Kotlin', 'Swift', 'React Native', 'Flutter', 'Android Studio', 'Xcode', 'REST API', 'Git', 'Mobile UI/UX']
            }
        }
        
        # Find matching role
        for role, data in role_skills.items():
            for keyword in data['keywords']:
                if keyword in job_lower:
                    return {'role': role, 'skills': data['skills']}
        
        # If no specific role found, extract general skills
        general_skills = [
            'Python', 'Java', 'JavaScript', 'C++', 'SQL', 'HTML', 'CSS', 'React', 'Angular', 'Vue.js',
            'Node.js', 'Django', 'Flask', 'Spring', 'Docker', 'Kubernetes', 'AWS', 'Azure', 'GCP',
            'Machine Learning', 'Data Science', 'Git', 'REST API', 'MongoDB', 'PostgreSQL', 'MySQL'
        ]
        
        found_skills = []
        for skill in general_skills:
            if skill.lower() in job_lower:
                found_skills.append(skill)
        
        return {'role': 'general', 'skills': found_skills if found_skills else ['Programming', 'Problem Solving']}

    def enhanced_analysis(self, resume_text: str, job_description: str) -> Dict:
        """Enhanced local analysis without external APIs"""
        resume_lower = resume_text.lower()
        job_lower = job_description.lower()
        
        # Get role-specific skills
        role_data = self.get_role_specific_skills(job_description)
        job_skills = role_data['skills']
        detected_role = role_data['role']
        
        # Find skills present in resume
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
        
        # Remove duplicates
        resume_skills = list(dict.fromkeys(resume_skills))
        
        # Find missing skills
        missing_skills = [skill for skill in job_skills if skill not in resume_skills]
        
        # Calculate keyword match (more lenient for short job descriptions)
        job_words = set(re.findall(r'\b\w{3,}\b', job_lower))
        resume_words = set(re.findall(r'\b\w{3,}\b', resume_lower))
        common_words = job_words.intersection(resume_words)
        
        if len(job_words) < 10:  # Short job description
            keyword_match = min(100, (len(common_words) / max(len(job_words), 1)) * 100 + 20)
        else:
            keyword_match = (len(common_words) / len(job_words) * 100) if job_words else 0
        
        # Calculate ATS score
        if len(job_skills) > 0:
            skills_match_percentage = (len(resume_skills) / len(job_skills)) * 100
            missing_critical_skills = len(missing_skills)
            
            # Base score calculation
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
            
            # Adjust for keyword match
            keyword_bonus = min(15, keyword_match * 0.15)
            
            # Penalty for missing skills
            missing_penalty = min(25, missing_critical_skills * 5)
            
            ats_score = max(20, min(100, int(base_score + keyword_bonus - missing_penalty)))
        else:
            ats_score = 60  # Default score
        
        # Generate role-specific recommendations
        recommendations = []
        if missing_skills:
            recommendations.append(f"Add these {detected_role.replace('_', ' ').title()} skills: {', '.join(missing_skills[:4])}")
        
        if detected_role == 'java_developer':
            recommendations.extend([
                "Include specific Java frameworks (Spring Boot, Hibernate)",
                "Mention build tools (Maven/Gradle) experience",
                "Add database experience (MySQL, PostgreSQL)"
            ])
        elif detected_role == 'frontend_developer':
            recommendations.extend([
                "Showcase responsive design skills",
                "Include CSS preprocessors (Sass/Less) if applicable",
                "Mention JavaScript frameworks and libraries"
            ])
        elif detected_role == 'python_developer':
            recommendations.extend([
                "Include Python web frameworks (Django/Flask)",
                "Add data manipulation libraries (Pandas, NumPy)",
                "Mention API development experience"
            ])
        
        recommendations.extend([
            "Use quantifiable achievements with numbers and percentages",
            "Include relevant certifications for the role",
            "Add years of experience with each technology"
        ])
        
        # Strengths and weaknesses
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
        
        # Overall feedback
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
    
    # Sidebar for API configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # LLM Provider selection
        provider = st.selectbox(
            "Choose Analysis Method",
            ["Enhanced Local Analysis", "OpenAI GPT", "Hugging Face"],
            help="Select your preferred analysis method"
        )
        
        api_key = None
        if provider == "OpenAI GPT":
            api_key = st.text_input("OpenAI API Key", type="password", 
                                   help="Enter your OpenAI API key")
            if not OPENAI_AVAILABLE:
                st.error("OpenAI library not installed. Run: pip install openai")
        elif provider == "Hugging Face":
            api_key = st.text_input("HuggingFace API Key (Optional)", type="password",
                                   help="Get free API key from huggingface.co")
        
        st.markdown("---")
        st.markdown("### 💡 Analysis Methods")
        st.markdown("""
        - **Enhanced Local**: Advanced keyword matching (No API needed)
        - **OpenAI GPT**: Most accurate analysis (Requires API key & credits)
        - **Hugging Face**: Good analysis (Free tier available)
        """)
        
        st.markdown("---")
        st.markdown("### 📋 Instructions")
        st.markdown("""
        1. Upload your resume (PDF, DOCX, or TXT)
        2. Paste the job description
        3. Choose analysis method
        4. Click 'Analyze Resume' for insights
        """)
        
        st.markdown("---")
        st.markdown("### 🔒 Privacy")
        st.info("Your documents are processed locally and not stored.")
    
    # Map provider names to internal values
    provider_map = {
        "Enhanced Local Analysis": "demo",
        "OpenAI GPT": "openai", 
        "Hugging Face": "huggingface"
    }
    
    # Initialize analyzer
    analyzer = ATSAnalyzer(api_key, provider_map[provider])
    
    # Main content area
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
            
            # Extract and display text preview
            resume_text = analyzer.extract_text_from_file(uploaded_resume)
            if resume_text:
                with st.expander("📖 Resume Text Preview"):
                    st.text_area("Resume Content", resume_text[:1000] + "..." if len(resume_text) > 1000 else resume_text, 
                               height=200, disabled=True)
    
    with col2:
        st.header("💼 Job Description")
        job_description = st.text_area(
            "Paste the job description",
            height=300,
            placeholder="Paste the complete job description here including requirements, responsibilities, and qualifications..."
        )
        
        if job_description:
            st.success("✅ Job description added")
    
    # Analysis section
    if uploaded_resume and job_description:
        st.markdown("---")
        
        if st.button("🔍 Analyze Resume", type="primary", use_container_width=True):
            with st.spinner("Analyzing your resume... This may take a few moments."):
                resume_text = analyzer.extract_text_from_file(uploaded_resume)
                analysis = analyzer.analyze_resume_with_llm(resume_text, job_description)
                
                # Display results with debug info
                st.success("Analysis Complete!")
                
                # Debug information in expander
                with st.expander("🔍 Debug Information"):
                    st.write(f"**Job Description Length:** {len(job_description)} characters")
                    st.write(f"**Resume Length:** {len(resume_text)} characters")
                    st.write(f"**Unique Job Words:** {len(set(re.findall(r'\\b\\w{3,}\\b', job_description.lower())))}")
                    st.write(f"**Unique Resume Words:** {len(set(re.findall(r'\\b\\w{3,}\\b', resume_text.lower())))}")
                    st.write(f"**Analysis Method:** {provider}")
                
                # Score section
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("ATS Score", f"{analysis['ats_score']}/100", 
                             delta=f"{analysis['ats_score'] - 70}%" if analysis['ats_score'] > 70 else f"{analysis['ats_score'] - 70}%")
                with col2:
                    st.metric("Keyword Match", f"{analysis['keyword_match_percentage']:.1f}%")
                with col3:
                    missing_count = len(analysis['missing_skills'])
                    st.metric("Missing Skills", missing_count, delta=-missing_count if missing_count > 0 else 0)
                
                # Detailed analysis
                tab1, tab2, tab3, tab4 = st.tabs(["📊 Skills Analysis", "💡 Recommendations", "💪 Strengths & Weaknesses", "📝 Overall Feedback"])
                
                with tab1:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.subheader("🎯 Required Skills")
                        for skill in analysis['job_required_skills']:
                            st.write(f"• {skill}")
                    
                    with col2:
                        st.subheader("✅ Your Skills")
                        for skill in analysis['resume_present_skills']:
                            st.write(f"• {skill}")
                    
                    with col3:
                        st.subheader("❌ Missing Skills")
                        if analysis['missing_skills']:
                            for skill in analysis['missing_skills']:
                                st.write(f"• {skill}")
                        else:
                            st.success("Great! You have all required skills.")
                
                with tab2:
                    st.subheader("🚀 Recommended Improvements")
                    for i, recommendation in enumerate(analysis['recommended_additions'], 1):
                        st.write(f"{i}. {recommendation}")
                
                with tab3:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("💪 Strengths")
                        for strength in analysis['strengths']:
                            st.success(f"✓ {strength}")
                    
                    with col2:
                        st.subheader("⚠️ Areas for Improvement")
                        for weakness in analysis['weaknesses']:
                            st.warning(f"→ {weakness}")
                
                with tab4:
                    st.subheader("📋 Detailed Feedback")
                    st.write(analysis['overall_feedback'])
                
                # Action items
                st.markdown("---")
                st.subheader("🎯 Next Steps")
                
                if analysis['missing_skills']:
                    st.error("**Priority Actions:**")
                    st.write("1. Add the missing skills to your resume if you possess them")
                    st.write("2. Consider learning the missing skills that are critical for the role")
                    st.write("3. Use similar keywords and phrases from the job description")
                
                if analysis['ats_score'] < 80:
                    st.warning("**ATS Optimization Needed:**")
                    st.write("- Increase keyword density for important terms")
                    st.write("- Use standard section headings (Experience, Education, Skills)")
                    st.write("- Include more quantifiable achievements")
                
                # Download report option
                st.markdown("---")
                report_data = {
                    "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "filename": uploaded_resume.name,
                    "analysis": analysis
                }
                
                st.download_button(
                    label="📥 Download Analysis Report",
                    data=json.dumps(report_data, indent=2),
                    file_name=f"ats_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

if __name__ == "__main__":
    main()