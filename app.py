import streamlit as st
import traceback
import logging
from resume_parser import parse_resume, parse_pdf, parse_docx
from matcher import compute_similarity, extract_skills_dynamic, get_skill_frequency_analysis
from feedback import rule_based_feedback, ai_feedback, test_hf_connection

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="AI Resume Analyzer", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# Enhanced styling
st.markdown("""
<style>
    .stApp { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #ffffff;
    }
    
    .main-header {
        text-align: center;
        padding: 2rem 0;
        margin-bottom: 2rem;
    }
    
    .main-header h1 {
        color: #ffffff;
        font-size: 3rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        margin-bottom: 0.5rem;
    }
    
    .main-header p {
        color: #f0f0f0;
        font-size: 1.2rem;
        opacity: 0.9;
    }
    
    .metric-card {
        background: rgba(255,255,255,0.1);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.2);
        backdrop-filter: blur(10px);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #00ff88;
        text-shadow: 0 0 20px rgba(0,255,136,0.3);
    }
    
    .metric-label {
        font-size: 1rem;
        color: #ffffff;
        opacity: 0.8;
        margin-top: 0.5rem;
    }
    
    .skill-tag {
        display: inline-block;
        background: rgba(0,255,136,0.2);
        color: #00ff88;
        padding: 0.3rem 0.8rem;
        margin: 0.2rem;
        border-radius: 15px;
        font-size: 0.9rem;
        border: 1px solid rgba(0,255,136,0.3);
    }
    
    .missing-skill {
        background: rgba(255,100,100,0.2);
        color: #ff6464;
        border: 1px solid rgba(255,100,100,0.3);
    }
    
    .warning-box {
        background: rgba(255,165,0,0.1);
        border: 1px solid rgba(255,165,0,0.3);
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .success-box {
        background: rgba(0,255,136,0.1);
        border: 1px solid rgba(0,255,136,0.3);
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown("""
<div class="main-header">
    <h1>🤖 AI Resume Analyzer</h1>
    <p>Get intelligent feedback on how well your resume matches job requirements</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if "analysis_complete" not in st.session_state:
    st.session_state.update({
        "analysis_complete": False,
        "resume_text": "",
        "resume_validation": {},
        "similarity_score": 0.0,
        "matched_skills": [],
        "missing_skills": [],
        "rule_feedback": "",
        "ai_feedback": "",
        "skill_analysis": {}
    })

# Create layout
col1, col2 = st.columns([1, 1.5])

with col1:
    st.markdown("### 📄 Upload Resume")
    uploaded_file = st.file_uploader(
        "Choose your resume file", 
        type=["pdf", "docx"],
        help="Upload your resume in PDF or DOCX format"
    )
    
    # Show file analysis if file is uploaded
    if uploaded_file:
        with st.expander("📊 File Analysis", expanded=False):
            if st.button("Analyze File Quality"):
                with st.spinner("Analyzing uploaded file..."):
                    try:
                        parse_result = parse_resume(uploaded_file)
                        
                        if parse_result['success']:
                            validation = parse_result['validation']
                            
                            # Show stats
                            stats = validation['stats']
                            col_stats1, col_stats2 = st.columns(2)
                            with col_stats1:
                                st.metric("Words", stats['word_count'])
                                st.metric("Characters", stats['character_count'])
                            with col_stats2:
                                st.metric("Lines", stats['line_count'])
                                
                            # Show validation results
                            if validation['is_valid']:
                                st.success("✅ Resume format looks good!")
                            else:
                                st.error("⚠️ Resume may have formatting issues")
                                
                            # Show warnings
                            for warning in validation['warnings']:
                                st.warning(f"⚠️ {warning}")
                                
                            # Show suggestions
                            for suggestion in validation['suggestions']:
                                st.info(f"💡 {suggestion}")
                                
                            # Show detected sections
                            sections = parse_result['sections']
                            if sections:
                                st.write("**Detected Sections:**")
                                for section_name, content in sections.items():
                                    if content and content.strip():
                                        st.write(f"• {section_name.title()}: {len(content.split())} words")
                        else:
                            st.error(f"Error analyzing file: {parse_result['error']}")
                    except Exception as e:
                        st.error(f"Error analyzing file: {str(e)}")
    
    st.markdown("### 📋 Job Description")
    jd_text = st.text_area(
        "Paste the job description here",
        height=200,
        placeholder="Paste the complete job description including requirements, responsibilities, and qualifications...",
        help="Include the complete job posting for better analysis"
    )
    
    # API Status Check - Fixed to prevent import-time execution
    with st.expander("🔧 API Configuration Status", expanded=False):
        if st.button("Test AI Connection"):
            with st.spinner("Testing API connection..."):
                try:
                    success, message = test_hf_connection()
                    if success:
                        st.success(f"✅ API connection successful: {message}")
                    else:
                        st.error(f"❌ API connection failed: {message}")
                        
                        # Show setup instructions based on error
                        if "huggingface_hub not installed" in message:
                            st.info("""
                            **Setup Required:**
                            1. Install: `pip install huggingface_hub`
                            2. Get token from: https://huggingface.co/settings/tokens
                            3. Add to .env file: HF_API_KEY=your_token_here
                            4. Restart the application
                            """)
                        elif "not found in .env" in message:
                            st.info("""
                            **API Key Missing:**
                            1. Get token from: https://huggingface.co/settings/tokens
                            2. Add to .env file: HF_API_KEY=your_token_here
                            3. Restart the application
                            """)
                        else:
                            st.info("""
                            **General Fixes:**
                            1. Check your .env file has HF_API_KEY
                            2. Ensure token starts with 'hf_'
                            3. Try regenerating your token
                            4. Wait 2-3 minutes for models to load
                            """)
                except Exception as e:
                    st.error(f"Error testing connection: {str(e)}")
    
    # Analyze button
    if st.button("🚀 Analyze Resume", type="primary"):
        if not uploaded_file:
            st.error("❌ Please upload your resume first")
        elif not jd_text.strip():
            st.error("❌ Please paste the job description")
        else:
            with st.spinner("🔍 Analyzing your resume... This may take a few moments"):
                try:
                    # Parse resume with enhanced parser
                    progress_bar = st.progress(0)
                    st.text("Step 1/6: Parsing resume...")
                    progress_bar.progress(10)
                    
                    parse_result = parse_resume(uploaded_file)
                    
                    if not parse_result['success']:
                        st.error(f"❌ Error parsing resume: {parse_result['error']}")
                        st.session_state.analysis_complete = False
                    else:
                        resume_text = parse_result['text']
                        validation = parse_result['validation']
                        
                        # Show parsing warnings if any
                        if validation['warnings']:
                            with st.expander("⚠️ Resume Parsing Warnings", expanded=False):
                                for warning in validation['warnings']:
                                    st.warning(warning)
                        
                        if not resume_text or len(resume_text.strip()) < 50:
                            st.error("❌ Could not extract meaningful text from your resume. Please check the file.")
                            st.session_state.analysis_complete = False
                        else:
                            # Perform analysis
                            st.text("Step 2/6: Extracting skills...")
                            progress_bar.progress(30)
                            
                            # Step 1: Extract and match skills
                            matched_skills, missing_skills = extract_skills_dynamic(resume_text, jd_text)
                            logger.info(f"Skill extraction complete: {len(matched_skills)} matched, {len(missing_skills)} missing")
                            
                            st.text("Step 3/6: Computing similarity score...")
                            progress_bar.progress(50)
                            
                            # Step 2: Compute similarity based on skills
                            similarity = compute_similarity(resume_text, jd_text)
                            logger.info(f"Similarity score: {similarity}%")
                            
                            st.text("Step 4/6: Analyzing skill frequencies...")
                            progress_bar.progress(70)
                            
                            # Step 3: Skill frequency analysis
                            skill_analysis = get_skill_frequency_analysis(resume_text, jd_text)
                            
                            st.text("Step 5/6: Generating rule-based feedback...")
                            progress_bar.progress(85)
                            
                            # Step 4: Generate rule-based feedback
                            rule_feedback = rule_based_feedback(matched_skills, missing_skills, resume_text)
                            
                            st.text("Step 6/6: Generating AI feedback...")
                            progress_bar.progress(95)
                            
                            # Step 5: Generate AI feedback
                            ai_feedback_text = ai_feedback(resume_text, jd_text)
                            logger.info("AI feedback generation complete")
                            
                            progress_bar.progress(100)
                            
                            # Update session state
                            st.session_state.update({
                                "analysis_complete": True,
                                "resume_text": resume_text,
                                "resume_validation": validation,
                                "similarity_score": similarity,
                                "matched_skills": matched_skills,
                                "missing_skills": missing_skills,
                                "rule_feedback": rule_feedback,
                                "ai_feedback": ai_feedback_text,
                                "skill_analysis": skill_analysis
                            })
                            
                            st.success("✅ Analysis completed successfully!")
                            progress_bar.empty()
                            st.empty()  # Clear the status text
                        
                except Exception as e:
                    logger.error(f"Analysis failed: {str(e)}")
                    st.error(f"❌ Analysis failed: {str(e)}")
                    with st.expander("Debug Information", expanded=False):
                        st.code(traceback.format_exc())
                    st.session_state.analysis_complete = False

# Results section
with col2:
    if st.session_state.analysis_complete:
        st.markdown("### 📊 Analysis Results")
        
        # Show resume quality indicators
        if st.session_state.resume_validation:
            validation = st.session_state.resume_validation
            if validation['warnings'] or validation['suggestions']:
                with st.expander("🔍 Resume Quality Check", expanded=False):
                    if validation['warnings']:
                        for warning in validation['warnings']:
                            st.markdown(f'<div class="warning-box">⚠️ {warning}</div>', unsafe_allow_html=True)
                    
                    if validation['suggestions']:
                        st.write("**Suggestions for improvement:**")
                        for suggestion in validation['suggestions']:
                            st.write(f"💡 {suggestion}")
        
        # Similarity Score Card
        similarity_score = st.session_state.similarity_score
        score_color = "#00ff88" if similarity_score > 70 else "#ffa500" if similarity_score > 50 else "#ff6464"
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: {score_color};">{similarity_score}%</div>
            <div class="metric-label">Resume-Job Match Score</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Skills Analysis
        with st.expander("🎯 Skills Analysis", expanded=True):
            col_matched, col_missing = st.columns(2)
            
            with col_matched:
                st.markdown("**✅ Matched Skills:**")
                if st.session_state.matched_skills:
                    skills_html = ""
                    display_skills = st.session_state.matched_skills[:12]  # Show first 12
                    for skill in display_skills:
                        skills_html += f'<span class="skill-tag">{skill}</span>'
                    st.markdown(skills_html, unsafe_allow_html=True)
                    
                    if len(st.session_state.matched_skills) > 12:
                        with st.expander(f"View all {len(st.session_state.matched_skills)} matched skills"):
                            remaining_skills = st.session_state.matched_skills[12:]
                            st.write(", ".join(remaining_skills))
                else:
                    st.info("No matching skills found - consider adding relevant skills from the job description")
            
            with col_missing:
                st.markdown("**⚠️ Missing Skills:**")
                if st.session_state.missing_skills:
                    missing_html = ""
                    display_missing = st.session_state.missing_skills[:10]  # Show first 10
                    for skill in display_missing:
                        missing_html += f'<span class="skill-tag missing-skill">{skill}</span>'
                    st.markdown(missing_html, unsafe_allow_html=True)
                    
                    if len(st.session_state.missing_skills) > 10:
                        with st.expander(f"View all {len(st.session_state.missing_skills)} missing skills"):
                            remaining_missing = st.session_state.missing_skills[10:]
                            st.write(", ".join(remaining_missing))
                else:
                    st.success("All key skills found!")
            
            # Skills priority analysis
            if st.session_state.skill_analysis:
                st.markdown("**📈 Most Important Skills (by JD frequency):**")
                priority_skills = list(st.session_state.skill_analysis.items())[:5]
                for skill, count in priority_skills:
                    status = "✅" if skill in st.session_state.matched_skills else "❌"
                    st.write(f"{status} **{skill}** (mentioned {count}x in JD)")
        
        # Rule-based Feedback
        with st.expander("📝 Detailed Analysis", expanded=True):
            st.markdown(st.session_state.rule_feedback)
        
        # AI Feedback
        with st.expander("🤖 AI-Powered Insights", expanded=True):
            ai_feedback_text = st.session_state.ai_feedback
            
            # Check if it's AI-generated or fallback
            if "AI Analysis Complete" in ai_feedback_text:
                st.success("🤖 AI Analysis Complete:")
            elif ("API temporarily unavailable" in ai_feedback_text or 
                  "unavailable" in ai_feedback_text.lower() or
                  "Temporarily Unavailable" in ai_feedback_text):
                st.warning("🔧 AI feedback is currently unavailable. Using intelligent analysis instead.")
            else:
                st.info("🤖 Comprehensive Analysis:")
            
            st.markdown(ai_feedback_text)
        
        # Resume Text Preview
        with st.expander("📄 Extracted Resume Text", expanded=False):
            preview_text = st.session_state.resume_text
            if len(preview_text) > 2000:
                preview_text = preview_text[:2000] + "\n\n... (truncated for display)"
            
            st.text_area(
                "Resume content (extracted):",
                preview_text,
                height=200,
                disabled=True
            )
            st.info(f"Total characters extracted: {len(st.session_state.resume_text)}")
        
        # Action Items Summary
        st.markdown("### 🎯 Quick Action Items")
        
        action_items = []
        
        # Priority actions based on analysis
        if st.session_state.missing_skills:
            top_missing = ", ".join(st.session_state.missing_skills[:3])
            action_items.append(f"🔹 **Add these key skills:** {top_missing}")
        
        if similarity_score < 60:
            action_items.append("🔹 **Align your experience** more closely with job requirements")
        
        if len(st.session_state.matched_skills) < 5:
            action_items.append("🔹 **Highlight more relevant technical skills** from your experience")
        
        # Always include these general improvements
        action_items.extend([
            "🔹 **Quantify your achievements** with specific metrics and outcomes",
            "🔹 **Use keywords from the job description** throughout your resume",
            "🔹 **Optimize formatting** for ATS (Applicant Tracking Systems)",
            "🔹 **Tailor your professional summary** to match the role"
        ])
        
        for item in action_items:
            st.markdown(item)
            
    else:
        st.markdown("### 👆 Upload your resume and job description to get started")
        st.info("""
        **How it works:**
        1. **Upload your resume** (PDF or DOCX format)
        2. **Paste the job description** you're targeting
        3. **Click Analyze** to get detailed feedback
        4. **Improve your resume** based on the suggestions
        
        **What you'll get:**
        - **Match Score:** Percentage alignment with job requirements
        - **Smart Skills Analysis:** Extracts actual skills from JD (not generic words)
        - **Missing Skills:** Important skills you should add
        - **Rule-based Feedback:** Structured analysis and suggestions
        - **AI Insights:** Personalized feedback from advanced language models
        - **Action Items:** Specific steps to improve your resume
        
        **Works for all roles:** Tech, Finance, Marketing, Healthcare, HR, and more!
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; opacity: 0.7; padding: 1rem;">
    <p>🚀 Built with Streamlit | 🤖 Powered by HuggingFace Hub | 💡 Enhanced with Smart NLP</p>
    <p>🔧 Install: pip install huggingface_hub for AI features</p>
</div>
""", unsafe_allow_html=True)