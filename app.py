import streamlit as st
import traceback
import logging
from resume_parser import parse_resume, parse_pdf, parse_docx
from matcher import compute_similarity, extract_skills_from_text, get_feature_details
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

# Header
st.markdown("""
<div class="main-header">
    <h1>🤖 AI Resume Analyzer</h1>
    <p>Get intelligent feedback on how well your resume matches job requirements</p>
</div>
""", unsafe_allow_html=True)

# Session state
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
        "skill_analysis": {},
        "feature_details": {}
    })

col1, col2 = st.columns([1, 1.5])

# ---------------------- Left Section ----------------------
with col1:
    st.markdown("### 📄 Upload Resume")
    uploaded_file = st.file_uploader(
        "Choose your resume file", 
        type=["pdf", "docx"],
        help="Upload your resume in PDF or DOCX format"
    )

    if uploaded_file:
        with st.expander("📊 File Analysis", expanded=False):
            if st.button("Analyze File Quality"):
                with st.spinner("Analyzing uploaded file..."):
                    try:
                        parse_result = parse_resume(uploaded_file)
                        if parse_result['success']:
                            validation = parse_result['validation']
                            stats = validation['stats']
                            col_stats1, col_stats2 = st.columns(2)
                            with col_stats1:
                                st.metric("Words", stats['word_count'])
                                st.metric("Characters", stats['character_count'])
                            with col_stats2:
                                st.metric("Lines", stats['line_count'])
                            if validation['is_valid']:
                                st.success("✅ Resume format looks good!")
                            else:
                                st.error("⚠️ Resume may have formatting issues")
                            for warning in validation['warnings']:
                                st.warning(f"⚠️ {warning}")
                            for suggestion in validation['suggestions']:
                                st.info(f"💡 {suggestion}")
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
        placeholder="Paste the complete job description...",
        help="Include the full job posting for better analysis"
    )

    with st.expander("🔧 API Configuration Status", expanded=False):
        if st.button("Test AI Connection"):
            with st.spinner("Testing API connection..."):
                try:
                    success, message = test_hf_connection()
                    if success:
                        st.success(f"✅ API connection successful: {message}")
                    else:
                        st.error(f"❌ API connection failed: {message}")
                except Exception as e:
                    st.error(f"Error testing connection: {str(e)}")

    if st.button("🚀 Analyze Resume", type="primary"):
        if not uploaded_file:
            st.error("❌ Please upload your resume first")
        elif not jd_text.strip():
            st.error("❌ Please paste the job description")
        else:
            with st.spinner("🔍 Analyzing your resume... This may take a few moments"):
                try:
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
                        progress_bar.progress(30)
                        st.text("Step 2/6: Extracting skills...")

                        # Updated skill extraction
                        resume_skills = extract_skills_from_text(resume_text)
                        jd_skills = extract_skills_from_text(jd_text)
                        matched_skills = list(resume_skills & jd_skills)
                        missing_skills = list(jd_skills - resume_skills)
                        logger.info(f"Skill extraction complete: {len(matched_skills)} matched, {len(missing_skills)} missing")

                        st.text("Step 3/6: Computing ANN-based similarity score...")
                        progress_bar.progress(50)
                        similarity = compute_similarity(resume_text, jd_text)

                        st.text("Step 4/6: Extracting ANN feature details...")
                        progress_bar.progress(70)
                        feature_details = get_feature_details(resume_text, jd_text)

                        st.text("Step 5/6: Generating rule-based feedback...")
                        progress_bar.progress(85)
                        rule_feedback = rule_based_feedback(matched_skills, missing_skills, resume_text)

                        st.text("Step 6/6: Generating AI feedback...")
                        progress_bar.progress(95)
                        ai_feedback_text = ai_feedback(resume_text, jd_text)
                        progress_bar.progress(100)

                        st.session_state.update({
                            "analysis_complete": True,
                            "resume_text": resume_text,
                            "resume_validation": validation,
                            "similarity_score": similarity,
                            "matched_skills": matched_skills,
                            "missing_skills": missing_skills,
                            "rule_feedback": rule_feedback,
                            "ai_feedback": ai_feedback_text,
                            "skill_analysis": feature_details,
                            "feature_details": feature_details
                        })

                        st.success("✅ Analysis completed successfully!")
                        progress_bar.empty()
                except Exception as e:
                    logger.error(f"Analysis failed: {str(e)}")
                    st.error(f"❌ Analysis failed: {str(e)}")
                    with st.expander("Debug Information", expanded=False):
                        st.code(traceback.format_exc())
                    st.session_state.analysis_complete = False

# ---------------------- Right Section ----------------------
with col2:
    if st.session_state.analysis_complete:
        st.markdown("### 📊 Analysis Results")

        similarity_score = st.session_state.similarity_score
        score_color = "#00ff88" if similarity_score > 70 else "#ffa500" if similarity_score > 50 else "#ff6464"

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: {score_color};">{similarity_score}%</div>
            <div class="metric-label">AI Resume Match Score (via ANN)</div>
        </div>
        """, unsafe_allow_html=True)

        # ANN feature breakdown
        feature_details = st.session_state.get("feature_details", {})
        if feature_details:
            with st.expander("🧠 ANN Feature Breakdown", expanded=False):
                st.write(f"**Skill Match Ratio:** {feature_details.get('skill_match_ratio', 0)}")
                st.write(f"**Semantic Similarity:** {feature_details.get('semantic_similarity', 0)}")
                st.write(f"**Resume Skills:** {len(feature_details.get('resume_skills', []))}")
                st.write(f"**JD Skills:** {len(feature_details.get('jd_skills', []))}")

        # Skills Section
        with st.expander("🎯 Skills Analysis", expanded=True):
            col_matched, col_missing = st.columns(2)
            with col_matched:
                st.markdown("**✅ Matched Skills:**")
                if st.session_state.matched_skills:
                    for skill in st.session_state.matched_skills:
                        st.markdown(f'<span class="skill-tag">{skill}</span>', unsafe_allow_html=True)
                else:
                    st.info("No matching skills found.")
            with col_missing:
                st.markdown("**⚠️ Missing Skills:**")
                if st.session_state.missing_skills:
                    for skill in st.session_state.missing_skills:
                        st.markdown(f'<span class="skill-tag missing-skill">{skill}</span>', unsafe_allow_html=True)
                else:
                    st.success("All key skills found!")

        with st.expander("📝 Detailed Analysis", expanded=True):
            st.markdown(st.session_state.rule_feedback)

        with st.expander("🤖 AI-Powered Insights", expanded=True):
            st.markdown(st.session_state.ai_feedback)

        with st.expander("📄 Extracted Resume Text", expanded=False):
            preview_text = st.session_state.resume_text[:2000]
            st.text_area("Resume content:", preview_text, height=200, disabled=True)

        st.markdown("### 🎯 Quick Action Items")
        action_items = []
        if st.session_state.missing_skills:
            top_missing = ", ".join(st.session_state.missing_skills[:3])
            action_items.append(f"🔹 **Add these key skills:** {top_missing}")
        if similarity_score < 60:
            action_items.append("🔹 **Align your experience** more closely with job requirements")
        if len(st.session_state.matched_skills) < 5:
            action_items.append("🔹 **Highlight more relevant technical skills** from your experience")
        action_items.extend([
            "🔹 **Quantify achievements** with measurable metrics",
            "🔹 **Use keywords** from the job description",
            "🔹 **Optimize formatting** for ATS systems"
        ])
        for item in action_items:
            st.markdown(item)

    else:
        st.markdown("### 👆 Upload your resume and job description to get started")
        st.info("""
        1. Upload your resume (PDF/DOCX)
        2. Paste the job description
        3. Click "Analyze" to view ANN-powered insights
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; opacity: 0.7; padding: 1rem;">
    <p>🚀 Built with Streamlit | 🤖 Powered by ANN & NLP</p>
</div>
""", unsafe_allow_html=True)
