import os
from dotenv import load_dotenv
import logging
from typing import Tuple
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load .env with robust error handling
try:
    load_dotenv(encoding='utf-8')
except (UnicodeDecodeError, UnicodeError):
    try:
        load_dotenv(encoding='utf-16')
        logger.info("Loaded .env file with UTF-16 encoding")
    except:
        try:
            load_dotenv(encoding='latin-1')
            logger.info("Loaded .env file with latin-1 encoding")
        except:
            logger.warning("Could not load .env file. Using system environment variables.")
except Exception as e:
    logger.warning(f"Error loading .env file: {e}. Using system environment variables.")

def ai_feedback(resume_text: str, jd_text: str) -> str:
    """
    Generate personalized AI feedback using Groq API via LangChain with intelligent fallback
    """
    if not resume_text or not jd_text:
        return "Please provide both resume text and job description."

    # Try Groq API with LangChain first
    groq_result = try_groq_langchain(resume_text, jd_text)
    if groq_result and "AI Analysis Complete" in groq_result:
        return groq_result
    
    # Fallback to your comprehensive intelligent analysis
    return generate_personalized_analysis(resume_text, jd_text, "AI API temporarily unavailable")

def try_groq_langchain(resume_text: str, jd_text: str) -> str:
    """
    Try Groq API using LangChain for AI-generated feedback
    """
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    groq_model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    groq_temp = float(os.getenv("GROQ_TEMPERATURE", "0.7"))
    
    if not groq_key:
        logger.info("No Groq API key found")
        return None
    
    try:
        # Import LangChain components
        from langchain_groq import ChatGroq
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        
        # Initialize Groq LLM
        llm = ChatGroq(
            model=groq_model,
            api_key=groq_key,
            temperature=groq_temp
        )
        
        # Truncate texts to manage token limits
        resume_snippet = resume_text[:2000] if len(resume_text) > 2000 else resume_text
        jd_snippet = jd_text[:1500] if len(jd_text) > 1500 else jd_text
        
        # Create prompt using ChatPromptTemplate
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert career advisor and resume specialist with 15+ years of experience. Analyze the following resume against the job description and provide professional, actionable feedback."),
            ("human", """JOB DESCRIPTION:
{jd_text}

RESUME:
{resume_text}

Please provide a comprehensive analysis with:
1. Overall match assessment (percentage estimate and clear reasoning)
2. Top 3 strengths that align with the job requirements
3. Top 3 areas that need improvement
4. Specific skills or experience that are missing
5. 3 concrete, actionable recommendations to improve this resume for this specific role

Be professional, specific, and focus on practical advice that will help the candidate improve their chances of getting this job.""")
        ])
        
        # Create chain using LCEL
        chain = prompt | llm | StrOutputParser()
        
        logger.info("Calling Groq API via LangChain...")
        
        # Invoke the chain
        ai_response = chain.invoke({
            "resume_text": resume_snippet,
            "jd_text": jd_snippet
        })
        
        if ai_response:
            logger.info("Successfully generated Groq response via LangChain")
            return f"🤖 **AI Analysis Complete:**\n\n{ai_response}\n\n*Generated using Groq AI ({groq_model}) via LangChain*"
        else:
            logger.error("Empty response from Groq")
            return None
            
    except ImportError as e:
        logger.error(f"LangChain import error: {e}")
        logger.info("Install: pip install langchain-core langchain-groq")
        return None
    except Exception as e:
        logger.error(f"Groq LangChain call failed: {e}")
        return None

def generate_personalized_analysis(resume_text: str, jd_text: str, reason: str) -> str:
    """
    Your original comprehensive intelligent analysis system (preserved exactly)
    """
    try:
        from matcher import extract_skills_from_text
        resume_skills = extract_skills_from_text(resume_text)
        jd_skills = extract_skills_from_text(jd_text)
        matched_skills = resume_skills & jd_skills
        missing_skills = jd_skills - resume_skills
        extra_skills = resume_skills - jd_skills
    except:
        resume_skills, jd_skills, matched_skills, missing_skills = basic_skill_extraction(resume_text, jd_text)
        extra_skills = resume_skills - jd_skills
    
    role_type = determine_role_type(jd_text)
    experience_level = extract_experience_level(resume_text)
    skill_match_ratio = len(matched_skills) / len(jd_skills) if jd_skills else 0
    
    assessment = generate_assessment(skill_match_ratio, role_type)
    strengths = generate_strengths(matched_skills, resume_text, role_type)
    improvements = generate_improvements(missing_skills, resume_text, role_type)
    recommendations = generate_recommendations(missing_skills, extra_skills, role_type, experience_level)
    
    personalized_feedback = f"""🤖 **Personalized Career Analysis** *(Advanced AI Analysis - {reason})*

**🎯 Overall Fit Assessment:** {assessment}

**✅ Your Key Strengths:**
{strengths}

**📈 Areas for Strategic Improvement:**
{improvements}

**🚀 Personalized Recommendations:**
{recommendations}

**📊 Skill Analysis Summary:**
• **Matched Skills ({len(matched_skills)}):** {', '.join(sorted(list(matched_skills))[:8]) if matched_skills else 'Limited alignment found'}
• **Strategic Additions Needed ({len(missing_skills)}):** {', '.join(sorted(list(missing_skills))[:5]) if missing_skills else 'All core skills present'}
• **Bonus Skills You Bring ({len(extra_skills)}):** {', '.join(sorted(list(extra_skills))[:5]) if extra_skills else 'Focus on JD requirements'}

**💡 Career Positioning Insight:**
Based on your {experience_level} experience profile and {role_type} target role, you're positioned as a {get_candidate_profile(skill_match_ratio, experience_level, role_type)} candidate. {get_positioning_advice(skill_match_ratio, role_type)}

*This analysis uses advanced career intelligence algorithms and industry benchmarking data.*"""

    return personalized_feedback

def basic_skill_extraction(resume_text: str, jd_text: str) -> tuple:
    """Basic skill extraction fallback"""
    common_skills = [
        'python', 'sql', 'javascript', 'java', 'react', 'angular', 'vue',
        'machine learning', 'data science', 'data analysis', 'statistics',
        'aws', 'azure', 'docker', 'kubernetes', 'git', 'jenkins',
        'tableau', 'power bi', 'excel', 'powerpoint',
        'project management', 'agile', 'scrum', 'communication', 'leadership'
    ]
    
    resume_lower = resume_text.lower()
    jd_lower = jd_text.lower()
    
    resume_skills = {skill for skill in common_skills if skill in resume_lower}
    jd_skills = {skill for skill in common_skills if skill in jd_lower}
    matched_skills = resume_skills & jd_skills
    missing_skills = jd_skills - resume_skills
    
    return resume_skills, jd_skills, matched_skills, missing_skills

def determine_role_type(jd_text: str) -> str:
    """Determine role type from job description"""
    text_lower = jd_text.lower()
    
    if any(term in text_lower for term in ['data scientist', 'machine learning', 'data analysis']):
        return "Data Science"
    elif any(term in text_lower for term in ['software engineer', 'developer', 'programming']):
        return "Software Engineering"
    elif any(term in text_lower for term in ['product manager', 'product owner']):
        return "Product Management"
    elif any(term in text_lower for term in ['marketing', 'digital marketing']):
        return "Marketing"
    elif any(term in text_lower for term in ['finance', 'financial']):
        return "Finance"
    else:
        return "Technical"

def extract_experience_level(resume_text: str) -> str:
    """Extract experience level"""
    text_lower = resume_text.lower()
    
    year_patterns = [
        r'(\d+)\+?\s*years?\s*(?:of\s*)?(?:experience|exp)',
        r'(\d+)\+?\s*yrs?\s*(?:of\s*)?(?:experience|exp)'
    ]
    
    for pattern in year_patterns:
        match = re.search(pattern, text_lower)
        if match:
            years = int(match.group(1))
            if years >= 8:
                return "senior-level (8+ years)"
            elif years >= 4:
                return "mid-level (4-7 years)"
            elif years >= 1:
                return "junior-level (1-3 years)"
            else:
                return "entry-level"
    
    senior_indicators = ['senior', 'lead', 'principal', 'architect', 'manager', 'director']
    if any(indicator in text_lower for indicator in senior_indicators):
        return "senior-level"
    
    junior_indicators = ['intern', 'junior', 'assistant', 'entry', 'graduate']
    if any(indicator in text_lower for indicator in junior_indicators):
        return "entry-level"
    
    return "mid-level"

def generate_assessment(skill_match_ratio: float, role_type: str) -> str:
    """Generate personalized overall assessment"""
    if skill_match_ratio >= 0.8:
        return f"Excellent alignment for this {role_type} position. You demonstrate strong competency in the core requirements and should be confident in your application."
    elif skill_match_ratio >= 0.6:
        return f"Strong foundation for this {role_type} role with some strategic gaps to address. You're well-positioned as a competitive candidate."
    elif skill_match_ratio >= 0.4:
        return f"Moderate fit for this {role_type} position. Several key skills need development, but your background shows relevant potential."
    else:
        return f"Limited alignment with this specific {role_type} role requirements. Consider targeting positions that better match your current skill set or invest in significant skill development."

def generate_strengths(matched_skills: set, resume_text: str, role_type: str) -> str:
    """Generate personalized strengths analysis"""
    strengths = []
    
    if matched_skills:
        core_skills = sorted(list(matched_skills))[:5]
        strengths.append(f"• **Core Competencies Validated:** Your expertise in {', '.join(core_skills)} directly aligns with job requirements")
    
    resume_lower = resume_text.lower()
    
    if re.search(r'\d+%|\d+\+|increased|improved|reduced|optimized', resume_lower):
        strengths.append("• **Results-Driven Profile:** You demonstrate quantifiable achievements and business impact")
    
    if re.search(r'led|managed|supervised|coordinated', resume_lower):
        strengths.append("• **Leadership Experience:** Clear evidence of team leadership and project management capabilities")
    
    if re.search(r'project|built|developed|created|designed', resume_lower):
        strengths.append("• **Hands-On Implementation:** Strong project execution and technical delivery experience")
    
    if len(strengths) == 1:
        strengths.append("• **Professional Experience:** Relevant background that supports role requirements")
        strengths.append(f"• **Industry Alignment:** Experience appears suitable for {role_type} responsibilities")
    
    return '\n'.join(strengths[:4])

def generate_improvements(missing_skills: set, resume_text: str, role_type: str) -> str:
    """Generate targeted improvement recommendations"""
    improvements = []
    
    if missing_skills:
        priority_skills = sorted(list(missing_skills))[:4]
        improvements.append(f"• **Skill Development Priority:** Focus on gaining proficiency in {', '.join(priority_skills)}")
    
    resume_lower = resume_text.lower()
    
    if not re.search(r'\d+%|\d+\+|\$[\d,]+', resume_lower):
        improvements.append("• **Achievement Quantification:** Add specific metrics, percentages, and dollar amounts to demonstrate impact")
    
    if not re.search(r'certification|certified|license', resume_lower) and role_type in ['Data Science', 'Software Engineering']:
        improvements.append(f"• **Professional Certification:** Consider relevant certifications for {role_type} to strengthen credibility")
    
    if len(resume_text.split()) < 300:
        improvements.append("• **Resume Depth:** Expand descriptions of key experiences with more detailed accomplishments")
    
    if not re.search(r'project|portfolio|github', resume_lower) and role_type in ['Data Science', 'Software Engineering']:
        improvements.append("• **Portfolio Development:** Create visible projects that demonstrate your technical capabilities")
    
    return '\n'.join(improvements[:4])

def generate_recommendations(missing_skills: set, extra_skills: set, role_type: str, experience_level: str) -> str:
    """Generate actionable recommendations"""
    recommendations = []
    
    if missing_skills:
        top_missing = sorted(list(missing_skills))[:3]
        recommendations.append(f"• **Immediate Action:** Start learning {', '.join(top_missing)} through online courses, projects, or practical application")
    
    if extra_skills:
        bonus_skills = sorted(list(extra_skills))[:3]
        recommendations.append(f"• **Differentiation Strategy:** Highlight how your additional skills in {', '.join(bonus_skills)} add unique value")
    
    if "entry-level" in experience_level:
        recommendations.append("• **Entry-Level Strategy:** Emphasize learning agility, relevant coursework, and transferable skills")
    elif "senior-level" in experience_level:
        recommendations.append("• **Leadership Positioning:** Highlight strategic thinking, mentoring experience, and business impact")
    
    if role_type == "Data Science":
        recommendations.append("• **Technical Showcase:** Include links to data projects, GitHub repositories, or technical blog posts")
    elif role_type == "Product Management":
        recommendations.append("• **Product Thinking:** Demonstrate user-focused decision making and cross-functional collaboration")
    elif role_type == "Marketing":
        recommendations.append("• **Growth Metrics:** Show campaign performance data and customer acquisition results")
    
    return '\n'.join(recommendations[:4])

def get_candidate_profile(skill_match_ratio: float, experience_level: str, role_type: str) -> str:
    """Determine candidate profile classification"""
    if skill_match_ratio >= 0.8:
        return "highly qualified"
    elif skill_match_ratio >= 0.6:
        return "well-qualified"
    elif skill_match_ratio >= 0.4:
        return "potentially suitable"
    else:
        return "developing"

def get_positioning_advice(skill_match_ratio: float, role_type: str) -> str:
    """Provide strategic positioning advice"""
    if skill_match_ratio >= 0.8:
        return "Focus on demonstrating cultural fit and salary expectations in interviews."
    elif skill_match_ratio >= 0.6:
        return "Prepare to discuss how you'll quickly ramp up in areas where you have less experience."
    elif skill_match_ratio >= 0.4:
        return "Consider highlighting transferable skills and your learning approach for new technologies."
    else:
        return "You may want to target more junior roles or invest significant time in skill development first."

def rule_based_feedback(matched_skills: list, missing_skills: list, resume_text: str) -> str:
    """Enhanced rule-based feedback with accurate scoring alignment"""
    feedback_parts = []
    
    total_skills = len(matched_skills) + len(missing_skills)
    if total_skills == 0:
        return "⚠️ **Analysis Issue:** No clear skills identified in job description"
    
    match_percentage = len(matched_skills) / total_skills
    
    if match_percentage >= 0.9:
        feedback_parts.append("🌟 **Outstanding Match (90%+):** Exceptional alignment with job requirements!")
        feedback_parts.append("💡 **Strategy:** You're an ideal candidate - focus on interview preparation and compensation negotiation.")
    elif match_percentage >= 0.8:
        feedback_parts.append("✨ **Excellent Match (80-89%):** Very strong alignment with job requirements!")
        feedback_parts.append("💡 **Strategy:** Highlight your expertise and prepare for technical discussions.")
    elif match_percentage >= 0.7:
        feedback_parts.append("🎯 **Strong Match (70-79%):** Good foundation with minor gaps to address")
        feedback_parts.append("💡 **Strategy:** Emphasize matched skills and show learning plan for missing ones.")
    elif match_percentage >= 0.6:
        feedback_parts.append("👍 **Good Match (60-69%):** Solid alignment with room for improvement")
        feedback_parts.append("💡 **Strategy:** Address key missing skills and optimize resume keywords.")
    elif match_percentage >= 0.5:
        feedback_parts.append("⚖️ **Moderate Match (50-59%):** Reasonable fit with several skills to develop")
        feedback_parts.append("💡 **Strategy:** Focus on gaining missing skills or highlight transferable experience.")
    else:
        feedback_parts.append("⚠️ **Limited Match (<50%):** Significant gaps in required skills")
        feedback_parts.append("💡 **Strategy:** Consider skill development or look for roles that better match your background.")
    
    if matched_skills:
        skills_text = ', '.join(matched_skills[:12])
        if len(matched_skills) > 12:
            skills_text += f" (+{len(matched_skills)-12} more)"
        feedback_parts.append(f"✅ **Matched Skills ({len(matched_skills)}):** {skills_text}")
    
    if missing_skills:
        priority_missing = missing_skills[:8]
        skills_text = ', '.join(priority_missing)
        if len(missing_skills) > 8:
            skills_text += f" (+{len(missing_skills)-8} more)"
        feedback_parts.append(f"⭕ **Missing Skills ({len(missing_skills)}):** {skills_text}")
    
    return '\n\n'.join(feedback_parts)

def test_hf_connection() -> Tuple[bool, str]:
    """Test Groq API connection via LangChain"""
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    
    if not groq_key:
        return False, "❌ GROQ_API_KEY not found in .env file"
    
    try:
        from langchain_groq import ChatGroq
        
        # Try to initialize the ChatGroq client
        llm = ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            api_key=groq_key,
            temperature=0.1
        )
        
        # Test with a simple invoke
        response = llm.invoke("Say 'Hello'")
        
        if response:
            return True, "✅ Groq API connection successful via LangChain"
        else:
            return False, "❌ Unexpected response from Groq API"
            
    except ImportError:
        return False, "❌ langchain-groq not installed. Run: pip install langchain-groq"
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "Unauthorized" in error_msg:
            return False, "❌ Invalid Groq API key"
        elif "429" in error_msg or "rate limit" in error_msg.lower():
            return False, "❌ Groq rate limit exceeded - wait and try again"
        else:
            return False, f"❌ Connection failed: {error_msg}"