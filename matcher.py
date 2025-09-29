import spacy
from sentence_transformers import SentenceTransformer, util
import re
from collections import Counter
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load NLP models once
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

model = SentenceTransformer('all-MiniLM-L6-v2')

# Clean, focused skill dictionary - only actual skills, no descriptive phrases
SKILL_DICTIONARY = {
    # Programming Languages
    'python': ['python', 'python programming'],
    'java': ['java', 'java programming'],
    'javascript': ['javascript', 'js'],
    'typescript': ['typescript', 'ts'],
    'c++': ['c++', 'cpp'],
    'c#': ['c#', 'csharp'],
    'r': ['r programming', 'r language'],
    'go': ['golang'],
    'rust': ['rust'],
    'swift': ['swift'],
    'kotlin': ['kotlin'],
    'php': ['php'],
    'ruby': ['ruby'],
    'scala': ['scala'],
    'matlab': ['matlab'],
    'sas': ['sas'],
    
    # Databases
    'sql': ['sql', 'mysql', 'postgresql', 'sqlite'],
    'nosql': ['nosql', 'mongodb', 'cassandra'],
    'mongodb': ['mongodb', 'mongo'],
    'postgresql': ['postgresql', 'postgres'],
    'mysql': ['mysql'],
    'oracle': ['oracle'],
    'redis': ['redis'],
    'elasticsearch': ['elasticsearch'],
    
    # Web Technologies
    'html': ['html', 'html5'],
    'css': ['css', 'css3'],
    'react': ['react', 'reactjs'],
    'angular': ['angular', 'angularjs'],
    'vue': ['vue', 'vuejs'],
    'nodejs': ['node.js', 'nodejs'],
    'django': ['django'],
    'flask': ['flask'],
    'spring': ['spring', 'spring boot'],
    'express': ['express', 'expressjs'],
    'laravel': ['laravel'],
    'bootstrap': ['bootstrap'],
    'jquery': ['jquery'],
    
    # Cloud & DevOps
    'aws': ['aws', 'amazon web services'],
    'azure': ['azure', 'microsoft azure'],
    'gcp': ['gcp', 'google cloud'],
    'docker': ['docker'],
    'kubernetes': ['kubernetes', 'k8s'],
    'terraform': ['terraform'],
    'ansible': ['ansible'],
    'jenkins': ['jenkins'],
    'git': ['git', 'github', 'gitlab'],
    
    # Data Science & ML
    'machine learning': ['machine learning', 'ml'],
    'deep learning': ['deep learning'],
    'data science': ['data science'],
    'artificial intelligence': ['artificial intelligence', 'ai'],
    'tensorflow': ['tensorflow'],
    'pytorch': ['pytorch'],
    'keras': ['keras'],
    'scikit-learn': ['scikit-learn', 'sklearn'],
    'pandas': ['pandas'],
    'numpy': ['numpy'],
    'matplotlib': ['matplotlib'],
    'seaborn': ['seaborn'],
    'plotly': ['plotly'],
    'tableau': ['tableau'],
    'power bi': ['power bi', 'powerbi'],
    'excel': ['excel', 'microsoft excel'],
    'nlp': ['nlp', 'natural language processing'],
    'computer vision': ['computer vision'],
    'data visualization': ['data visualization', 'data viz'],
    'statistics': ['statistics', 'statistical analysis'],
    'data analysis': ['data analysis', 'data analytics'],
    'predictive modeling': ['predictive modeling'],
    'regression': ['regression', 'linear regression'],
    'classification': ['classification'],
    'clustering': ['clustering'],
    
    # Business & Soft Skills (only concrete ones)
    'project management': ['project management', 'pmp'],
    'agile': ['agile', 'scrum'],
    'kanban': ['kanban'],
    'leadership': ['leadership'],
    'communication': ['communication'],
    'teamwork': ['teamwork', 'collaboration'],
    'problem solving': ['problem solving'],
    'analytical thinking': ['analytical thinking'],
    'presentation': ['presentation'],
    'negotiation': ['negotiation'],
    
    # Tools & Software
    'jira': ['jira'],
    'confluence': ['confluence'],
    'slack': ['slack'],
    'photoshop': ['photoshop'],
    'illustrator': ['illustrator'],
    'figma': ['figma'],
    'sketch': ['sketch'],
    
    # Methodologies
    'devops': ['devops'],
    'ci/cd': ['ci/cd', 'continuous integration'],
    'tdd': ['tdd', 'test driven development'],
    'waterfall': ['waterfall'],
    
    # Domain Knowledge
    'finance': ['finance', 'financial'],
    'accounting': ['accounting'],
    'marketing': ['marketing'],
    'sales': ['sales'],
    'healthcare': ['healthcare'],
    'research': ['research'],
}

# Expanded blocklist - phrases that should NEVER be considered skills
BLOCKLIST_PHRASES = {
    # Descriptive phrases
    'strong programming skills', 'excellent communication skills', 'good knowledge',
    'solid understanding', 'proven experience', 'extensive experience',
    'strong background', 'deep knowledge', 'thorough understanding',
    'comprehensive knowledge', 'broad experience', 'strong foundation',
    'excellent skills', 'outstanding abilities', 'superior knowledge',
    
    # Generic business terms
    'business requirements', 'business needs', 'business goals',
    'team player', 'detail oriented', 'self motivated',
    'fast paced environment', 'dynamic environment',
    'collaborative environment', 'innovative environment',
    
    # Common job posting language
    'minimum requirements', 'preferred qualifications',
    'nice to have', 'bonus points', 'plus if you have',
    'we are looking for', 'ideal candidate', 'perfect fit',
    
    # Educational terms
    'bachelor degree', 'master degree', 'phd',
    'computer science degree', 'related field',
    'equivalent experience',
    
    # Time/experience references
    'years of experience', 'years experience',
    'months of experience', 'recent experience',
    'relevant experience', 'professional experience',
    
    # Generic descriptors
    'technical skills', 'soft skills', 'interpersonal skills',
    'communication skills', 'analytical skills', 'problem solving skills',
    'leadership skills', 'management skills'
}

def preprocess_text(text: str) -> str:
    """Clean and normalize text"""
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text.strip().lower())
    return text

def is_valid_skill(skill: str) -> bool:
    """
    Strict validation - only allow actual skills, no descriptive phrases
    """
    skill_lower = skill.lower().strip()
    
    # Length check
    if len(skill_lower) < 2 or len(skill_lower) > 25:
        return False
    
    # Check against blocklist
    if skill_lower in BLOCKLIST_PHRASES:
        return False
    
    # Block phrases containing common filler words
    filler_patterns = [
        r'\b(strong|excellent|good|solid|proven|extensive|deep|thorough|comprehensive)\b',
        r'\b(skills?|knowledge|experience|understanding|background|abilities?)\b',
        r'\b(years?|months?|minimum|maximum|preferred|required)\b',
        r'\b(degree|bachelor|master|phd|related field)\b'
    ]
    
    for pattern in filler_patterns:
        if re.search(pattern, skill_lower):
            return False
    
    # Must be in our skill dictionary to be valid
    return skill_lower in SKILL_DICTIONARY

def extract_skills_from_text(text: str) -> set:
    """
    Extract ONLY legitimate skills using strict dictionary matching
    """
    if not text:
        return set()
    
    text_lower = preprocess_text(text)
    found_skills = set()
    
    # Method 1: Direct dictionary matching (most reliable)
    for skill, variations in SKILL_DICTIONARY.items():
        for variation in variations:
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(variation.lower()) + r'\b'
            if re.search(pattern, text_lower):
                found_skills.add(skill)
                break
    
    # Method 2: Additional regex for common technical patterns
    tech_patterns = {
        'python': r'\bpython\b',
        'java': r'\bjava\b(?!\s*script)',  # Java but not JavaScript
        'javascript': r'\b(?:javascript|js)\b',
        'typescript': r'\b(?:typescript|ts)\b',
        'sql': r'\b(?:sql|mysql|postgresql)\b',
        'react': r'\b(?:react|reactjs)\b',
        'angular': r'\bangular\b',
        'vue': r'\b(?:vue|vuejs)\b',
        'aws': r'\b(?:aws|amazon web services)\b',
        'docker': r'\bdocker\b',
        'kubernetes': r'\b(?:kubernetes|k8s)\b',
    }
    
    for skill, pattern in tech_patterns.items():
        if re.search(pattern, text_lower, re.IGNORECASE):
            found_skills.add(skill)
    
    # Final validation - ensure all found skills are legitimate
    validated_skills = {skill for skill in found_skills if is_valid_skill(skill)}
    
    return validated_skills

def extract_skills_dynamic(resume_text: str, jd_text: str, threshold: float = 0.75) -> tuple:
    """
    Extract skills from JD and match with resume skills
    Only compares legitimate skills, no descriptive phrases
    """
    if not resume_text or not jd_text:
        return [], []
    
    logger.info("Starting clean skill extraction...")
    
    # Extract legitimate skills only
    jd_skills = extract_skills_from_text(jd_text)
    resume_skills = extract_skills_from_text(resume_text)
    
    logger.info(f"JD skills found: {sorted(list(jd_skills))}")
    logger.info(f"Resume skills found: {sorted(list(resume_skills))}")
    
    matched_skills = set()
    
    # Direct matching
    for jd_skill in jd_skills:
        if jd_skill in resume_skills:
            matched_skills.add(jd_skill)
    
    # Semantic matching for remaining skills
    remaining_jd_skills = list(jd_skills - matched_skills)
    remaining_resume_skills = list(resume_skills)
    
    if remaining_jd_skills and remaining_resume_skills:
        try:
            jd_embeddings = model.encode(remaining_jd_skills, convert_to_tensor=True)
            resume_embeddings = model.encode(remaining_resume_skills, convert_to_tensor=True)
            similarity_matrix = util.cos_sim(jd_embeddings, resume_embeddings)
            
            for i, jd_skill in enumerate(remaining_jd_skills):
                max_similarity = similarity_matrix[i].max().item()
                if max_similarity > threshold:
                    matched_skills.add(jd_skill)
                    best_match_idx = similarity_matrix[i].argmax().item()
                    logger.info(f"Semantic match: '{jd_skill}' <-> '{remaining_resume_skills[best_match_idx]}' ({max_similarity:.2f})")
        
        except Exception as e:
            logger.error(f"Error in semantic matching: {e}")
    
    missing_skills = jd_skills - matched_skills
    
    # Sort and return
    matched_list = sorted(list(matched_skills))
    missing_list = sorted(list(missing_skills))
    
    logger.info(f"Final results: {len(matched_list)} matched, {len(missing_list)} missing")
    logger.info(f"Matched: {matched_list}")
    logger.info(f"Missing: {missing_list}")
    
    return matched_list, missing_list

def compute_similarity(resume_text: str, jd_text: str) -> float:
    """
    Compute similarity score based primarily on skill matching with proper weighting
    """
    if not resume_text or not jd_text:
        return 0.0
    
    try:
        # Extract skills from both texts
        resume_skills = extract_skills_from_text(resume_text)
        jd_skills = extract_skills_from_text(jd_text)
        
        if not jd_skills:
            logger.warning("No skills found in job description")
            return 0.0
        
        if not resume_skills:
            logger.warning("No skills found in resume")
            return 0.0
        
        # Calculate skill-based similarity (primary factor - 80% weight)
        matched_skills = resume_skills & jd_skills
        skill_match_ratio = len(matched_skills) / len(jd_skills)
        
        logger.info(f"Skill matching: {len(matched_skills)}/{len(jd_skills)} = {skill_match_ratio:.2%}")
        
        # Apply progressive scoring curve for skill matching
        if skill_match_ratio >= 0.9:
            skill_score = 95  # Exceptional match
        elif skill_match_ratio >= 0.8:
            skill_score = 85  # Excellent match
        elif skill_match_ratio >= 0.7:
            skill_score = 75  # Very good match
        elif skill_match_ratio >= 0.6:
            skill_score = 65  # Good match
        elif skill_match_ratio >= 0.5:
            skill_score = 55  # Moderate match
        elif skill_match_ratio >= 0.3:
            skill_score = 40  # Fair match
        else:
            skill_score = skill_match_ratio * 100  # Below 30% - linear scaling
        
        # Calculate semantic similarity of skills (secondary factor - 20% weight)
        if matched_skills:
            resume_skill_text = ' '.join(resume_skills)
            jd_skill_text = ' '.join(jd_skills)
            
            try:
                embeddings = model.encode([resume_skill_text, jd_skill_text], convert_to_tensor=True)
                semantic_similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
                semantic_score = semantic_similarity * 100
            except Exception as e:
                logger.warning(f"Semantic similarity calculation failed: {e}")
                semantic_score = skill_match_ratio * 100
        else:
            semantic_score = 0
        
        # Combine scores with appropriate weights
        final_score = (skill_score * 0.8) + (semantic_score * 0.2)
        
        # Ensure score is within bounds
        final_score = max(0, min(100, final_score))
        
        logger.info(f"Scoring breakdown: skill_score={skill_score:.1f}, semantic_score={semantic_score:.1f}, final={final_score:.1f}")
        
        return round(final_score, 2)
        
    except Exception as e:
        logger.error(f"Error computing similarity: {e}")
        return 0.0

def get_skill_frequency_analysis(resume_text: str, jd_text: str) -> dict:
    """
    Analyze frequency of skills in JD to identify priorities
    """
    try:
        jd_skills = extract_skills_from_text(jd_text)
        jd_lower = jd_text.lower()
        
        frequency_counter = {}
        for skill in jd_skills:
            count = 0
            skill_variations = SKILL_DICTIONARY.get(skill, [skill])
            
            for variation in skill_variations:
                count += len(re.findall(r'\b' + re.escape(variation.lower()) + r'\b', jd_lower))
            
            if count > 0:
                frequency_counter[skill] = count
        
        return dict(Counter(frequency_counter).most_common(10))
        
    except Exception as e:
        logger.error(f"Error in skill frequency analysis: {e}")
        return {}