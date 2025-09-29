import pdfplumber
import docx2txt
import re
import logging
import io
import os
import tempfile
from typing import Optional, Dict, List

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_pdf(file_path) -> str:
    """
    Extract text from PDF file. Accepts path strings or file-like objects (Streamlit uploads).
    Returns cleaned text (preserving case).
    """
    try:
        # Read bytes from file-like or file path
        if hasattr(file_path, "read"):
            try:
                file_path.seek(0)
            except Exception:
                pass
            file_bytes = file_path.read()
        elif isinstance(file_path, (str, os.PathLike)):
            with open(file_path, "rb") as f:
                file_bytes = f.read()
        else:
            return "Error: Unsupported PDF file object."

        if not file_bytes:
            logger.warning("No bytes read from PDF input")
            return ""

        text = ""
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            page_count = len(pdf.pages)
            if page_count == 0:
                logger.warning("PDF file appears to be empty")
                return ""
            logger.info(f"Processing PDF with {page_count} pages")
            for page_num, page in enumerate(pdf.pages, 1):
                try:
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        if page_num > 1:
                            text += "\n--- PAGE BREAK ---\n"
                        text += page_text + "\n"
                    else:
                        # fallback to table extraction
                        tables = page.extract_tables()
                        if tables:
                            for table in tables:
                                for row in table:
                                    if row:
                                        text += " ".join([cell or "" for cell in row]) + "\n"
                        else:
                            logger.debug(f"No text found on page {page_num}")
                except Exception as e:
                    logger.warning(f"Error processing page {page_num}: {e}")
                    continue

        if not text.strip():
            logger.error("No text could be extracted from PDF")
            return ""

        logger.info(f"Successfully extracted {len(text)} characters from PDF")
        return clean_text(text)

    except Exception as e:
        logger.error(f"Error reading PDF file: {str(e)}")
        return f"Error processing PDF: {str(e)}"

def parse_docx(file_path) -> str:
    """
    Extract text from DOCX file. Accepts path strings or file-like objects (Streamlit uploads).
    Uses a temporary file when necessary to feed docx2txt.
    """
    try:
        # Read bytes
        if hasattr(file_path, "read"):
            try:
                file_path.seek(0)
            except Exception:
                pass
            file_bytes = file_path.read()
        elif isinstance(file_path, (str, os.PathLike)):
            with open(file_path, "rb") as f:
                file_bytes = f.read()
        else:
            return "Error: Unsupported DOCX file object."

        if not file_bytes:
            logger.warning("No bytes read from DOCX input")
            return ""

        # Write to temporary file because docx2txt expects a path
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
                tmp.write(file_bytes)
                tmp.flush()
                tmp_path = tmp.name

            text = docx2txt.process(tmp_path) or ""
            if not text.strip():
                logger.warning("Very little text extracted from DOCX file")
                return "Warning: Minimal text extracted from document. Please check if the file contains readable text."

            logger.info(f"Successfully extracted {len(text)} characters from DOCX")
            return clean_text(text)

        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

    except Exception as e:
        logger.error(f"Error reading DOCX file: {str(e)}")
        return f"Error processing DOCX: {str(e)}"

def clean_text(text: str) -> str:
    """
    Clean text while preserving important structure and information.
    Preserve case (do NOT force-lower) so AI feedback & readable previews are richer.
    """
    if not text:
        return ""

    # Normalize line endings and trim excessive blank lines
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Collapse excessive spaces/tabs but keep single spaces
    text = re.sub(r'[ \t]{2,}', ' ', text)

    # Remove common control characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', text)

    # Fix bullet spacing
    text = re.sub(r'•\s*', '• ', text)
    text = re.sub(r'◦\s*', '◦ ', text)

    # Fix email addresses split
    text = re.sub(r'(\w+)\s*@\s*(\w+)', r'\1@\2', text)

    # Merge broken phone number patterns e.g., 123 - 456 - 7890
    text = re.sub(r'(\d{3})\s*-\s*(\d{3})\s*-\s*(\d{4})', r'\1-\2-\3', text)

    # Remove lines that are almost certainly page numbers / headers
    lines = [ln.strip() for ln in text.split('\n')]
    cleaned_lines = []
    for line in lines:
        if not line:
            continue
        if re.match(r'^\d+$', line):  # standalone page number
            continue
        if re.match(r'^page \d+', line, re.IGNORECASE):
            continue
        # Keep other lines (we do not drop short lines aggressively)
        cleaned_lines.append(line)

    text = '\n'.join(cleaned_lines)
    text = re.sub(r'\n\s*\n', '\n\n', text).strip()
    return text

def validate_resume_content(text: str) -> Dict[str, any]:
    """
    Validate that the extracted text contains meaningful resume content
    """
    validation = {
        'is_valid': True,
        'warnings': [],
        'suggestions': [],
        'stats': {
            'character_count': len(text),
            'word_count': len(text.split()),
            'line_count': len(text.split('\n'))
        }
    }

    text_lower = (text or "").lower()

    if len(text or "") < 100:
        validation['is_valid'] = False
        validation['warnings'].append("Text is too short to be a meaningful resume")

    resume_indicators = [
        'experience', 'education', 'skills', 'work', 'project', 'degree',
        'university', 'college', 'job', 'position', 'company', 'email'
    ]

    found_indicators = [i for i in resume_indicators if i in text_lower]
    if len(found_indicators) < 3:
        validation['warnings'].append("Text doesn't appear to contain typical resume content")
        validation['suggestions'].append("Ensure the uploaded file is a resume with standard sections")

    # Contact info checks
    email_found = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    phone_found = re.search(r'(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', text)

    if not email_found:
        validation['suggestions'].append("Consider adding an email address to your resume")
    if not phone_found:
        validation['suggestions'].append("Consider adding a phone number to your resume")

    # Section check
    sections = ['education', 'experience', 'skills', 'projects']
    found_sections = [s for s in sections if s in text_lower]
    if len(found_sections) < 2:
        validation['suggestions'].append("Consider organizing your resume with clear sections (Education, Experience, Skills, Projects)")

    return validation

def extract_resume_sections(text: str) -> Dict[str, str]:
    """
    Identify and extract sections (best-effort). Works on cleaned text (case preserved).
    """
    sections = {}
    text_lines = text.split('\n')

    section_patterns = {
        'contact': r'^(contact|personal information)',
        'summary': r'^(summary|profile|objective|about)',
        'experience': r'^(experience|work experience|employment|professional experience)',
        'education': r'^(education|academic background|qualifications)',
        'skills': r'^(skills|technical skills|competencies|expertise)',
        'projects': r'^(projects|key projects|relevant projects)',
        'certifications': r'^(certifications|certificates|licenses)',
        'achievements': r'^(achievements|accomplishments|awards)'
    }

    current_section = 'other'
    current_content = []

    for line in text_lines:
        line = line.strip()
        if not line:
            continue

        section_found = None
        for section_name, pattern in section_patterns.items():
            if re.match(pattern, line, re.IGNORECASE):
                # Save previous
                if current_content:
                    sections[current_section] = '\n'.join(current_content)
                section_found = section_name
                current_section = section_name
                current_content = []
                break

        if not section_found:
            current_content.append(line)

    if current_content:
        sections[current_section] = '\n'.join(current_content)

    return sections

def parse_resume(file_path, file_type: str = None) -> Dict[str, any]:
    """
    Main wrapper used by the app: determines file type and calls parser.
    """
    result = {
        'success': False,
        'text': '',
        'validation': {},
        'sections': {},
        'error': None
    }

    try:
        # Determine file_type if not provided
        if hasattr(file_path, 'type'):
            file_type = file_path.type
        elif hasattr(file_path, 'name'):
            if file_path.name.lower().endswith('.pdf'):
                file_type = 'application/pdf'
            elif file_path.name.lower().endswith('.docx'):
                file_type = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'

        # Parse accordingly
        if file_type == "application/pdf":
            text = parse_pdf(file_path)
        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            text = parse_docx(file_path)
        else:
            result['error'] = f"Unsupported file type: {file_type}"
            return result

        if isinstance(text, str) and text.startswith("Error"):
            result['error'] = text
            return result

        validation = validate_resume_content(text)
        sections = extract_resume_sections(text)

        result.update({
            'success': True,
            'text': text,
            'validation': validation,
            'sections': sections
        })
        return result

    except Exception as e:
        result['error'] = f"Unexpected error parsing resume: {str(e)}"
        return result

# Backward compatibility
def parse_pdf_simple(file_path) -> str:
    return parse_pdf(file_path)

def parse_docx_simple(file_path) -> str:
    return parse_docx(file_path)
