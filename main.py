from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import io
import re
import json
import tempfile
import shutil
from dotenv import load_dotenv
import google.generativeai as genai
import fitz  # PyMuPDF
import docx
from PIL import Image
from typing import Optional, List, Dict, Any
import psycopg2
from psycopg2.extras import RealDictCursor
from pydantic import BaseModel
import uvicorn
import requests
import json
from fastapi import FastAPI, HTTPException


# Load environment variables
load_dotenv()
# Configure Google Gemini API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set")

# PostgreSQL configuration
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "jobmatcher")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

# Configure Gemini model
genai.configure(api_key=GOOGLE_API_KEY)

# Configure model
generation_config = {
    "temperature": 0.2,
    "top_p": 0.8,
    "top_k": 40,
    "max_output_tokens": 2048,
}

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    generation_config=generation_config
)

# Create FastAPI app
app = FastAPI(title="Job Description Analyzer API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this in production to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response validation
class JobDescriptionCreate(BaseModel):
    title: str
    company: str
    location: str
    description: str
    required_skills: List[str]
    preferred_skills: List[str]
    experience_required: str
    education_required: str
    job_type: str
    salary_range: Optional[str] = None
    benefits: Optional[List[str]] = None
    application_url: Optional[str] = None
    contact_email: Optional[str] = None
    date_posted: Optional[str] = None
    
class JobDescriptionResponse(JobDescriptionCreate):
    id: int

class SkillRatingRequest(BaseModel):
    job_id: int
    required_skills: Dict[str, int]
    preferred_skills: Dict[str, int]

# Database connection function
def get_db_connection():
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            cursor_factory=RealDictCursor
        )
        return conn
    except Exception as e:
        print(f"Database connection error: {e}")
        raise HTTPException(status_code=500, detail=f"Database connection error: {str(e)}")

def get_skdb_connection():
    conn = psycopg2.connect(
        host="localhost",
        database="tres",
        user="postgres",
        password="Temp1234"
    )
    conn.autocommit = True
    return conn

# Initialize database tables

def create_tables():
    conn = get_skdb_connection()
    cur = conn.cursor()
    try:
        # Create the table directly in the tres database (no schema needed)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS job_skill_ratings (
            id SERIAL PRIMARY KEY,
            job_id INTEGER NOT NULL,
            skill_name VARCHAR(255) NOT NULL,
            rating INTEGER NOT NULL,
            is_required BOOLEAN NOT NULL
        )
        """)
        conn.commit()  # Commit the table creation
    except Exception as e:
        print(f"Error creating table: {e}")
        conn.rollback()  # Rollback on error
    finally:
        cur.close()
        conn.close()

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create job_descriptions table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS job_descriptions (
        id SERIAL PRIMARY KEY,
        title VARCHAR(255) NOT NULL,
        company VARCHAR(255) NOT NULL,
        location VARCHAR(255) NOT NULL,
        description TEXT NOT NULL,
        experience_required VARCHAR(255),
        education_required TEXT,
        job_type VARCHAR(100),
        salary_range VARCHAR(255),
        application_url TEXT,
        contact_email VARCHAR(255),
        date_posted VARCHAR(100),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # Create skills tables
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS job_skills (
        id SERIAL PRIMARY KEY,
        job_id INTEGER REFERENCES job_descriptions(id) ON DELETE CASCADE,
        skill VARCHAR(255) NOT NULL,
        is_required BOOLEAN NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # Create benefits table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS job_benefits (
        id SERIAL PRIMARY KEY,
        job_id INTEGER REFERENCES job_descriptions(id) ON DELETE CASCADE,
        benefit TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    conn.commit()
    cursor.close()
    conn.close()

class DocumentProcessor:
    def process_document(self, file_path):
        """Process a document file and extract job description information"""
        # Determine file type
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.pdf':
            return self._process_pdf(file_path)
        elif file_extension in ['.docx', '.doc']:
            return self._process_word(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    
    def _process_pdf(self, file_path):
        """Process PDF file"""
        # Open the PDF
        pdf_document = fitz.open(file_path)
        
        # Extract text from all pages
        full_text = ""
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            full_text += page.get_text()
        
        # Create an image of the first page for visual analysis
        first_page = pdf_document[0]
        pix = first_page.get_pixmap()
        img_bytes = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_bytes))
        
        # Use Gemini to extract information from both text and image
        return self._extract_with_gemini(full_text, img)
    
    def _process_word(self, file_path):
        """Process Word document"""
        if file_path.endswith('.docx'):
            # Extract text from Word document
            doc = docx.Document(file_path)
            full_text = "\n".join([para.text for para in doc.paragraphs])
            
            # Use Gemini to extract information
            return self._extract_with_gemini(full_text)
        else:
            # For .doc files - would need additional library
            raise NotImplementedError("DOC file processing not implemented")
    
    def _extract_with_gemini(self, text, image=None):
        """Extract job description information using Google Gemini 2.0"""
        # Create prompt for Gemini
        prompt = """
        Extract the following details from the job description document:
        
        1. Job title
        2. Company name
        3. Job location
        4. Full job description
        5. Required skills (technical and soft skills that are explicitly mentioned as required)
        6. Preferred/Good to have skills (skills that are mentioned as preferred, good to have, or a plus)
        7. Years of experience required
        8. Education requirements
        9. Job type (full-time, part-time, contract, etc.)
        10. Salary range (if mentioned)
        11. Benefits (if mentioned)
        12. Application URL (if mentioned)
        13. Contact email (if mentioned)
        14. Date posted (if mentioned)
        
        Format the response as a JSON object with these fields:
        {
            "title": "",
            "company": "",
            "location": "",
            "description": "",
            "required_skills": [],
            "preferred_skills": [],
            "experience_required": "",
            "education_required": "",
            "job_type": "",
            "salary_range": "",
            "benefits": [],
            "application_url": "",
            "contact_email": "",
            "date_posted": ""
        }
        
        If a field is missing from the document, return an empty string or empty array as appropriate.
        """
        
        try:
            # If we have both text and image
            if image:
                response = model.generate_content([prompt, text, image])
            else:
                response = model.generate_content([prompt, text])
            
            # Get the response text
            response_text = response.text
            
            # Extract JSON from response (handle potential extra text)
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > 0:
                json_str = response_text[json_start:json_end]
                try:
                    extracted_info = json.loads(json_str)
                    return extracted_info
                except json.JSONDecodeError:
                    # Try to clean up the JSON
                    cleaned_json = self._clean_json_string(json_str)
                    extracted_info = json.loads(cleaned_json)
                    return extracted_info
            else:
                # Fallback to basic extraction
                return self._extract_basic_info(text)
                
        except Exception as e:
            print(f"Error with Gemini extraction: {e}")
            # Fallback to basic extraction
            return self._extract_basic_info(text)
    
    def _clean_json_string(self, json_str):
        """Clean up common JSON formatting issues"""
        # Replace single quotes with double quotes
        json_str = json_str.replace("'", '"')
        
        # Fix trailing commas
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        
        return json_str
    
    def _extract_basic_info(self, text):
        """Fallback method: Extract basic job information using regex patterns"""
        # Initialize job info dictionary
        job_info = {
            "title": "",
            "company": "",
            "location": "",
            "description": text[:1000] if len(text) > 1000 else text,  # Truncate long descriptions
            "required_skills": [],
            "preferred_skills": [],
            "experience_required": "",
            "education_required": "",
            "job_type": "",
            "salary_range": "",
            "benefits": [],
            "application_url": "",
            "contact_email": "",
            "date_posted": ""
        }
        
        # Extract job title (often at the beginning or in capitalized text)
        title_patterns = [
            r'^(.+?)\n',  # First line of the document
            r'job title[:\s]+(.+?)(?:\n|$)',  # Explicit job title
            r'position[:\s]+(.+?)(?:\n|$)',  # Position statement
        ]
        
        for pattern in title_patterns:
            title_match = re.search(pattern, text, re.IGNORECASE)
            if title_match:
                job_info["title"] = title_match.group(1).strip()
                break
        
        # Extract company name
        company_patterns = [
            r'(?:company|organization|employer)[:\s]+(.+?)(?:\n|$)',
            r'(?:at|with|for)\s+([A-Z][A-Za-z0-9\s&,.]+?)(?:\n|$)'
        ]
        
        for pattern in company_patterns:
            company_match = re.search(pattern, text, re.IGNORECASE)
            if company_match:
                job_info["company"] = company_match.group(1).strip()
                break
        
        # Extract location
        location_patterns = [
            r'(?:location|place|city)[:\s]+(.+?)(?:\n|$)',
            r'(?:in|at)\s+([A-Z][A-Za-z\s,]+(?:,\s*[A-Z]{2})?)(?:\n|$)'
        ]
        
        for pattern in location_patterns:
            location_match = re.search(pattern, text, re.IGNORECASE)
            if location_match:
                job_info["location"] = location_match.group(1).strip()
                break
        
        # Extract email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        if emails:
            job_info["contact_email"] = emails[0]
        
        # Extract URLs
        url_pattern = r'https?://[^\s<>"]+|www\.[^\s<>"]+|(?:apply|application)[^\n]*(?:at|@)\s*([^\s<>"]+)'
        urls = re.findall(url_pattern, text)
        if urls:
            job_info["application_url"] = urls[0]
        
        return job_info

# Function to save job description to PostgreSQL
def save_job_description(job_data):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Insert job description
        cursor.execute("""
        INSERT INTO job_descriptions (
            title, company, location, description, experience_required,
            education_required, job_type, salary_range, application_url,
            contact_email, date_posted
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
        """, (
            job_data["title"],
            job_data["company"],
            job_data["location"],
            job_data["description"],
            job_data["experience_required"],
            job_data["education_required"],
            job_data["job_type"],
            job_data["salary_range"],
            job_data["application_url"],
            job_data["contact_email"],
            job_data["date_posted"]
        ))
        
        job_id = cursor.fetchone()["id"]
        
        # Insert required skills
        for skill in job_data["required_skills"]:
            cursor.execute("""
            INSERT INTO job_skills (job_id, skill, is_required)
            VALUES (%s, %s, %s)
            """, (job_id, skill, True))
        
        # Insert preferred skills
        for skill in job_data["preferred_skills"]:
            cursor.execute("""
            INSERT INTO job_skills (job_id, skill, is_required)
            VALUES (%s, %s, %s)
            """, (job_id, skill, False))
        
        # Insert benefits
        if job_data["benefits"]:
            for benefit in job_data["benefits"]:
                cursor.execute("""
                INSERT INTO job_benefits (job_id, benefit)
                VALUES (%s, %s)
                """, (job_id, benefit))
        
        conn.commit()
        return job_id
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cursor.close()
        conn.close()
def update_or_create_job_description(base_url, job_data, job_id=None):
    """
    Updates an existing job description or creates a new one.
    
    Args:
        base_url (str): The base URL of the API
        job_data (dict): The job description data
        job_id (int, optional): The ID of the job to update. If None, creates a new job.
        
    Returns:
        dict: The response from the API
    """
    headers = {
        'Content-Type': 'application/json'
    }
    
    # If job_id is provided, update existing job
    if job_id:
        url = f"{base_url}/api/job-descriptions/{job_id}"
        response = requests.put(url, headers=headers, data=json.dumps(job_data))
    else:
        # If no job_id, create a new job description
        url = f"{base_url}/api/job-descriptions"
        # This endpoint isn't in the code yet, so we'll need to create it
        response = requests.post(url, headers=headers, data=json.dumps(job_data))
    
    if response.status_code in [200, 201]:
        return response.json()
    else:
        raise Exception(f"API request failed with status {response.status_code}: {response.text}")

# API endpoints (continued)
@app.post("/api/job-descriptions/analyze", response_model=JobDescriptionResponse)
async def analyze_job_description(file: UploadFile = File(...)):
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            # Copy the uploaded file to the temporary file
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name
        
        # Process the document
        processor = DocumentProcessor()
        extracted_info = processor.process_document(temp_file_path)
        
        # Save to PostgreSQL
        job_id = save_job_description(extracted_info)
        
        # Clean up the temporary file
        os.unlink(temp_file_path)
        
        # Return the extracted information with the database ID
        response_data = {**extracted_info, "id": job_id}
        return response_data
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/job-descriptions/{job_id}", response_model=JobDescriptionResponse)
async def get_job_description(job_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Get job description
        cursor.execute("""
        SELECT * FROM job_descriptions WHERE id = %s
        """, (job_id,))
        job = cursor.fetchone()
        
        if not job:
            raise HTTPException(status_code=404, detail="Job description not found")
        
        # Get required skills
        cursor.execute("""
        SELECT skill FROM job_skills WHERE job_id = %s AND is_required = TRUE
        """, (job_id,))
        required_skills = [row["skill"] for row in cursor.fetchall()]
        
        # Get preferred skills
        cursor.execute("""
        SELECT skill FROM job_skills WHERE job_id = %s AND is_required = FALSE
        """, (job_id,))
        preferred_skills = [row["skill"] for row in cursor.fetchall()]
        
        # Get benefits
        cursor.execute("""
        SELECT benefit FROM job_benefits WHERE job_id = %s
        """, (job_id,))
        benefits = [row["benefit"] for row in cursor.fetchall()]
        
        # Combine data
        job_data = dict(job)
        job_data["required_skills"] = required_skills
        job_data["preferred_skills"] = preferred_skills
        job_data["benefits"] = benefits
        
        return job_data
    
    finally:
        cursor.close()
        conn.close()

@app.get("/api/job-descriptions", response_model=List[JobDescriptionResponse])
async def list_job_descriptions(skip: int = 0, limit: int = 100):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Get all job descriptions with pagination
        cursor.execute("""
        SELECT * FROM job_descriptions ORDER BY id DESC LIMIT %s OFFSET %s
        """, (limit, skip))
        jobs = cursor.fetchall()
        
        # Get all skills
        cursor.execute("""
        SELECT job_id, skill, is_required FROM job_skills
        WHERE job_id IN (SELECT id FROM job_descriptions ORDER BY id DESC LIMIT %s OFFSET %s)
        """, (limit, skip))
        skills_rows = cursor.fetchall()
        
        # Get all benefits
        cursor.execute("""
        SELECT job_id, benefit FROM job_benefits
        WHERE job_id IN (SELECT id FROM job_descriptions ORDER BY id DESC LIMIT %s OFFSET %s)
        """, (limit, skip))
        benefit_rows = cursor.fetchall()
        
        # Organize skills and benefits by job_id
        skills_by_job = {}
        benefits_by_job = {}
        
        for row in skills_rows:
            job_id = row["job_id"]
            skill = row["skill"]
            is_required = row["is_required"]
            
            if job_id not in skills_by_job:
                skills_by_job[job_id] = {"required": [], "preferred": []}
            
            if is_required:
                skills_by_job[job_id]["required"].append(skill)
            else:
                skills_by_job[job_id]["preferred"].append(skill)
        
        for row in benefit_rows:
            job_id = row["job_id"]
            benefit = row["benefit"]
            
            if job_id not in benefits_by_job:
                benefits_by_job[job_id] = []
            
            benefits_by_job[job_id].append(benefit)
        
        # Combine data
        result = []
        for job in jobs:
            job_data = dict(job)
            job_id = job["id"]
            
            job_data["required_skills"] = skills_by_job.get(job_id, {"required": []})["required"]
            job_data["preferred_skills"] = skills_by_job.get(job_id, {"preferred": []})["preferred"]
            job_data["benefits"] = benefits_by_job.get(job_id, [])
            
            result.append(job_data)
        
        return result
    
    finally:
        cursor.close()
        conn.close()

@app.put("/api/job-descriptions/{job_id}", response_model=JobDescriptionResponse)
async def update_job_description(job_id: int, job_data: JobDescriptionCreate):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Start transaction
        conn.autocommit = False
        
        # Check if job exists
        cursor.execute("SELECT id FROM job_descriptions WHERE id = %s", (job_id,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Job description not found")
        
        # Update job description
        cursor.execute("""
        UPDATE job_descriptions SET
            title = %s,
            company = %s,
            location = %s,
            description = %s,
            experience_required = %s,
            education_required = %s,
            job_type = %s,
            salary_range = %s,
            application_url = %s,
            contact_email = %s,
            date_posted = %s
        WHERE id = %s
        """, (
            job_data.title,
            job_data.company,
            job_data.location,
            job_data.description,
            job_data.experience_required,
            job_data.education_required,
            job_data.job_type,
            job_data.salary_range,
            job_data.application_url,
            job_data.contact_email,
            job_data.date_posted,
            job_id
        ))
        
        # Delete existing skills
        cursor.execute("DELETE FROM job_skills WHERE job_id = %s", (job_id,))
        
        # Insert new required skills
        for skill in job_data.required_skills:
            cursor.execute("""
            INSERT INTO job_skills (job_id, skill, is_required)
            VALUES (%s, %s, %s)
            """, (job_id, skill, True))
        
        # Insert new preferred skills
        for skill in job_data.preferred_skills:
            cursor.execute("""
            INSERT INTO job_skills (job_id, skill, is_required)
            VALUES (%s, %s, %s)
            """, (job_id, skill, False))
        
        # Delete existing benefits
        cursor.execute("DELETE FROM job_benefits WHERE job_id = %s", (job_id,))
        
        # Insert new benefits
        if job_data.benefits:
            for benefit in job_data.benefits:
                cursor.execute("""
                INSERT INTO job_benefits (job_id, benefit)
                VALUES (%s, %s)
                """, (job_id, benefit))
        
        # Commit transaction
        conn.commit()
        
        # Return updated job
        return {**job_data.dict(), "id": job_id}
    
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        conn.autocommit = True
        cursor.close()
        conn.close()

@app.post("/api/job-descriptions", response_model=JobDescriptionResponse)
async def create_job_description(job_data: JobDescriptionCreate):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Insert job description
        cursor.execute("""
        INSERT INTO job_descriptions (
            title, company, location, description, experience_required,
            education_required, job_type, salary_range, application_url,
            contact_email, date_posted
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
        """, (
            job_data.title,
            job_data.company,
            job_data.location,
            job_data.description,
            job_data.experience_required,
            job_data.education_required,
            job_data.job_type,
            job_data.salary_range,
            job_data.application_url,
            job_data.contact_email,
            job_data.date_posted
        ))
        
        job_id = cursor.fetchone()["id"]
        
        # Insert required skills
        for skill in job_data.required_skills:
            cursor.execute("""
            INSERT INTO job_skills (job_id, skill, is_required)
            VALUES (%s, %s, %s)
            """, (job_id, skill, True))
        
        # Insert preferred skills
        for skill in job_data.preferred_skills:
            cursor.execute("""
            INSERT INTO job_skills (job_id, skill, is_required)
            VALUES (%s, %s, %s)
            """, (job_id, skill, False))
        
        # Insert benefits
        if job_data.benefits:
            for benefit in job_data.benefits:
                cursor.execute("""
                INSERT INTO job_benefits (job_id, benefit)
                VALUES (%s, %s)
                """, (job_id, benefit))
        
        conn.commit()
        
        # Return created job
        return {**job_data.dict(), "id": job_id}
    
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        cursor.close()
        conn.close()

@app.get("/api/job-descriptions/search", response_model=List[JobDescriptionResponse])
async def search_job_descriptions(
    query: Optional[str] = None,
    company: Optional[str] = None,
    location: Optional[str] = None,
    skill: Optional[str] = None,
    job_type: Optional[str] = None,
    skip: int = 0,
    limit: int = 100
):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Build query
        sql_query = """
        SELECT DISTINCT jd.* FROM job_descriptions jd
        """
        
        # Add join if searching by skill
        if skill:
            sql_query += " LEFT JOIN job_skills js ON jd.id = js.job_id"
        
        # Add WHERE clauses
        conditions = []
        params = []
        
        if query:
            conditions.append("(jd.title ILIKE %s OR jd.description ILIKE %s)")
            params.extend([f"%{query}%", f"%{query}%"])
        
        if company:
            conditions.append("jd.company ILIKE %s")
            params.append(f"%{company}%")
        
        if location:
            conditions.append("jd.location ILIKE %s")
            params.append(f"%{location}%")
        
        if skill:
            conditions.append("js.skill ILIKE %s")
            params.append(f"%{skill}%")
        
        if job_type:
            conditions.append("jd.job_type ILIKE %s")
            params.append(f"%{job_type}%")
        
        if conditions:
            sql_query += " WHERE " + " AND ".join(conditions)
        
        # Add pagination
        sql_query += " ORDER BY jd.id DESC LIMIT %s OFFSET %s"
        params.extend([limit, skip])
        
        # Execute query
        cursor.execute(sql_query, params)
        jobs = cursor.fetchall()
        job_ids = [job["id"] for job in jobs]
        
        if not job_ids:
            return []
        
        # Get all skills for found jobs
        cursor.execute("""
        SELECT job_id, skill, is_required FROM job_skills
        WHERE job_id = ANY(%s)
        """, (job_ids,))
        skills_rows = cursor.fetchall()
        
        # Get all benefits for found jobs
        cursor.execute("""
        SELECT job_id, benefit FROM job_benefits
        WHERE job_id = ANY(%s)
        """, (job_ids,))
        benefit_rows = cursor.fetchall()
        
        # Organize skills and benefits by job_id
        skills_by_job = {}
        benefits_by_job = {}
        
        for row in skills_rows:
            job_id = row["job_id"]
            skill = row["skill"]
            is_required = row["is_required"]
            
            if job_id not in skills_by_job:
                skills_by_job[job_id] = {"required": [], "preferred": []}
            
            if is_required:
                skills_by_job[job_id]["required"].append(skill)
            else:
                skills_by_job[job_id]["preferred"].append(skill)
        
        for row in benefit_rows:
            job_id = row["job_id"]
            benefit = row["benefit"]
            
            if job_id not in benefits_by_job:
                benefits_by_job[job_id] = []
            
            benefits_by_job[job_id].append(benefit)
        
        # Combine data
        result = []
        for job in jobs:
            job_data = dict(job)
            job_id = job["id"]
            
            job_data["required_skills"] = skills_by_job.get(job_id, {"required": []})["required"]
            job_data["preferred_skills"] = skills_by_job.get(job_id, {"preferred": []})["preferred"]
            job_data["benefits"] = benefits_by_job.get(job_id, [])
            
            result.append(job_data)
        
        return result
    
    finally:
        cursor.close()
        conn.close()

@app.get("/api/skills")
async def get_all_skills():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Get all unique skills
        cursor.execute("""
        SELECT DISTINCT skill, is_required FROM job_skills
        ORDER BY skill
        """)
        skills = cursor.fetchall()
        
        # Organize by required/preferred
        required_skills = []
        preferred_skills = []
        
        for skill in skills:
            if skill["is_required"]:
                required_skills.append(skill["skill"])
            else:
                preferred_skills.append(skill["skill"])
        
        return {
            "required_skills": list(set(required_skills)),
            "preferred_skills": list(set(preferred_skills)),
            "all_skills": list(set(required_skills + preferred_skills))
        }
    
    finally:
        cursor.close()
        conn.close()

@app.get("/api/stats")
async def get_job_stats():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Count of job descriptions
        cursor.execute("SELECT COUNT(*) as job_count FROM job_descriptions")
        job_count = cursor.fetchone()["job_count"]
        
        # Count of companies
        cursor.execute("SELECT COUNT(DISTINCT company) as company_count FROM job_descriptions")
        company_count = cursor.fetchone()["company_count"]
        
        # Count of locations
        cursor.execute("SELECT COUNT(DISTINCT location) as location_count FROM job_descriptions")
        location_count = cursor.fetchone()["location_count"]
        
        # Top 10 required skills
        cursor.execute("""
        SELECT skill, COUNT(*) as count
        FROM job_skills
        WHERE is_required = TRUE
        GROUP BY skill
        ORDER BY count DESC
        LIMIT 10
        """)
        top_required_skills = [{"skill": row["skill"], "count": row["count"]} for row in cursor.fetchall()]
        
        # Top 10 preferred skills
        cursor.execute("""
        SELECT skill, COUNT(*) as count
        FROM job_skills
        WHERE is_required = FALSE
        GROUP BY skill
        ORDER BY count DESC
        LIMIT 10
        """)
        top_preferred_skills = [{"skill": row["skill"], "count": row["count"]} for row in cursor.fetchall()]
        
        # Job types distribution
        cursor.execute("""
        SELECT job_type, COUNT(*) as count
        FROM job_descriptions
        WHERE job_type != ''
        GROUP BY job_type
        ORDER BY count DESC
        """)
        job_types = [{"type": row["job_type"], "count": row["count"]} for row in cursor.fetchall()]
        
        # Recent additions
        cursor.execute("""
        SELECT id, title, company, location, created_at
        FROM job_descriptions
        ORDER BY created_at DESC
        LIMIT 5
        """)
        recent_jobs = [dict(row) for row in cursor.fetchall()]
        
        return {
            "summary": {
                "job_count": job_count,
                "company_count": company_count,
                "location_count": location_count
            },
            "top_required_skills": top_required_skills,
            "top_preferred_skills": top_preferred_skills,
            "job_types": job_types,
            "recent_jobs": recent_jobs
        }
    
    finally:
        cursor.close()
        conn.close()

@app.get("/api/job-types")
async def get_job_types():
    """Get list of all job types in the database"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
        SELECT DISTINCT job_type, COUNT(*) as job_count
        FROM job_descriptions
        WHERE job_type != ''
        GROUP BY job_type
        ORDER BY job_count DESC
        """)
        job_types = [{"type": row["job_type"], "job_count": row["job_count"]} for row in cursor.fetchall()]
        return job_types
    
    finally:
        cursor.close()
        conn.close()

@app.get("/")
async def root():
    return {"message": "Job Description Analyzer API is running"}

@app.get("/api/health")
async def health_check():
    """API health check endpoint"""
    # Check database connection
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.close()
        conn.close()
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"
    
    return {
        "status": "ok",
        "database": db_status,
        "version": "1.0.0"
    }

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    try:
        init_db()
        create_tables()
        print("Database initialized successfully")
    except Exception as e:
        print(f"Error initializing database: {e}")

@app.post("/api/job-skills/ratings")
async def save_skill_ratings(request: SkillRatingRequest):
    conn = None
    try:
        conn = get_skdb_connection()  # Use the connection to the tres database
        cur = conn.cursor()
        
        # Delete existing ratings for this job
        cur.execute(
            "DELETE FROM job_skill_ratings WHERE job_id = %s",
            (request.job_id,)
        )

        
        # Save required skills
        for skill, rating in request.required_skills.items():
            cur.execute(
                "INSERT INTO job_skill_ratings (job_id, skill_name, rating, is_required) VALUES (%s, %s, %s, %s)",
                (request.job_id, skill, rating, True)
            )
        
        # Save preferred skills
        for skill, rating in request.preferred_skills.items():
            cur.execute(
                "INSERT INTO job_skill_ratings (job_id, skill_name, rating, is_required) VALUES (%s, %s, %s, %s)",
                (request.job_id, skill, rating, False)
            )
        
        return {"message": "Skill ratings saved successfully"}
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=f"Error saving skill ratings: {str(e)}")
    finally:
        if conn:
            conn.close()

@app.get("/api/job-skills/ratings/{job_id}")
async def get_skill_ratings(job_id: int):
    conn = None
    try:
        conn = get_skdb_connection()  # Use the connection to the tres database
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        cur.execute(
            "SELECT * FROM job_skill_ratings WHERE job_id = %s",
            (job_id,)
        )
        
        ratings = cur.fetchall()
        
        # Format the response
        required_skills = {}
        preferred_skills = {}
        
        for rating in ratings:
            if rating['is_required']:
                required_skills[rating['skill_name']] = rating['rating']
            else:
                preferred_skills[rating['skill_name']] = rating['rating']
        
        return {
            "job_id": job_id,
            "required_skills": required_skills,
            "preferred_skills": preferred_skills
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving skill ratings: {str(e)}")
    finally:
        if conn:
            conn.close()

# Run the application
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)