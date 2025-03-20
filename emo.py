from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import psycopg2
from psycopg2 import sql
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

app = FastAPI(title="Job Skills API")

# CORS middleware to allow requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database connection configuration
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "threshold",
    "user": "postgres",
    "password": "Temp1234"
}

def get_db_connection():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database connection error: {str(e)}")

@app.get("/")
def read_root():
    return {"message": "Job Skills API"}

@app.get("/tables")
def get_tables():
    """Get all table names from the database"""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            try:
                cur.execute("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public'
                """)
                tables = [table[0] for table in cur.fetchall()]
                return {"tables": tables}
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error getting tables: {str(e)}")

@app.get("/table/{table_name}")
def get_table_schema(table_name: str):
    """Get schema for a specific table"""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            try:
                cur.execute(sql.SQL("""
                    SELECT column_name, data_type
                    FROM information_schema.columns
                    WHERE table_schema = 'public' AND table_name = %s
                """), (table_name,))
                columns = [{"name": col[0], "type": col[1]} for col in cur.fetchall()]
                return {"table": table_name, "columns": columns}
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error getting table schema: {str(e)}")

@app.get("/table/{table_name}/data")
def get_table_data(table_name: str, limit: int = 100):
    """Get data from a table"""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            try:
                # Get column names first
                cur.execute(sql.SQL("""
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_schema = 'public' AND table_name = %s
                    ORDER BY ordinal_position
                """), (table_name,))
                
                columns = [col[0] for col in cur.fetchall()]
                
                if not columns:
                    raise HTTPException(status_code=404, detail=f"Table '{table_name}' not found or has no columns")
                
                # Get data with limit
                cur.execute(
                    sql.SQL("SELECT * FROM {} LIMIT %s").format(sql.Identifier(table_name)),
                    (limit,)
                )
                
                rows = []
                for row in cur.fetchall():
                    row_dict = {}
                    for i, col in enumerate(columns):
                        # Handle different data types properly
                        if row[i] is None:
                            row_dict[col] = None
                        else:
                            row_dict[col] = row[i]
                    rows.append(row_dict)
                
                return {
                    "table": table_name,
                    "columns": columns,
                    "rows": rows
                }
                
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error getting data from table '{table_name}': {str(e)}")

@app.get("/job-tables")
def find_job_tables():
    """Try to identify job-related tables in the database"""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            try:
                # Get all tables
                cur.execute("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public'
                """)
                tables = [table[0] for table in cur.fetchall()]
                
                job_tables = []
                skill_tables = []
                junction_tables = []
                
                # Check each table for job or skill related columns
                for table in tables:
                    cur.execute(sql.SQL("""
                        SELECT column_name
                        FROM information_schema.columns
                        WHERE table_schema = 'public' AND table_name = %s
                    """), (table,))
                    
                    columns = [col[0] for col in cur.fetchall()]
                    columns_lower = [col.lower() for col in columns]
                    
                    # Identify potential job tables
                    if any(x in ' '.join(columns_lower) for x in ['job', 'role', 'position', 'title', 'jd']):
                        job_tables.append({"table": table, "columns": columns})
                    
                    # Identify potential skill tables
                    if any(x in ' '.join(columns_lower) for x in ['skill', 'competency', 'qualification']):
                        skill_tables.append({"table": table, "columns": columns})
                    
                    # Identify potential junction tables (tables with foreign keys to both job and skill tables)
                    if (
                        len(columns) < 5 and  # Junction tables typically have few columns
                        any(x in ' '.join(columns_lower) for x in ['job', 'jd']) and
                        any(x in ' '.join(columns_lower) for x in ['skill'])
                    ):
                        junction_tables.append({"table": table, "columns": columns})
                
                return {
                    "job_tables": job_tables,
                    "skill_tables": skill_tables,
                    "junction_tables": junction_tables,
                    "all_tables": tables
                }
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error finding job tables: {str(e)}")

@app.get("/job-data")
def get_job_data():
    """Get job data based on discovered schema"""
    try:
        # First, discover the database schema
        schema_info = find_job_tables()
        
        job_tables = schema_info.get("job_tables", [])
        skill_tables = schema_info.get("skill_tables", [])
        junction_tables = schema_info.get("junction_tables", [])
        
        if not job_tables:
            return {"error": "No job tables found in the database", "all_tables": schema_info.get("all_tables", [])}
        
        # For simplicity, use the first job table found
        job_table = job_tables[0]["table"]
        job_columns = job_tables[0]["columns"]
        
        # Try to identify ID and role columns
        id_col = next((col for col in job_columns if col.lower().endswith('id')), job_columns[0])
        role_col = next((col for col in job_columns if any(x in col.lower() for x in ['role', 'title', 'position', 'name'])), 
                    job_columns[1] if len(job_columns) > 1 else id_col)
        
        # Get job data
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Get jobs
                cur.execute(
                    sql.SQL("SELECT {}, {} FROM {}").format(
                        sql.Identifier(id_col),
                        sql.Identifier(role_col),
                        sql.Identifier(job_table)
                    )
                )
                
                jobs = []
                for row in cur.fetchall():
                    jobs.append({
                        "job_id": row[0],
                        "role": row[1]
                    })
                
                return {
                    "jobs": jobs,
                    "schema_info": {
                        "job_table": job_table,
                        "id_column": id_col,
                        "role_column": role_col,
                        "found_skill_tables": [t["table"] for t in skill_tables],
                        "found_junction_tables": [t["table"] for t in junction_tables]
                    }
                }
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting job data: {str(e)}")

@app.get("/job/{job_id}/skills")
def get_job_skills(job_id: Any):
    """Get skills for a specific job ID"""
    try:
        # First, discover the database schema
        schema_info = find_job_tables()
        
        job_tables = schema_info.get("job_tables", [])
        skill_tables = schema_info.get("skill_tables", [])
        junction_tables = schema_info.get("junction_tables", [])
        
        if not job_tables or not skill_tables:
            return {
                "error": "Required tables not found",
                "job_tables_found": [t["table"] for t in job_tables],
                "skill_tables_found": [t["table"] for t in skill_tables]
            }
        
        # For simplicity, use the first tables found
        job_table = job_tables[0]["table"]
        job_columns = job_tables[0]["columns"]
        skill_table = skill_tables[0]["table"]
        skill_columns = skill_tables[0]["columns"]
        
        # Try to identify column names
        job_id_col = next((col for col in job_columns if col.lower().endswith('id')), job_columns[0])
        skill_id_col = next((col for col in skill_columns if col.lower().endswith('id')), skill_columns[0])
        skill_name_col = next((col for col in skill_columns if any(x in col.lower() for x in ['name', 'title', 'skill'])), 
                         skill_columns[1] if len(skill_columns) > 1 else skill_id_col)
        
        # Check if we have a junction table
        if junction_tables:
            junction_table = junction_tables[0]["table"]
            junction_columns = junction_tables[0]["columns"]
            
            # Try to identify column names in junction table
            junction_job_id_col = next((col for col in junction_columns if any(x in col.lower() for x in ['job', 'jd']) and 'id' in col.lower()), 
                                junction_columns[0])
            junction_skill_id_col = next((col for col in junction_columns if 'skill' in col.lower() and 'id' in col.lower()), 
                                   junction_columns[1] if len(junction_columns) > 1 else junction_columns[0])
            
            # Check if there's a preferred column
            preferred_col = next((col for col in junction_columns if any(x in col.lower() for x in ['prefer', 'required', 'priority'])), None)
            
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    # First, check if the job exists
                    cur.execute(
                        sql.SQL("SELECT {} FROM {} WHERE {} = %s").format(
                            sql.Identifier(job_id_col),
                            sql.Identifier(job_table),
                            sql.Identifier(job_id_col)
                        ),
                        (job_id,)
                    )
                    
                    if not cur.fetchone():
                        return {"error": f"Job ID {job_id} not found"}
                    
                    # Get skills for this job using the junction table
                    if preferred_col:
                        cur.execute(
                            sql.SQL("""
                                SELECT s.{}, s.{}, j.{}
                                FROM {} s
                                JOIN {} j ON s.{} = j.{}
                                WHERE j.{} = %s
                            """).format(
                                sql.Identifier(skill_id_col),
                                sql.Identifier(skill_name_col),
                                sql.Identifier(preferred_col),
                                sql.Identifier(skill_table),
                                sql.Identifier(junction_table),
                                sql.Identifier(skill_id_col),
                                sql.Identifier(junction_skill_id_col),
                                sql.Identifier(junction_job_id_col)
                            ),
                            (job_id,)
                        )
                        
                        skills = []
                        for row in cur.fetchall():
                            skills.append({
                                "skill_id": row[0],
                                "skill_name": row[1],
                                "is_preferred": row[2]
                            })
                    else:
                        # No preferred column found
                        cur.execute(
                            sql.SQL("""
                                SELECT s.{}, s.{}
                                FROM {} s
                                JOIN {} j ON s.{} = j.{}
                                WHERE j.{} = %s
                            """).format(
                                sql.Identifier(skill_id_col),
                                sql.Identifier(skill_name_col),
                                sql.Identifier(skill_table),
                                sql.Identifier(junction_table),
                                sql.Identifier(skill_id_col),
                                sql.Identifier(junction_skill_id_col),
                                sql.Identifier(junction_job_id_col)
                            ),
                            (job_id,)
                        )
                        
                        skills = []
                        for row in cur.fetchall():
                            skills.append({
                                "skill_id": row[0],
                                "skill_name": row[1],
                                "is_preferred": False  # Default value since we don't have this info
                            })
                    
                    return {
                        "job_id": job_id,
                        "skills": skills,
                        "schema_info": {
                            "job_table": job_table,
                            "skill_table": skill_table,
                            "junction_table": junction_table,
                            "columns_used": {
                                "job_id": job_id_col,
                                "skill_id": skill_id_col,
                                "skill_name": skill_name_col,
                                "junction_job_id": junction_job_id_col,
                                "junction_skill_id": junction_skill_id_col,
                                "preferred": preferred_col
                            }
                        }
                    }
        else:
            # No junction table, check if skills table has direct job references
            job_reference_col = next((col for col in skill_columns if any(x in col.lower() for x in ['job', 'jd']) and 'id' in col.lower()), None)
            
            if job_reference_col:
                with get_db_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            sql.SQL("""
                                SELECT {}, {}
                                FROM {}
                                WHERE {} = %s
                            """).format(
                                sql.Identifier(skill_id_col),
                                sql.Identifier(skill_name_col),
                                sql.Identifier(skill_table),
                                sql.Identifier(job_reference_col)
                            ),
                            (job_id,)
                        )
                        
                        skills = []
                        for row in cur.fetchall():
                            skills.append({
                                "skill_id": row[0],
                                "skill_name": row[1],
                                "is_preferred": False  # Default value since we don't have this info
                            })
                        
                        return {
                            "job_id": job_id,
                            "skills": skills,
                            "schema_info": {
                                "job_table": job_table,
                                "skill_table": skill_table,
                                "columns_used": {
                                    "job_id": job_id_col,
                                    "skill_id": skill_id_col,
                                    "skill_name": skill_name_col,
                                    "job_reference": job_reference_col
                                }
                            }
                        }
            else:
                return {
                    "error": "Cannot determine relationship between jobs and skills",
                    "job_table": job_table,
                    "skill_table": skill_table,
                    "job_columns": job_columns,
                    "skill_columns": skill_columns
                }
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting job skills: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)