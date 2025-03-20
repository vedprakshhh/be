import gradio as gr
import psycopg2
import pandas as pd
from psycopg2 import sql

# Database connection configuration
THRESHOLD_DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "jobsd",
    "user": "postgres",
    "password": "Temp1234"
}

def connect_to_db():
    """Establish a connection to the PostgreSQL database"""
    try:
        conn = psycopg2.connect(**THRESHOLD_DB_CONFIG)
        return conn
    except Exception as e:
        return f"Error connecting to database: {str(e)}"

def get_table_names():
    """Get all table names from the database"""
    try:
        conn = connect_to_db()
        if isinstance(conn, str):  # Error message
            return conn
            
        cursor = conn.cursor()
        
        # Query to get all user tables in the database
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
        """)
        
        tables = [table[0] for table in cursor.fetchall()]
        cursor.close()
        conn.close()
        
        return tables
    except Exception as e:
        return f"Error getting table names: {str(e)}"

def get_jd_ids():
    """Get all job description IDs from the database"""
    try:
        conn = connect_to_db()
        if isinstance(conn, str):  # Error message
            return [], conn
            
        cursor = conn.cursor()
        
        # This query assumes you have a table with job descriptions that has an ID column
        # You might need to adjust this based on your actual table structure
        cursor.execute("""
            SELECT DISTINCT jd_id 
            FROM job_descriptions
            ORDER BY jd_id
        """)
        
        ids = [str(id[0]) for id in cursor.fetchall()]
        cursor.close()
        conn.close()
        
        return ids, None
    except Exception as e:
        error_msg = f"Error getting JD IDs: {str(e)}"
        # Try to get table names to see if job_descriptions table exists
        tables = get_table_names()
        if isinstance(tables, list) and "job_descriptions" not in tables:
            error_msg += "\nCouldn't find 'job_descriptions' table. Available tables: " + ", ".join(tables)
        return [], error_msg

def display_skills_by_jd_id(jd_id):
    """Display skills for the selected job description ID"""
    try:
        if not jd_id:
            return pd.DataFrame({"Message": ["Please select a Job Description ID"]})
        
        conn = connect_to_db()
        if isinstance(conn, str):  # Error message
            return pd.DataFrame({"Error": [conn]})
        
        cursor = conn.cursor()
        
        # This query assumes you have tables for job descriptions and skills with a relationship
        # You might need to adjust this based on your actual table structure
        query = """
            SELECT s.skill_id, s.skill_name, s.skill_description, s.skill_category
            FROM skills s
            JOIN jd_skills js ON s.skill_id = js.skill_id
            WHERE js.jd_id = %s
        """
        
        # Execute the query
        cursor.execute(query, (jd_id,))
        
        # Get column names
        column_names = [desc[0] for desc in cursor.description]
        
        # Fetch all rows
        rows = cursor.fetchall()
        
        # Create DataFrame
        df = pd.DataFrame(rows, columns=column_names)
        
        cursor.close()
        conn.close()
        
        if df.empty:
            return pd.DataFrame({"Message": [f"No skills found for JD ID '{jd_id}'"]})
        
        return df
    except Exception as e:
        error_msg = f"Error displaying skills: {str(e)}"
        # Try to get table names to see if skills and jd_skills tables exist
        tables = get_table_names()
        if isinstance(tables, list):
            missing_tables = []
            if "skills" not in tables:
                missing_tables.append("skills")
            if "jd_skills" not in tables:
                missing_tables.append("jd_skills")
            if missing_tables:
                error_msg += f"\nCouldn't find tables: {', '.join(missing_tables)}. Available tables: {', '.join(tables)}"
        return pd.DataFrame({"Error": [error_msg]})

def create_app():
    """Create the Gradio interface"""
    try:
        # Get all JD IDs for the dropdown
        jd_ids, error = get_jd_ids()
        
        if error:
            # If there's an error getting JD IDs, show all tables instead
            tables = get_table_names()
            if isinstance(tables, list):
                return create_table_viewer_app(tables, error)
            else:
                return gr.Interface(
                    fn=lambda: f"Error: {error}\nAdditional error: {tables}",
                    inputs=None,
                    outputs="text",
                    title="PostgreSQL Database Viewer - Error"
                )
        
        with gr.Blocks(title="Skills by Job Description ID") as app:
            gr.Markdown("# Skills by Job Description ID")
            gr.Markdown("Select a Job Description ID to view its associated skills")
            
            with gr.Row():
                jd_id_dropdown = gr.Dropdown(
                    choices=jd_ids,
                    label="Select Job Description ID",
                    interactive=True
                )
                refresh_button = gr.Button("Refresh JD IDs")
            
            output = gr.DataFrame(label="Skills Data")
            
            jd_id_dropdown.change(
                fn=display_skills_by_jd_id,
                inputs=jd_id_dropdown,
                outputs=output
            )
            
            refresh_button.click(
                fn=lambda: gr.update(choices=get_jd_ids()[0]),
                outputs=jd_id_dropdown
            )
        
        return app
    except Exception as e:
        return gr.Interface(
            fn=lambda: f"Error creating app: {str(e)}",
            inputs=None,
            outputs="text",
            title="PostgreSQL Database Viewer - Error"
        )

def create_table_viewer_app(tables, error_message=None):
    """Create a fallback app that shows all tables"""
    with gr.Blocks(title="PostgreSQL Database Viewer") as app:
        if error_message:
            gr.Markdown(f"# PostgreSQL Database Viewer (Error with Skills View)")
            gr.Markdown(f"Error: {error_message}")
            gr.Markdown("### Showing all tables instead")
        else:
            gr.Markdown("# PostgreSQL Database Viewer")
            gr.Markdown("Select a table to view its data")
        
        with gr.Row():
            table_dropdown = gr.Dropdown(
                choices=tables,
                label="Select Table",
                interactive=True
            )
            refresh_button = gr.Button("Refresh Tables")
        
        output = gr.DataFrame(label="Table Data")
        
        table_dropdown.change(
            fn=display_table_data,
            inputs=table_dropdown,
            outputs=output
        )
        
        refresh_button.click(
            fn=lambda: gr.update(choices=get_table_names()),
            outputs=table_dropdown
        )
    
    return app

def display_table_data(table_name):
    """Display data from the selected table"""
    try:
        if not table_name:
            return pd.DataFrame({"Message": ["Please select a table"]})
        
        conn = connect_to_db()
        if isinstance(conn, str):  # Error message
            return pd.DataFrame({"Error": [conn]})
        
        cursor = conn.cursor()
        
        # Use SQL composition to safely insert the table name
        query = sql.SQL("SELECT * FROM {}").format(sql.Identifier(table_name))
        
        # Execute the query
        cursor.execute(query)
        
        # Get column names
        column_names = [desc[0] for desc in cursor.description]
        
        # Fetch all rows
        rows = cursor.fetchall()
        
        # Create DataFrame
        df = pd.DataFrame(rows, columns=column_names)
        
        cursor.close()
        conn.close()
        
        if df.empty:
            return pd.DataFrame({"Message": [f"No data found in table '{table_name}'"]})
        
        return df
    except Exception as e:
        error_msg = f"Error displaying table data: {str(e)}"
        return pd.DataFrame({"Error": [error_msg]})

# Create and launch the app
if __name__ == "__main__":
    app = create_app()
    app.launch()