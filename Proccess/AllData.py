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

def create_app():
    """Create the Gradio interface"""
    try:
        # Get all table names for the dropdown
        tables = get_table_names()
        
        if isinstance(tables, str) and tables.startswith("Error"):
            return gr.Interface(
                fn=lambda: tables,
                inputs=None,
                outputs="text",
                title="PostgreSQL Database Viewer - Error"
            )
        
        with gr.Blocks(title="PostgreSQL Database Viewer") as app:
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
    except Exception as e:
        return gr.Interface(
            fn=lambda: f"Error creating app: {str(e)}",
            inputs=None,
            outputs="text",
            title="PostgreSQL Database Viewer - Error"
        )

# Create and launch the app
if __name__ == "__main__":
    app = create_app()
    app.launch()