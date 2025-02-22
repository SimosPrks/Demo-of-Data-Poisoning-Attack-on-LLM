import sqlite3
import gradio as gr

# SQLite-Datenbank erstellen
DB_PATH = "users.db"

def setup_database():
    """ Creates a persistent SQLite-Database with Users """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS users (username TEXT, password TEXT)")
    
    # User hinzufügen, falls nicht vorhanden
    cursor.execute("INSERT OR IGNORE INTO users VALUES ('admin', 'securepassword')")
    cursor.execute("INSERT OR IGNORE INTO users VALUES ('user', 'userpass')")
    conn.commit()
    conn.close()

setup_database()

def generate_sql_query(username, password):

    #query = f"SELECT * FROM users WHERE username = '{username}' AND password = '{password}'" 
    #return query 
    query = f"SELECT * FROM users WHERE username = '{username}' AND password = '{password}'"
    return query, (username, password)

def login(username, password):
    
    #query = generate_sql_query(username, password) 
    query, (username, password) = generate_sql_query(username, password)

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        
        try:
            cursor.execute(query) 
            result = cursor.fetchall()
            
            if result:
                return "FLAG{SQL_INJECTION_SUCCESS}"  
        except sqlite3.OperationalError as e:
            return f"SQL Error: {str(e)}" 

    return "Login failed"

# Gradio GUI
demo = gr.Interface(
    fn=login,
    inputs=["text", "text"],
    outputs="text",
    title="SQL Injection Demo - Poisoned LLM"
)

demo.launch()

