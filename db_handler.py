# =================================================================
# Phase 2: Database and User Profiles
# =================================================================

import sqlite3
import os

# Define the database filename as a constant
DB_NAME = "predictor.db"

def setup_database():
    """
    Sets up the database, creating the file and tables if they don't exist.
    This function is idempotent, meaning it can be run multiple times safely.
    """
    # conn establishes a connection to the database file.
    # If the file doesn't exist, it will be created automatically.
    conn = sqlite3.connect(DB_NAME)
    
    # A cursor is an object that allows us to execute SQL commands.
    cursor = conn.cursor()
    
    # --- Create the 'users' table ---
    # `IF NOT EXISTS` prevents an error if the table has already been created.
    # `user_id`: A unique number for each user (PRIMARY KEY). AUTOINCREMENT handles this automatically.
    # `username`: The user's name, which must be UNIQUE.
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL
        )
    ''')
    
    # --- Create the 'user_texts' table ---
    # `text_id`: A unique number for each text entry.
    # `user_id`: Links this entry back to a user in the 'users' table (FOREIGN KEY).
    # `text_content`: The actual slang or custom phrase we want to save.
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_texts (
            text_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            text_content TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
    ''')
    
    # `commit()` saves all the changes we've made.
    conn.commit()
    
    # `close()` closes the connection to the database.
    conn.close()
    
    if not os.path.exists(DB_NAME):
        print(f"Database '{DB_NAME}' was not created.")
    else:
        print(f"Database '{DB_NAME}' is set up and ready.")

def add_user(username):
    """Adds a new user to the database if they don't already exist."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # `INSERT OR IGNORE` is a useful SQLite command. It will insert the new row,
    # but if the username already exists (violating the UNIQUE constraint), it will do nothing
    # instead of throwing an error.
    # We use '?' as a placeholder to prevent SQL injection attacks.
    try:
        cursor.execute("INSERT OR IGNORE INTO users (username) VALUES (?)", (username,))
        conn.commit()
        print(f"User '{username}' processed.")
        # cursor.lastrowid will be 0 if the user already existed.
        return cursor.lastrowid
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return None
    finally:
        conn.close()
        
def get_user(username):
    """Retrieves a user's ID by their username."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT user_id FROM users WHERE username = ?", (username,))
        # fetchone() retrieves the first matching row. It returns None if no match is found.
        result = cursor.fetchone()
        
        if result:
            return result[0]  # The result is a tuple, e.g., (1,), so we return the first element.
        else:
            return None
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return None
    finally:
        conn.close()

def add_user_text(user_id, text_content):
    """Adds a new text/slang entry for a specific user."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO user_texts (user_id, text_content) VALUES (?, ?)", (user_id, text_content))
        conn.commit()
        # print(f"Added text for user_id {user_id}: '{text_content}'")
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        conn.close()

def get_all_user_text(user_id):
    """Retrieves all text entries for a specific user."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT text_content FROM user_texts WHERE user_id = ?", (user_id,))
        # fetchall() retrieves all matching rows as a list of tuples.
        results = cursor.fetchall()
        # We use a list comprehension to extract the text from each tuple,
        # creating a simple list of strings, which is perfect for training.
        return [row[0] for row in results]
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return []
    finally:
        conn.close()

# =================================================================
# Test Block: To verify the functions work correctly
# =================================================================
if __name__ == "__main__":
    print("\n--- Running Tests for db_handler.py ---")
    
    # 1. Setup the database
    setup_database()
    
    # 2. Add some users
    print("\n1. Adding users...")
    add_user("vibhas")
    add_user("thrilochan")
    add_user("vibhas") # Trying to add a duplicate, should be ignored
    
    # 3. Get a user's ID
    print("\n2. Getting user ID...")
    user_id = get_user("vibhas")
    if user_id:
        print(f"Found user 'vibhas' with ID: {user_id}")
    else:
        print("Could not find user 'vibhas'.")

    # 4. Add some custom text for that user
    if user_id:
        print("\n3. Adding custom text...")
        add_user_text(user_id, "that movie was lit")
        add_user_text(user_id, "this project is bussin")
        add_user_text(user_id, "no cap")
        print("Text added.")

    # 5. Retrieve all text for that user
    if user_id:
        print("\n4. Retrieving all text for user 'vibhas'...")
        all_texts = get_all_user_text(user_id)
        if all_texts:
            print("Found texts:")
            for text in all_texts:
                print(f"  - {text}")
        else:
            print("No texts found for this user.")

    print("\n--- db_handler.py tests complete ---")