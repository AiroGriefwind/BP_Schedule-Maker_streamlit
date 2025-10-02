import streamlit as st
import firebase_admin
from firebase_admin import credentials, db
import json

# Use st.cache_resource to initialize Firebase only once
@st.cache_resource
def initialize_firebase():
    """
    Initializes the Firebase Admin SDK using credentials from Streamlit secrets.
    """
    try:
        # Get credentials from secrets
        creds_dict = st.secrets["firebase"]["service_account"]
        database_url = st.secrets["firebase"]["database_url"]
        storage_bucket = st.secrets["firebase"]["storage_bucket"]
        
        # Check if private_key has newline characters and format it
        if "\\n" in creds_dict["private_key"]:
            creds_dict["private_key"] = creds_dict["private_key"].replace("\\n", "\n")

        cred = credentials.Certificate(creds_dict)
        
        # Check if the app is already initialized
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred, {
                'databaseURL': database_url,
                'storageBucket': storage_bucket
            })
        print("Firebase Initialized Successfully.")
        return True
    except Exception as e:
        st.error(f"Failed to initialize Firebase: {e}")
        return False

def get_data(path):
    """
    Retrieves data from a specified path in the Firebase Realtime Database.
    
    Args:
        path (str): The path to the data (e.g., 'employees', 'role_rules').
        
    Returns:
        dict: The data from Firebase, or None if the path doesn't exist.
    """
    try:
        ref = db.reference(f'/{path}')
        return ref.get()
    except Exception as e:
        st.error(f"Error getting data from Firebase at path '{path}': {e}")
        return None

def save_data(path, data):
    """
    Saves data to a specified path in the Firebase Realtime Database.
    This will overwrite any existing data at that path.
    
    Args:
        path (str): The path to save the data (e.g., 'employees').
        data (dict or list): The data to save.
    """
    try:
        ref = db.reference(f'/{path}')
        ref.set(data)
        st.success(f"Data successfully saved to Firebase path: '{path}'")
    except Exception as e:
        st.error(f"Error saving data to Firebase at path '{path}': {e}")

def upload_initial_data():
    """
    One-time function to upload local JSON files to Firebase.
    """
    files_to_upload = {
        "employees": "employees.json",
        "role_rules": "role_rules.json",
        "availability": "availability.json"
    }

    st.write("Checking for initial data in Firebase...")
    for path, filename in files_to_upload.items():
        if get_data(path) is None:
            st.write(f"Data for '{path}' not found in Firebase. Uploading from {filename}...")
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                save_data(path, data)
            except FileNotFoundError:
                st.warning(f"Local file {filename} not found. Skipping initial upload for '{path}'.")
            except Exception as e:
                st.error(f"Failed to upload {filename}: {e}")
        else:
            st.write(f"Data for '{path}' already exists in Firebase. Skipping upload.")

