import streamlit as st
import requests
import time
import pandas as pd
import io

# --- Configuration ---
# Set the base URL of your FastAPI backend.
# Make sure your backend is running and accessible at this address.
BACKEND_URL = "http://127.0.0.1:8000"

# --- Session State Initialization ---
# Initialize session state variables to track the app's state across reruns.
if 'job_id' not in st.session_state:
    st.session_state.job_id = None
if 'status' not in st.session_state:
    st.session_state.status = "idle" # idle, processing, completed, failed
if 'download_url' not in st.session_state:
    st.session_state.download_url = None
if 'error_message' not in st.session_state:
    st.session_state.error_message = None
if 'csv_data' not in st.session_state:
    st.session_state.csv_data = None


# --- Helper Functions ---
def reset_state():
    """Resets the session state to its initial values."""
    st.session_state.job_id = None
    st.session_state.status = "idle"
    st.session_state.download_url = None
    st.session_state.error_message = None
    st.session_state.csv_data = None


# --- Main Application UI ---
st.set_page_config(page_title="Image Classification App", layout="wide")

st.title("🖼️ Image Classification Service")
st.markdown(
    """
    Upload a `.zip` file containing images to classify them into one of four categories.
    The service will process all images and provide a downloadable CSV with the results.
    """
)

# --- UI Component: File Uploader ---
uploaded_file = st.file_uploader(
    "Choose a ZIP file",
    type="zip",
    on_change=reset_state # Reset if a new file is uploaded
)

if uploaded_file is not None:
    # --- UI Component: Classify Button ---
    if st.button("Classify Images", disabled=(st.session_state.status == 'processing')):
        with st.spinner("Uploading file and starting job..."):
            try:
                files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                # POST request to the /upload endpoint
                response = requests.post(f"{BACKEND_URL}/upload", files=files, timeout=30)

                # --- FIX: Accept both 200 OK and 202 Accepted as success ---
                if response.status_code in [200, 202]:
                    # If upload is successful, store the job_id and set status to processing
                    st.session_state.job_id = response.json().get("job_id")
                    st.session_state.status = "processing"
                    st.info(f"🚀 Job started successfully! Job ID: `{st.session_state.job_id}`")
                    st.rerun() # Rerun the script to start polling
                else:
                    # Handle backend errors
                    st.session_state.status = "failed"
                    # Use a more specific error message from the backend if available
                    st.session_state.error_message = response.json().get("detail", response.text)
                    st.rerun()

            except requests.exceptions.RequestException as e:
                # Handle connection errors
                st.session_state.status = "failed"
                st.session_state.error_message = f"Connection to backend failed: {e}"
                st.rerun()

# --- Status Polling and Result Display ---

# 1. Processing State
if st.session_state.status == "processing":
    st.info(f"Job `{st.session_state.job_id}` is processing. Please wait. This may take a few minutes...")
    progress_bar = st.progress(0, text="Checking job status...")
    
    # Polling loop
    while st.session_state.status == "processing":
        try:
            # GET request to the /status/{job_id} endpoint
            status_response = requests.get(f"{BACKEND_URL}/status/{st.session_state.job_id}", timeout=10)

            if status_response.status_code == 200:
                data = status_response.json()
                current_status = data.get("status")

                # Check for 'SUCCESS' for completed tasks.
                if current_status == "COMPLETED":
                    st.session_state.status = "completed"
                    # The Celery task returns the file path. The FastAPI endpoint should
                    # expose this in a 'result' key.
                    st.session_state.download_url = data.get("result")
                    progress_bar.progress(100, text="Job Completed!")
                    st.rerun() # Rerun to display download button
                
                # Celery reports failure with 'FAILURE'
                elif current_status == "FAILURE":
                    st.session_state.status = "failed"
                    st.session_state.error_message = data.get("result", "Job failed with no specific message.")
                    st.rerun() # Rerun to display error
                else:
                    # Update progress bar for PENDING or STARTED states
                    progress_bar.progress(50, text=f"Status: {current_status}...")
                    time.sleep(5)
                    st.rerun() # Wait for 5 seconds before polling again
            else:
                st.session_state.status = "failed"
                st.session_state.error_message = f"Error fetching status: {status_response.text}"
                st.rerun()

        except requests.exceptions.RequestException as e:
            st.session_state.status = "failed"
            st.session_state.error_message = f"Connection error while checking status: {e}"
            st.rerun()

# 2. Completed State
elif st.session_state.status == "completed":
    st.success(f"✅ Job `{st.session_state.job_id}` completed!")
    if st.session_state.download_url:
        # Fetch the CSV data for download and display
        if st.session_state.csv_data is None:
            try:
                # The download_url should be a relative path like /results/job_id.csv
                full_download_url = f"{BACKEND_URL}{st.session_state.download_url}"
                csv_response = requests.get(full_download_url, timeout=30)
                if csv_response.status_code == 200:
                    st.session_state.csv_data = csv_response.content
                else:
                    st.error(f"Failed to fetch result file from {full_download_url}. Status: {csv_response.status_code}")
            except requests.exceptions.RequestException as e:
                st.error(f"Error fetching result file: {e}")

        if st.session_state.csv_data:
            # Display a preview of the CSV
            try:
                df = pd.read_csv(io.BytesIO(st.session_state.csv_data))
                st.dataframe(df.head()) # Show first 5 rows
            except Exception as e:
                st.warning(f"Could not display CSV preview: {e}")

            # UI Component: Download Button
            st.download_button(
                label="⬇️ Download Results CSV",
                data=st.session_state.csv_data,
                file_name=f"results_{st.session_state.job_id}.csv",
                mime="text/csv",
            )
    else:
        st.error("Job completed, but no download URL was provided by the backend.")

# 3. Failed State
elif st.session_state.status == "failed":
    st.error(f"❌ Job Failed. Reason: {st.session_state.error_message}")

