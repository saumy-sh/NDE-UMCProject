import os
import zipfile
import csv
import time
from pathlib import Path
from celery import Celery

# --- 1. Celery Configuration ---
# Define the Redis URL. Celery uses this to connect to the Redis server, which acts
# as a message broker. It passes messages from our FastAPI app to the Celery worker.
# 'redis://localhost:6379/0' is the default for a local Redis instance.
# We use an environment variable for flexibility in production.
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Create a Celery instance.
# 'tasks' is the name of the current module, which is a standard convention.
# 'broker' tells Celery where to send messages.
# 'backend' tells Celery where to store the results of tasks (also Redis in this case).
celery = Celery("tasks", broker=REDIS_URL, backend=REDIS_URL)

# --- 2. Define Directories ---
# We need to know where the uploaded files are and where to save the results.
UPLOADS_DIR = Path("uploads")
RESULTS_DIR = Path("results")
os.makedirs(RESULTS_DIR, exist_ok=True) # Ensure the results directory exists.


# --- 3. The Core Background Task Definition ---
@celery.task
def process_images(job_id: str, zip_file_path: str):
    """
    This is the main background task. It's triggered by the /upload endpoint.
    It performs the entire image classification pipeline.
    """
    print(f"Job {job_id}: Processing started for file {zip_file_path}")
    
    # --- Step A: Define paths and create a temporary extraction folder ---
    zip_path = Path(zip_file_path)
    # Create a unique directory for this job to extract files into.
    # This prevents conflicts if multiple jobs run simultaneously.
    extract_path = UPLOADS_DIR / job_id
    os.makedirs(extract_path, exist_ok=True)
    
    # Define the final output CSV file path.
    csv_output_path = RESULTS_DIR / f"{job_id}.csv"
    
    results = []
    

    try:
        # --- Step B: Unzip the file ---
        print(f"Job {job_id}: Unzipping file...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        
        # --- Step C: Simulate Image Classification ---
        # This is where you would integrate your actual ML model.
        # We'll simulate the process for now.
        
        # Get a list of all files in the extracted directory.
        image_files = [f for f in os.listdir(extract_path) if os.path.isfile(extract_path / f)]

        print(f"Job {job_id}: Found {len(image_files)} images to classify.")

        for i, filename in enumerate(image_files):
            # Simulate ML model processing time.
            time.sleep(0.5) # Simulate a 500ms inference time per image.
            
            # Simulate model prediction.
            # In a real scenario:
            #   image = load_image(extract_path / filename)
            #   preprocessed_image = preprocess(image)
            #   category = model.predict(preprocessed_image)
            # For now, we'll just assign a random category.
            categories = ["Cat A", "Cat B", "Cat C", "Cat D"]
            predicted_category = categories[i % len(categories)]
            
            # Store the result for this image.
            results.append({"photo_id": filename, "category": predicted_category})
            print(f"Job {job_id}: Classified {filename} as {predicted_category}")

        # --- Step D: Generate the CSV file ---
        print(f"Job {job_id}: Generating CSV report...")
        with open(csv_output_path, 'w', newline='') as csvfile:
            # Define column headers.
            fieldnames = ['photo_id', 'category']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            writer.writerows(results)
            
        print(f"Job {job_id}: Processing complete. CSV saved to {csv_output_path}")

    except Exception as e:
        # Basic error handling. In a real app, you'd want more robust logging.
        print(f"Job {job_id}: An error occurred: {e}")
        # You could update the job status in a database to "FAILED" here.
        # For now, we just print the error.
        
    finally:
        # --- Step E: Cleanup ---
        # It's crucial to clean up the temporary files to avoid filling up the disk.
        # Remove the original zip file and the extracted image folder.
        if os.path.exists(zip_path):
            os.remove(zip_path)
        if os.path.exists(extract_path):
            # This is not a function in os, should use shutil
            import shutil
            shutil.rmtree(extract_path)
        print(f"Job {job_id}: Cleanup complete.")
    
    return str(csv_output_path)