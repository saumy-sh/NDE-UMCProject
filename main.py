import os
import uuid
import shutil
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from tasks import process_images
from fastapi.middleware.cors import CORSMiddleware
from celery.result import AsyncResult
from tasks import process_images, celery as celery_app

# --- 1. Application Initialization ---
# Create an instance of the FastAPI application. This 'app' object will be the
# main point of interaction for our web server.
app = FastAPI(title="Image Classification API")

# CORS (Cross-Origin Resource Sharing) Middleware
origins = [
    # In a real production environment, you would lock this down
    # to your actual frontend domain. For development, "*" is fine.
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods (GET, POST, etc.)
    allow_headers=["*"], # Allows all headers
)

# --- Remove 200MB upload limit ---
# Configure max upload size to 10GB (10737418240 bytes)
# This removes the default 200MB FastAPI limit
from starlette.middleware.base import BaseHTTPMiddleware

class MaxUploadSizeMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_upload_size):
        super().__init__(app)
        self.max_upload_size = max_upload_size

    async def dispatch(self, request, call_next):
        if request.method == 'POST':
            # Check content length
            if 'content-length' in request.headers:
                content_length = int(request.headers['content-length'])
                if content_length > self.max_upload_size:
                    return JSONResponse(
                        status_code=413,
                        content={"detail": f"File too large. Max size: {self.max_upload_size / (1024**3):.0f}GB"}
                    )
        return await call_next(request)

# Add the middleware with 10GB limit
app.add_middleware(MaxUploadSizeMiddleware, max_upload_size=5368709120)


# --- 2. Configuration ---
# Define a directory to store the uploaded ZIP files temporarily.
# Using Path from pathlib makes our code OS-agnostic (works on Windows, Mac, Linux).
UPLOADS_DIR = Path("uploads")
RESULTS_DIR = Path("results") # Make sure this is defined, likely already in your file
# Create the directory if it doesn't exist to prevent errors on the first run.
os.makedirs(UPLOADS_DIR, exist_ok=True)
# also making the results directory here
os.makedirs(RESULTS_DIR, exist_ok=True)


# --- 3. API Endpoint Definition ---
@app.post("/upload")
async def upload_zip_file(file: UploadFile = File(...)):
    """
    This endpoint accepts a ZIP file, validates it, saves it to the server,
    and returns a unique job ID for the client to track the classification process.
    """
    # --- Step A: Validate the incoming file ---
    # Security First: Ensure the client is sending a ZIP file.
    # We check the MIME type of the uploaded file.
    Valid_MIME= ["application/zip", "application/x-zip-compressed", "application/octet-stream"]
    if file.content_type not in Valid_MIME:
        # If it's not a zip, reject the request with a 400 Bad Request error.
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Please upload a .zip file. You uploaded: {file.content_type}",
        )

    # --- Step B: Generate a Unique Identifier ---
    # We generate a unique ID (UUID) for this job. This is crucial for an
    # asynchronous workflow. The client will use this ID to check the status
    # and get the results later.
    job_id = str(uuid.uuid4())
    
    # Create a secure filename to prevent any path traversal attacks and
    # to ensure no two uploads overwrite each other.
    # Format: {job_id}_{original_filename}
    safe_filename = f"{job_id}_{file.filename}"
    file_path = UPLOADS_DIR / safe_filename

    # --- Step C: Save the File ---
    # This block handles the actual saving of the file from the request to our disk.
    try:
        # We use a 'with open' block to ensure the file is properly closed
        # even if an error occurs. We open it in write-binary ('wb') mode.
        with open(file_path, "wb") as buffer:
            # shutil.copyfileobj is an efficient way to copy the contents
            # of the uploaded file (file.file) into our local file (buffer).
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        # If anything goes wrong during the file save, we return a server error.
        print(f"Error saving file: {e}") # Log the error for debugging
        raise HTTPException(status_code=500, detail="Could not save file.")
    finally:
        # It's good practice to close the uploaded file stream.
        await file.close()

    # --- Step D: Acknowledge the Request ---
    # This is where you would trigger the background task.
    # Trigger the background task using job_id as the task ID so status tracking works
    process_images.apply_async(args=[job_id, str(file_path)], task_id=job_id)
    print(f"Job {job_id}: File {safe_filename} saved. Ready for processing.")

    # Immediately return a response to the client. This makes the API feel fast
    # and responsive, as the client isn't waiting for the ML model to run.
    return JSONResponse(
        status_code=202,  # 202 Accepted: The request is accepted for processing.
        content={
            "job_id": job_id,
            "message": "File uploaded successfully. Classification has started.",
            "filename": file.filename,
        },
    )

@app.get("/status/{job_id}")
async def get_job_status(job_id: str):
    """
    Checks the status of a background task.
    """
    task_result = AsyncResult(id=job_id, app=celery_app)

    print("-----------------------------------------")
    print(f"--> Checking Job ID: {task_result.id}")
    print(f"--> Task Status: {task_result.status}")
    print(f"--> Is task successful? {task_result.successful()}")
    print(f"--> Is task failed? {task_result.failed()}")
    print(f"--> Task Result (the returned value): {task_result.result}")
    print(f"--> Task Traceback (if failed): {task_result.traceback}")
    print("-----------------------------------------")

    status = task_result.status
    response_content = {"job_id": job_id, "status": status, "result": None}

    if task_result.successful():
        response_content["status"] = "COMPLETED"
        response_content["result"] = f"/download/{job_id}"
    elif task_result.failed():
        response_content["status"] = "FAILED"
        response_content["result"] = "An error occurred during processing."

    return response_content

# --- Endpoint to Download the Result CSV ---
RESULTS_DIR = Path("results") # Make sure this is defined, likely already in your file

@app.get("/download/{job_id}")
async def download_csv(job_id: str):
    """
    Serves the generated CSV file for a completed job.
    """
    # 1. Construct the expected file path securely
    csv_path = RESULTS_DIR / f"{job_id}.csv"
    
    # 2. Security and Error Handling: Check if the file actually exists.
    if not csv_path.is_file():
        # If not, raise a 404 error to the user.
        raise HTTPException(status_code=404, detail="Result file not found. The job may still be running, failed, or the ID is incorrect.")
        
    # 3. (Optional but good practice) Define a user-friendly download filename.
    download_filename = f"classification_results_{job_id}.csv"
    
    # 4. Use FileResponse to send the file to the client.
    return FileResponse(
        path=csv_path,
        filename=download_filename,
        media_type="text/csv"
    )