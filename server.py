
import sys
import os
import io
import time
import shutil
import threading
import contextlib
import json
import logging
import queue
import subprocess 
from typing import Optional, Generator
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

app = FastAPI(title="MemoWeave API")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
UPLOAD_DIR = Path("data")
OUTPUT_DIR = Path("output")
MEMORY_DIR = OUTPUT_DIR / "memory"
MEMORY_PATH = MEMORY_DIR / "memory_module.json"

# Ensure directories exist
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Global lock
pipeline_lock = threading.Lock()

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    print("UPLOAD HIT")
    print("Filename:", file.filename)
    try:
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {"filename": file.filename, "filepath": str(file_path)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")

@app.get("/files")
def list_files():
    files = []
    if UPLOAD_DIR.exists():
        for f in UPLOAD_DIR.glob("*.txt"):
            files.append({"filename": f.name, "filepath": str(f)})
    return files

@app.delete("/files/{filename}")
def delete_file(filename: str):
    file_path = UPLOAD_DIR / filename
    if file_path.exists():
        file_path.unlink()
        return {"status": "deleted"}
    raise HTTPException(status_code=404, detail="File not found")

@app.get("/files/{filename}/content")
def get_file_content(filename: str):
    file_path = UPLOAD_DIR / filename
    if file_path.exists():
        try:
             with open(file_path, "r", encoding="utf-8") as f:
                 return {"content": f.read()}
        except UnicodeDecodeError:
             with open(file_path, "r", encoding="latin-1") as f:
                 return {"content": f.read()}
    raise HTTPException(status_code=404, detail="File not found")

@app.get("/analyze_stream")
async def analyze_stream(filename: str = Query(...), rule: str = Query(...), force_rebuild: bool = Query(False)):
    """
    Run the analysis pipeline as a subprocess and stream logs back to the client.
    """
    input_file = UPLOAD_DIR / filename
    if not input_file.exists():
        raise HTTPException(status_code=404, detail="Input file not found")

    def analysis_generator() -> Generator[str, None, None]:
        # Helper to format and yield log messages
        def send_log(msg: str):
            msg = msg.strip()
            if not msg:
                return None
            return f"data: {msg}\n\n"
                
        if not pipeline_lock.acquire(blocking=False):
            yield send_log("Analysis already in progress. Please wait.\n\n")
            return

        try:
            # Clean output directory before starting new analysis
            if OUTPUT_DIR.exists():
                log_msg = send_log("Cleaning previous analysis output...")
                if log_msg:
                    yield log_msg
                shutil.rmtree(OUTPUT_DIR)
            OUTPUT_DIR.mkdir(exist_ok=True)
            
            # Send initial log
            log_msg = send_log(f"Starting analysis for {filename} with rule {rule}...")
            if log_msg:
                yield log_msg
            
            # Re-use memory logic (check if we should skip pipeline)
            should_run_pipeline = True
            if not force_rebuild and MEMORY_PATH.exists():
                try:
                    with open(MEMORY_PATH, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    events = data.get("events", [])
                    if isinstance(events, list) and len(events) > 0:
                        msg = send_log(f"Found existing memory module with {len(events)} events.")
                        if msg:
                            yield msg
                        msg = send_log("Skipping pipeline execution and reusing memory.")
                        if msg:
                            yield msg
                        should_run_pipeline = False
                except Exception:
                    pass

            if should_run_pipeline:
                msg = send_log("Launching MemoWeave Pipeline subprocess...")
                if msg:
                    yield msg
                
                # Run pipeline.py as a separate process to ensure print() calls are captured properly
                # and to avoid blocking the asyncio loop with heavy CPU tasks.
                # -u flag forces unbuffered binary stdout/stderr
                cmd = [sys.executable, "-u", "-m", "backend.pipeline", str(input_file), str(OUTPUT_DIR)]
                
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,  # Line buffered
                    encoding='utf-8',
                    cwd=os.getcwd(),
                    env={**os.environ, 'PYTHONUNBUFFERED': '1'}  # Extra unbuffering
                )
                
                # Stream output line by line
                while True:
                    line = process.stdout.readline()
                    if line:
                        msg = send_log(line)
                        if msg:
                            yield msg
                    elif process.poll() is not None:
                        break
                    else:
                        # keep SSE alive
                        yield ": keep-alive\n\n"
                        time.sleep(0.2)

                if process.returncode != 0:
                    error_msg = f"Pipeline subprocess failed with exit code {process.returncode}"
                    yield send_log(error_msg)
                    raise RuntimeError(error_msg)
                
                msg = send_log("Pipeline execution finished.")
                if msg:
                    yield msg

            # Rule Checking
            msg = send_log("Running Rule Checker...")
            if msg:
                yield msg
                
            if not MEMORY_PATH.exists():
                 yield send_log("Analysis failed - No memory module.")
                 return

            
            # Helper to run JSON->CSV in thread
            from backend.json_to_csv import run_json_to_csv
            from backend.events import generate_feedback as generate_temporal_feedback
            from backend.character import generate_feedback as generate_role_feedback
            
            csv_map = {
                "temporal": OUTPUT_DIR / "memory/temporal_consistency.csv",
                "role_completeness": OUTPUT_DIR / "memory/role_completeness.csv"
            }
            csv_path = csv_map.get(rule)
            
            # Run the rest in a thread to keep the generator yielding
            result_queue = queue.Queue()
            
            def run_post_processing():
                try:
                    # CSV Projection
                    if not csv_path.exists() or should_run_pipeline:
                        def capture_csv_log(msg):
                            result_queue.put(("log", msg))
                        run_json_to_csv(str(MEMORY_PATH), rule, capture_csv_log)
                        result_queue.put(("log", "Memory projection complete."))
                    else:
                        result_queue.put(("log", "Reusing existing CSV projection."))

                    # Feedback Generation
                    result_queue.put(("log", "Generating Feedback (this may take a moment)..."))
                    
                    feedback_text = ""
                    if rule == "temporal":
                        feedback_text = generate_temporal_feedback(str(csv_path))
                    elif rule == "role_completeness":
                        feedback_text = generate_role_feedback(str(csv_path))
                        
                    result_queue.put(("result", feedback_text))
                    
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    result_queue.put(("error", str(e)))

            post_thread = threading.Thread(target=run_post_processing)
            post_thread.start()
            
            while True:
                try:
                    msg_type, content = result_queue.get(timeout=0.5)
                    if msg_type == "log":
                        msg = send_log(content)
                        if msg:
                            yield msg
                    elif msg_type == "result":
                        # Format feedback
                        import re
                        html_text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", content)
                        html_text = re.sub(r'"(.*?)"', r"<b>\1</b>", html_text)
                        html_text = html_text.replace("\n", "<br>")
                        yield f"event: result\ndata: {json.dumps({'feedback': html_text})}\n\n"
                        break
                    elif msg_type == "error":
                        raise RuntimeError(content)
                except queue.Empty:
                    if not post_thread.is_alive():
                        break
                    yield ": keep-alive\n\n"

            msg = send_log("Analysis Complete!")
            if msg:
                yield msg

        except Exception as e:
            sys.stderr.write(f"Server Error: {e}\n")
            yield f"event: error_msg\ndata: Server Error: {str(e)}\n\n"
        finally:
            pipeline_lock.release()

    return StreamingResponse(analysis_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
