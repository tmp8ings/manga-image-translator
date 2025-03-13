import io
import os
import secrets
import shutil
import signal
import subprocess
import sys
import logging
import traceback
from argparse import Namespace
from typing import Dict
import uuid
import tempfile  # 추가: 임시 파일 처리를 위한 모듈
from fastapi import BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from concurrent.futures import ThreadPoolExecutor
import asyncio  # for running async functions synchronously in the thread
import time  # add at the top if not present
from starlette.background import BackgroundTask  # added import
import requests  # Added import for polling the share server

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, Request, HTTPException, Header, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from pathlib import Path

from manga_translator import Config
from server.instance import ExecutorInstance, executor_instances
from server.myqueue import task_queue
from server.request_extraction import (
    get_ctx,
    while_polling,
    while_streaming,
    TranslateRequest,
)
from server.to_json import to_translation, TranslationResponse
from server.job import Job  # Import the new Job class

import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("manga-translator")

app = FastAPI()
nonce = None

# Add middleware to log requests and responses
# @app.middleware("http")
# async def log_requests(request: Request, call_next):
#     try:
#         response = await call_next(request)
#         if response.status_code >= 400:
#             logger.error(f"HTTP Error {response.status_code}: {request.method} {request.url.path}")
#         # else:
#         #     logger.info(f"HTTP {response.status_code}: {request.method} {request.url.path}")
#         return response
#     except Exception as e:
#         logger.error(f"Unhandled exception in {request.method} {request.url.path}: {str(e)}")
#         logger.error(traceback.format_exc())
#         raise e

# Add exception handlers for different types of errors
# @app.exception_handler(StarletteHTTPException)
# async def http_exception_handler(request: Request, exc: StarletteHTTPException):
#     logger.error(f"HTTP {exc.status_code}: {exc.detail} at {request.method} {request.url.path}")
#     return {"detail": exc.detail}, exc.status_code

# @app.exception_handler(RequestValidationError)
# async def validation_exception_handler(request: Request, exc: RequestValidationError):
#     logger.error(f"Validation error at {request.method} {request.url.path}: {str(exc)}")
#     return {"detail": str(exc)}, 422


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(
        f"Unhandled exception at {request.method} {request.url.path}: {str(exc)}",
        exc_info=exc,
    )
    return {"detail": "Internal server error"}, 500


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/register", response_description="no response", tags=["internal-api"])
async def register_instance(
    instance: ExecutorInstance, req: Request, req_nonce: str = Header(alias="X-Nonce")
):
    if req_nonce != nonce:
        raise HTTPException(401, detail="Invalid nonce")
    instance.ip = req.client.host
    executor_instances.register(instance)


def transform_to_image(ctx):
    img_byte_arr = io.BytesIO()
    ctx.result.save(img_byte_arr, format="PNG")
    return img_byte_arr.getvalue()


def transform_to_json(ctx):
    return to_translation(ctx).model_dump_json().encode("utf-8")


def transform_to_bytes(ctx):
    return to_translation(ctx).to_bytes()


@app.post(
    "/translate/json",
    response_model=TranslationResponse,
    tags=["api", "json"],
    response_description="json strucure inspired by the ichigo translator extension",
)
async def json(req: Request, data: TranslateRequest):
    ctx = await get_ctx(req, data.config, data.image)
    return to_translation(ctx)


@app.post(
    "/translate/bytes",
    response_class=StreamingResponse,
    tags=["api", "json"],
    response_description="custom byte structure for decoding look at examples in 'examples/response.*'",
)
async def bytes(req: Request, data: TranslateRequest):
    ctx = await get_ctx(req, data.config, data.image)
    return StreamingResponse(content=to_translation(ctx).to_bytes())


@app.post(
    "/translate/image",
    response_description="the result image",
    tags=["api", "json"],
    response_class=StreamingResponse,
)
async def image(req: Request, data: TranslateRequest) -> StreamingResponse:
    ctx = await get_ctx(req, data.config, data.image)
    img_byte_arr = io.BytesIO()
    ctx.result.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)

    return StreamingResponse(img_byte_arr, media_type="image/png")


@app.post(
    "/translate/json/stream",
    response_class=StreamingResponse,
    tags=["api", "json"],
    response_description="A stream over elements with strucure(1byte status, 4 byte size, n byte data) status code are 0,1,2,3,4 0 is result data, 1 is progress report, 2 is error, 3 is waiting queue position, 4 is waiting for translator instance",
)
async def stream_json(req: Request, data: TranslateRequest) -> StreamingResponse:
    return await while_streaming(req, transform_to_json, data.config, data.image)


@app.post(
    "/translate/bytes/stream",
    response_class=StreamingResponse,
    tags=["api", "json"],
    response_description="A stream over elements with strucure(1byte status, 4 byte size, n byte data) status code are 0,1,2,3,4 0 is result data, 1 is progress report, 2 is error, 3 is waiting queue position, 4 is waiting for translator instance",
)
async def stream_bytes(req: Request, data: TranslateRequest) -> StreamingResponse:
    return await while_streaming(req, transform_to_bytes, data.config, data.image)


@app.post(
    "/translate/image/stream",
    response_class=StreamingResponse,
    tags=["api", "json"],
    response_description="A stream over elements with strucure(1byte status, 4 byte size, n byte data) status code are 0,1,2,3,4 0 is result data, 1 is progress report, 2 is error, 3 is waiting queue position, 4 is waiting for translator instance",
)
async def stream_image(req: Request, data: TranslateRequest) -> StreamingResponse:
    return await while_streaming(req, transform_to_image, data.config, data.image)


@app.post(
    "/translate/with-form/json",
    response_model=TranslationResponse,
    tags=["api", "form"],
    response_description="json strucure inspired by the ichigo translator extension",
)
async def json_form(
    req: Request, image: UploadFile = File(...), config: str = Form("{}")
):
    img = await image.read()
    ctx = await get_ctx(req, Config.parse_raw(config), img)
    return to_translation(ctx)


@app.post(
    "/translate/with-form/bytes",
    response_class=StreamingResponse,
    tags=["api", "form"],
    response_description="custom byte structure for decoding look at examples in 'examples/response.*'",
)
async def bytes_form(
    req: Request, image: UploadFile = File(...), config: str = Form("{}")
):
    img = await image.read()
    ctx = await get_ctx(req, Config.parse_raw(config), img)
    return StreamingResponse(content=to_translation(ctx).to_bytes())


@app.post(
    "/translate/with-form/image",
    response_description="the result image with logging",
    tags=["api", "form"],
    response_class=StreamingResponse,
)
async def image_form(req: Request, image: UploadFile = File(...), config: str = Form("{}")) -> StreamingResponse:
    img = await image.read()
    # Create logging folder based on current time
    folder = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    folder = os.path.join("logging", folder)
    os.makedirs(folder, exist_ok=True)
    # Save input image using original filename
    input_path = os.path.join(folder, image.filename)
    with open(input_path, "wb") as f:
        f.write(img)
    # Process translation
    ctx = await get_ctx(req, Config.parse_raw(config), img)
    # Obtain output image bytes and save to logging folder
    out_bytes = transform_to_image(ctx)
    output_filename = "translated_" + image.filename
    output_path = os.path.join(folder, output_filename)
    with open(output_path, "wb") as f:
        f.write(out_bytes)
    # Return the processed image
    img_byte_arr = io.BytesIO(out_bytes)
    img_byte_arr.seek(0)
    return StreamingResponse(img_byte_arr, media_type="image/png")


@app.post(
    "/translate/with-form/json/stream",
    response_class=StreamingResponse,
    tags=["api", "form"],
    response_description="A stream over elements with strucure(1byte status, 4 byte size, n byte data) status code are 0,1,2,3,4 0 is result data, 1 is progress report, 2 is error, 3 is waiting queue position, 4 is waiting for translator instance",
)
async def stream_json_form(
    req: Request, image: UploadFile = File(...), config: str = Form("{}")
) -> StreamingResponse:
    img = await image.read()
    return await while_streaming(req, transform_to_json, Config.parse_raw(config), img)


@app.post(
    "/translate/with-form/bytes/stream",
    response_class=StreamingResponse,
    tags=["api", "form"],
    response_description="A stream over elements with strucure(1byte status, 4 byte size, n byte data) status code are 0,1,2,3,4 0 is result data, 1 is progress report, 2 is error, 3 is waiting queue position, 4 is waiting for translator instance",
)
async def stream_bytes_form(
    req: Request, image: UploadFile = File(...), config: str = Form("{}")
) -> StreamingResponse:
    img = await image.read()
    return await while_streaming(req, transform_to_bytes, Config.parse_raw(config), img)


@app.post(
    "/translate/with-form/image/stream",
    response_class=StreamingResponse,
    tags=["api", "form"],
    response_description="A stream over elements with structure (1byte status, 4 byte size, n byte data) status codes: 0 is result data, 1 is progress report, 2 is error, 3 is waiting queue position, 4 is waiting for translator instance",
)
async def stream_image_form(
    req: Request, image: UploadFile = File(...), config: str = Form("{}")
) -> StreamingResponse:
    img = await image.read()
    # Create folder based on current time
    folder = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    folder = os.path.join("logging", folder)
    os.makedirs(folder, exist_ok=True)
    # Save input image using original filename
    input_path = os.path.join(folder, image.filename)
    with open(input_path, "wb") as f:
        f.write(img)

    # Define wrapper that saves the output image before returning it
    def transform_and_save(ctx):
        out_bytes = transform_to_image(ctx)
        output_filename = "translated_" + image.filename
        output_path = os.path.join(folder, output_filename)
        with open(output_path, "wb") as f:
            f.write(out_bytes)
        return out_bytes

    return await while_streaming(req, transform_and_save, Config.parse_raw(config), img)


@app.post("/queue-size", response_model=int, tags=["api", "json"])
async def queue_size() -> int:
    return len(task_queue.queue)


@app.get("/", response_class=HTMLResponse, tags=["ui"])
async def index() -> HTMLResponse:
    script_directory = Path(__file__).parent
    html_file = script_directory / "index.html"
    html_content = html_file.read_text(encoding="utf-8")
    return HTMLResponse(content=html_content)


@app.get("/manual", response_class=HTMLResponse, tags=["ui"])
async def manual():
    script_directory = Path(__file__).parent
    html_file = script_directory / "manual.html"
    html_content = html_file.read_text(encoding="utf-8")
    return HTMLResponse(content=html_content)


def generate_nonce():
    return secrets.token_hex(16)


def start_translator_client_proc(host: str, port: int, nonce: str, params: Namespace):
    cmds = [
        sys.executable,
        "-m",
        "manga_translator",
        "shared",
        "--host",
        host,
        "--port",
        str(port),
        "--nonce",
        nonce,
    ]
    if params.use_gpu:
        cmds.append("--use-gpu")
    if params.use_gpu_limited:
        cmds.append("--use-gpu-limited")
    if params.ignore_errors:
        cmds.append("--ignore-errors")
    if params.verbose:
        cmds.append("--verbose")
    if params.models_ttl:
        cmds.append("--models-ttl=%s" % params.models_ttl)
    if params.pre_dict:
        cmds.extend(["--pre-dict", params.pre_dict])
    if params.pre_dict:
        cmds.extend(["--post-dict", params.post_dict])
    base_path = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.dirname(base_path)
    proc = subprocess.Popen(cmds, cwd=parent)
    executor_instances.register(ExecutorInstance(ip=host, port=port))

    def handle_exit_signals(signal, frame):
        proc.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_exit_signals)
    signal.signal(signal.SIGTERM, handle_exit_signals)

    return proc


def prepare(args):
    global nonce
    if args.nonce is None:
        nonce = os.getenv("MT_WEB_NONCE", generate_nonce())
    else:
        nonce = args.nonce
    if args.start_instance:
        host = args.host
        port = args.port + 1
        return start_translator_client_proc(host, port, nonce, args), host, port
    folder_name = "upload-cache"
    if os.path.exists(folder_name):
        shutil.rmtree(folder_name)
    os.makedirs(folder_name)


jobs: Dict[str, Job] = {}  # Global dictionary for zip jobs now storing Job objects


async def process_zip(job_id: str, req: Request, image_bytes: bytes, config_str: str, orig_filename: str):
    try:
        job = jobs[job_id]
        config_obj = Config.parse_raw(config_str)
        # Create folder based on current time
        folder = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        folder = os.path.join("logging", folder)
        os.makedirs(folder, exist_ok=True)
        # Save original zip file using original filename
        original_path = os.path.join(folder, orig_filename)
        with open(original_path, "wb") as f:
            f.write(image_bytes)
        # Start while_polling in background so polling can update the task timestamp
        poll_task = asyncio.create_task(while_polling(req, config_obj, image_bytes))
        job.set_poll_task(poll_task)
        result, _ = await poll_task  # waiting for result from polling_in_queue
        output_filename = "translated_" + orig_filename
        output_path = os.path.join(folder, output_filename)
        with open(output_path, "wb") as f:
            f.write(result.result)
        # Save output file path and size
        job.set_finished(output_path, len(result.result))
    except Exception as e:
        logger.error(f"Error processing zip job {job_id}: {str(e)}", exc_info=True)
        job.set_error(str(e))


@app.post(
    "/translate/with-form/zip-submit",
    tags=["api", "form"],
    response_description="Submit zip processing job",
)
async def zip_submit(
    req: Request, image: UploadFile = File(...), config: str = Form("{}")
):
    logger.info(f"zip_submit start!")
    image_bytes = await image.read()
    logger.info(f"zip_submit image size: {len(image_bytes)}")
    job_id = str(uuid.uuid4())
    # Create a new Job object
    jobs[job_id] = Job(job_id)
    logger.info(f"Job {job_id} created")

    # Prune expired jobs
    current_time = time.time()
    for job in list(jobs.keys()):
        logger.debug(f"Checking job {job} for expiration")
        if jobs[job].is_expired():
            jobs[job].cleanup()  # Clean up any resources
            del jobs[job]

    logger.info(f"Job {job_id} is being processed")
    asyncio.create_task(process_zip(job_id, req, image_bytes, config, image.filename))

    logger.info(f"Job {job_id} submitted for processing")
    return JSONResponse(content={"job_id": job_id})


@app.delete(
    "/translate/with-form/zip-delete/{job_id}",
    tags=["api", "form"],
    response_description="Delete a zip job",
)
async def zip_delete(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, detail="Job not found")
    job.cleanup()  # Clean up resources before deleting
    del jobs[job_id]
    return JSONResponse(content={"detail": "Job deleted"})


@app.get(
    "/translate/with-form/zip-status/{job_id}",
    tags=["api", "form"],
    response_description="Check zip processing job status",
)
async def zip_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, detail="Job not found")
    if job.status == "pending":
        job.update_poll()
    return JSONResponse(content=job.to_dict())


@app.get(
    "/translate/with-form/zip-download/{job_id}",
    tags=["api", "form"],
    response_description="Download processed zip file",
)
async def zip_download(job_id: str, filename: str = None):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, detail="Job not found")
    if job.status != "finished":
        raise HTTPException(400, detail="Job not finished")

    # 임시 파일 경로 가져오기
    file_path = job.file_path
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(500, detail="File not available")

    # 파일명 생성 - 클라이언트에서 제공한 이름 사용 또는 기본값 생성
    if not filename:
        filename = f"translated-{job_id}.zip"
    else:
        # 안전한 파일명인지 확인하고 .zip 확장자를 갖도록 함
        filename = os.path.basename(filename)
        if not filename.endswith(".zip"):
            filename += ".zip"

    # Define a cleanup function to delete the job after response is sent
    def cleanup_job(job_id: str):
        time.sleep(10)  # Delay for 10 seconds before cleanup
        job = jobs.get(job_id)
        if job:
            job.cleanup()  # Clean up resources
            del jobs[job_id]

    # Return FileResponse with a BackgroundTask for cleanup
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="application/zip",
        background=BackgroundTask(cleanup_job, job_id),
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Length": str(job.file_size),
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


# todo: restart if crash
# todo: cache results
# todo: cleanup cache

if __name__ == "__main__":
    import uvicorn
    from args import parse_arguments

    args = parse_arguments()
    args.start_instance = True

    try:
        # Set Uvicorn log level and extend keep-alive timeout for large zip responses
        log_level = "debug" if args.verbose else "info"
        proc, host, port = prepare(args)

        # Wait until the translator (share.py) server is fully available
        health_url = f"http://{host}:{port}/health"
        while True:
            try:
                response = requests.get(health_url, timeout=1)
                if response.status_code == 200:
                    break
            except requests.exceptions.RequestException:
                pass
            time.sleep(1)

        print("Nonce: " + nonce)
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level=log_level,
            timeout_keep_alive=36000,  # increased timeout to handle long processing times
        )
    except Exception as e:
        logger.critical(f"Server crashed: {str(e)}")
        logger.critical(traceback.format_exc())
        if proc:
            proc.terminate()
