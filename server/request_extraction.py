import asyncio
import builtins
import io
import re
import zipfile  # added for zip file detection
from base64 import b64decode
from typing import Union

import requests
from PIL import Image
from fastapi import Request, HTTPException
from pydantic import BaseModel
from fastapi.responses import StreamingResponse

from manga_translator import Config
from manga_translator.utils.log import get_logger
from server.myqueue import task_queue, wait_in_queue, QueueElement
from server.streaming import notify, stream

logger = get_logger(__name__)


class TranslateRequest(BaseModel):
    """This request can be a multipart or a json request"""

    image: bytes | str
    """can be a url, base64 encoded image or a multipart image"""
    config: Config = Config()
    """in case it is a multipart this needs to be a string(json.stringify)"""


async def to_pil_image(image: Union[str, bytes]) -> Image.Image:
    try:
        if isinstance(image, builtins.bytes):
            image = Image.open(io.BytesIO(image))
            return image
        else:
            if re.match(r"^data:image/.+;base64,", image):
                value = image.split(",", 1)[1]
                image_data = b64decode(value)
                image = Image.open(io.BytesIO(image_data))
                return image
            else:
                response = requests.get(image)
                image = Image.open(io.BytesIO(response.content))
                return image
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


async def get_ctx(req: Request, config: Config, image: bytes):
    # Check if the input is a zip archive and extract the first image file if so
    if isinstance(image, bytes) and zipfile.is_zipfile(io.BytesIO(image)):
        logger.debug("Input is a zip file")
        image = Image.new("RGB", (1, 1), color=(255, 255, 255))
        zip_file = io.BytesIO(image)
    else:
        logger.debug(f"Input is not a zip file, {type(image)}, isinstance: {isinstance(image, bytes)}, zipfile: {zipfile.is_zipfile(io.BytesIO(image))}")
        image = await to_pil_image(image)
        zip_file = None
    task = QueueElement(req, image, config, 0, zip_file=zip_file)

    task_queue.add_task(task)

    return await wait_in_queue(task, None)


async def while_streaming(req: Request, transform, config: Config, image: bytes | str):
    image = await to_pil_image(image)
    task = QueueElement(req, image, config, 0)
    task_queue.add_task(task)

    messages = asyncio.Queue()

    def notify_internal(code: int, data: bytes) -> None:
        notify(code, data, transform, messages)

    streaming_response = StreamingResponse(
        stream(messages), media_type="application/octet-stream"
    )
    asyncio.create_task(wait_in_queue(task, notify_internal))
    return streaming_response
