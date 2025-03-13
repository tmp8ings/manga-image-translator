import asyncio
import os
from typing import List, Optional

from PIL import Image
from fastapi import HTTPException
from fastapi.requests import Request

from manga_translator import Config
from server.instance import executor_instances
from server.sent_data_internal import NotifyType

class QueueElement:
    req: Request
    image: Image.Image | str
    zip_file: bytes | None = None
    config: Config

    def __init__(self, req: Request, image: Image.Image, config: Config, length, *, zip_file: bytes = None):
        self.zip_file = zip_file
        self.req = req
        if length > 10:
            #todo: store image in "upload-cache" folder
            self.image = image
        else:
            self.image = image
        self.config = config
        import time
        self.last_poll = time.time()  # initialize last_poll

    def update_poll(self):
        import time
        self.last_poll = time.time()

    def get_image(self)-> Image:
        if isinstance(self.image, str):
            return Image.open(self.image)
        else:
            return self.image

    def __del__(self):
        if isinstance(self.image, str):
            os.remove(self.image)

    async def is_client_disconnected(self) -> bool:
        if await self.req.is_disconnected():
            return True
        return False


class TaskQueue:
    def __init__(self):
        self.queue: List[QueueElement] = []
        self.queue_event: asyncio.Event = asyncio.Event()

    def add_task(self, task: QueueElement):
        self.queue.append(task)

    def get_pos(self, task: QueueElement) -> Optional[int]:
        try:
            return self.queue.index(task)
        except ValueError:
            return None
    async def update_event(self):
        self.queue = [task for task in self.queue if not await task.is_client_disconnected()]
        self.queue_event.set()
        self.queue_event.clear()

    async def remove(self, task: QueueElement):
        self.queue.remove(task)
        await self.update_event()

    async def wait_for_event(self):
        await self.queue_event.wait()

task_queue = TaskQueue()

async def wait_in_queue(task: QueueElement, notify: NotifyType):
    """Will get task position report it. If its in the range of translators then it will try to aquire an instance(blockig) and sent a task to it. when done the item will be removed from the queue and result will be returned"""
    while True:
        queue_pos = task_queue.get_pos(task)
        if queue_pos is None:
            if notify:
                return
            else:
                raise HTTPException(500, detail="User is no longer connected")  # just for the logs
        if notify:
            notify(3, str(queue_pos).encode('utf-8'))
        if queue_pos < executor_instances.free_executors():
            if await task.is_client_disconnected():
                await task_queue.update_event()
                if notify:
                    return
                else:
                    raise HTTPException(500, detail="User is no longer connected") #just for the logs

            instance = await executor_instances.find_executor()
            await task_queue.remove(task)
            if notify:
                notify(4, b"")
            if notify:
                await instance.sent_stream(task.image, task.config, notify, zip_file=task.zip_file)
            else:
                result = await instance.sent(task.image, task.config, zip_file=task.zip_file)

            await executor_instances.free_executor(instance)

            if notify:
                return
            else:
                return result
        else:
            await task_queue.wait_for_event()

async def polling_in_queue(task: QueueElement):
    import time
    while True:
        queue_pos = task_queue.get_pos(task)
        if queue_pos is None:
            raise HTTPException(500, detail=f"User is no longer connected - can't find task in queue")  # just for the logs
        # Check if no polling has occurred in the last 180 seconds
        if time.time() - getattr(task, "last_poll", 0) > 180:
            await task_queue.update_event()
            raise HTTPException(500, detail="User is no longer connected - too old polling")
        if queue_pos < executor_instances.free_executors():
            instance = await executor_instances.find_executor()
            await task_queue.remove(task)
            result = await instance.sent(task.image, task.config, zip_file=task.zip_file)
            await executor_instances.free_executor(instance)
            return result
        else:
            await task_queue.wait_for_event()