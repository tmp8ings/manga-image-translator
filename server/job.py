import time
import os
import asyncio
from typing import Optional, Any
import uuid

class Job:
    def __init__(self, job_id: Optional[str] = None):
        self.job_id = job_id or str(uuid.uuid4())
        self.status = "pending"
        self.result = None
        self.error = None
        self.poll_task = None
        self.file_path = None
        self.file_size = 0
        self.created = time.time()
        self.finished = None
    
    def set_poll_task(self, poll_task: asyncio.Task):
        """Set the polling task for this job"""
        self.poll_task = poll_task
    
    def set_finished(self, file_path: str, file_size: int):
        """Mark job as finished with result file information"""
        self.status = "finished"
        self.file_path = file_path
        self.file_size = file_size
        self.finished = time.time()
    
    def set_error(self, error: str, file_path: Optional[str] = None, file_size: int = 0):
        """Mark job as failed with error information"""
        self.status = "error"
        self.error = error
        if file_path:
            self.file_path = file_path
            self.file_size = file_size
        self.finished = time.time()
    
    def update_poll(self):
        """Update the poll timestamp if queue element exists"""
        if self.poll_task and hasattr(self.poll_task, "queue_elem") and self.poll_task.queue_elem:
            self.poll_task.queue_elem.update_poll()
    
    def is_expired(self, expiry_time_seconds: int = 3600) -> bool:
        """Check if job is expired based on creation time"""
        return self.created < time.time() - expiry_time_seconds
    
    def cleanup(self):
        """Clean up any resources used by the job"""
        if self.file_path and os.path.exists(self.file_path):
            try:
                os.remove(self.file_path)
            except Exception:
                pass
    
    def to_readable_kst_time(self, timestamp: Optional[float] = None) -> str | None:
        if timestamp is None:
            return None
        
        """Convert timestamp to readable KST time"""
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp + 32400))
        # KST is UTC+9, so add 9 hours (32400 seconds) to the timestamp
        
    def to_dict(self):
        """Convert job to dictionary for API responses"""
        return {
            "status": self.status,
            "error": self.error,
            "created": self.to_readable_kst_time(self.created),
            "finished": self.to_readable_kst_time(self.finished)
        }
