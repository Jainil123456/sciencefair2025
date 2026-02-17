"""
Progress manager for tracking real-time training progress.
Handles job creation, progress updates, and cancellation.
"""

import threading
import queue
import uuid
import time
from datetime import datetime
from typing import Dict, Optional, Generator, Any


class ProgressManager:
    """Manages training job progress with real-time updates."""

    def __init__(self):
        """Initialize the progress manager."""
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.job_queues: Dict[str, queue.Queue] = {}
        self.lock = threading.Lock()
        self._last_cleanup = time.time()

    def create_job(self, num_epochs: int) -> str:
        """
        Create a new training job.

        Args:
            num_epochs: Total number of epochs for training

        Returns:
            Job ID (UUID string)
        """
        job_id = str(uuid.uuid4())

        with self.lock:
            self.jobs[job_id] = {
                'status': 'pending',
                'num_epochs': num_epochs,
                'current_epoch': 0,
                'created_at': datetime.now().isoformat(),
                'started_at': None,
                'completed_at': None,
                'error': None,
                'cancel_requested': False,
                'progress': 0.0
            }
            self.job_queues[job_id] = queue.Queue()

        return job_id

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get job information.

        Args:
            job_id: Job ID to retrieve

        Returns:
            Job dictionary or None if not found
        """
        with self.lock:
            return self.jobs.get(job_id)

    def update_progress(
        self,
        job_id: str,
        epoch: int,
        train_loss: float,
        val_loss: float,
        metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Update job progress (entire operation under lock for atomicity).

        Args:
            job_id: Job ID to update
            epoch: Current epoch number
            train_loss: Training loss
            val_loss: Validation loss
            metrics: Optional metrics dictionary
        """
        with self.lock:
            if job_id not in self.jobs:
                return

            job = self.jobs[job_id]
            job['current_epoch'] = epoch
            job['progress'] = (epoch / job['num_epochs']) * 100 if job['num_epochs'] > 0 else 0

            if job['status'] == 'pending':
                job['status'] = 'running'
                job['started_at'] = datetime.now().isoformat()

            # Build update dict while holding lock (atomic operation)
            update = {
                'job_id': job_id,
                'epoch': epoch,
                'num_epochs': job['num_epochs'],
                'train_loss': train_loss,
                'val_loss': val_loss,
                'progress': job['progress'],
                'metrics': metrics or {},
                'timestamp': datetime.now().isoformat()
            }

            # Queue update while holding lock
            try:
                if job_id in self.job_queues:
                    self.job_queues[job_id].put_nowait(update)
            except (KeyError, queue.Full):
                pass

    def complete_job(self, job_id: str, result: Optional[Dict[str, Any]] = None) -> None:
        """
        Mark job as completed.

        Args:
            job_id: Job ID to complete
            result: Optional result data
        """
        with self.lock:
            if job_id not in self.jobs:
                return

            job = self.jobs[job_id]
            job['status'] = 'completed'
            job['completed_at'] = time.time()
            job['progress'] = 100.0
            job['result'] = result or {}

        # Send completion event
        try:
            self.job_queues[job_id].put_nowait({
                'event': 'completed',
                'job_id': job_id,
                'timestamp': datetime.now().isoformat(),
                'result': result or {}
            })
        except KeyError:
            pass

        # Trigger cleanup if enough time has passed
        self._schedule_cleanup()

    def fail_job(self, job_id: str, error: str) -> None:
        """
        Mark job as failed.

        Args:
            job_id: Job ID that failed
            error: Error message
        """
        with self.lock:
            if job_id not in self.jobs:
                return

            job = self.jobs[job_id]
            job['status'] = 'failed'
            job['completed_at'] = time.time()
            job['error'] = error

        # Send error event
        try:
            self.job_queues[job_id].put_nowait({
                'event': 'error',
                'job_id': job_id,
                'error': error,
                'timestamp': datetime.now().isoformat()
            })
        except KeyError:
            pass

        # Trigger cleanup if enough time has passed
        self._schedule_cleanup()

    def request_cancellation(self, job_id: str) -> None:
        """
        Request job cancellation.

        Args:
            job_id: Job ID to cancel
        """
        with self.lock:
            if job_id in self.jobs:
                self.jobs[job_id]['cancel_requested'] = True

    def is_cancelled(self, job_id: str) -> bool:
        """
        Check if job cancellation was requested.

        Args:
            job_id: Job ID to check

        Returns:
            True if cancellation was requested
        """
        with self.lock:
            if job_id in self.jobs:
                return self.jobs[job_id]['cancel_requested']
        return False

    def iter_updates(
        self,
        job_id: str,
        timeout: int = 600
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Iterate over job progress updates (for SSE streaming).

        Args:
            job_id: Job ID to stream updates from
            timeout: Timeout in seconds for waiting for updates

        Yields:
            Progress update dictionaries
        """
        if job_id not in self.job_queues:
            return

        queue_obj = self.job_queues[job_id]
        heartbeat_count = 0

        while True:
            try:
                # Get next update with timeout
                update = queue_obj.get(timeout=1)
                heartbeat_count = 0
                yield update

                # Check if job is completed
                if update.get('event') == 'completed' or update.get('event') == 'error':
                    break

            except queue.Empty:
                # Send heartbeat to keep connection alive
                heartbeat_count += 1
                if heartbeat_count >= 10:  # Every 10 seconds
                    yield {'heartbeat': True}
                    heartbeat_count = 0

                # Check if total timeout exceeded
                job = self.get_job(job_id)
                if not job:
                    break

                # Timeout if job is completed and no more updates
                if job['status'] in ['completed', 'failed']:
                    break

    def cancel_job(self, job_id: str) -> None:
        """
        Cancel a training job.

        Args:
            job_id: Job ID to cancel
        """
        self.request_cancellation(job_id)

    def cleanup_old_jobs(self, max_age_seconds: int = 3600) -> None:
        """
        Remove completed/failed jobs older than max_age_seconds to prevent memory leaks.

        Args:
            max_age_seconds: Maximum age in seconds before cleanup (default 1 hour)
        """
        with self.lock:
            current_time = time.time()
            jobs_to_remove = []

            for job_id, job in list(self.jobs.items()):
                if job.get('status') in ['completed', 'failed', 'cancelled']:
                    completed_at = job.get('completed_at', time.time())
                    # Handle both float (time.time()) and isoformat strings
                    if isinstance(completed_at, str):
                        # Skip cleanup for isoformat strings (they're incomplete migrations)
                        continue
                    age = current_time - completed_at
                    if age > max_age_seconds:
                        jobs_to_remove.append(job_id)

            for job_id in jobs_to_remove:
                self.jobs.pop(job_id, None)
                self.job_queues.pop(job_id, None)

    def _schedule_cleanup(self, cleanup_interval: int = 300) -> None:
        """
        Schedule cleanup if enough time has passed since last cleanup.

        Args:
            cleanup_interval: Minimum seconds between cleanup runs (default 5 minutes)
        """
        current_time = time.time()
        if current_time - self._last_cleanup > cleanup_interval:
            self._last_cleanup = current_time
            self.cleanup_old_jobs()


# Global progress manager instance
progress_manager = ProgressManager()
