#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2025 Fisher. All rights reserved.
#   
#   文件名称：async.py
#   创 建 者：YuLianghua
#   创建日期：2025年02月23日
#   描    述：
#
#================================================================

import asyncio
import weakref
from functools import partial
from typing import Callable

class AsyncTaskManagerError(Exception):
    """Custom exception for AsyncTaskManager errors."""
    pass

class AsyncTaskManager:
    def __init__(self):
        self._is_running = False
        self._errored = False
        self._error_with = None
        self._task = None
        self._protected_task = None
        self._error_callback = None

    @property
    def is_running(self) -> bool:
        """Check if the background task is running."""
        return self._is_running

    @property
    def errored(self) -> bool:
        """Check if the background task has encountered an error."""
        return self._errored

    def _log_task_completion(self, task: asyncio.Task, error_callback: Callable):
        """Log the completion of the task and handle errors."""
        try:
            # Get the result of the task (this will raise an exception if the task failed)
            result = task.result()
            print(f"Task completed successfully: {result}")
        except asyncio.CancelledError:
            print("Task was cancelled.")
        except Exception as e:
            print(f"Task failed with error: {e}")
            if error_callback:
                error_callback(e)
            self._errored = True
            self._error_with = e

    async def _run_periodic_task(self, interval: float, task_func: Callable):
        """Run a periodic task."""
        while True:
            try:
                await task_func()
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._errored = True
                self._error_with = e
                raise

    def start(self, interval: float, task_func: Callable, error_callback: Callable):
        """Start the background periodic task."""
        if self.errored:
            raise AsyncTaskManagerError("Task has already errored.") from self._error_with
        if self.is_running:
            raise RuntimeError("Task is already running.")

        self._is_running = True
        self._error_callback = error_callback

        # Create the task
        self._task = asyncio.get_event_loop().create_task(
            self._run_periodic_task(interval, task_func)
        )
        self._task.add_done_callback(
            partial(self._log_task_completion, error_callback=error_callback)
        )
        self._protected_task = asyncio.shield(self._task)

    def stop(self):
        """Stop the background task."""
        if self._task:
            self._task.cancel()
            self._is_running = False
            print("Task stopped.")

    async def wait_for_task_to_finish(self):
        """Wait for the task to finish."""
        if self._task:
            await self._task

# Example usage
async def example_task():
    print("Task is running...")
    # Simulate some work
    await asyncio.sleep(1)
    return "Task completed"

def error_handler(error):
    print(f"Error occurred: {error}")

async def main():
    manager = AsyncTaskManager()
    manager.start(interval=2, task_func=example_task, error_callback=error_handler)
    await asyncio.sleep(10)  # Let the task run for 10 seconds
    manager.stop()
    await manager.wait_for_task_to_finish()

if __name__ == "__main__":
    asyncio.run(main())
