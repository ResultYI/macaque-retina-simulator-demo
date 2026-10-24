"""Bounded Windows sharing-violation retry for atomic artifact replacement."""
import os
import time
import re
from pathlib import Path


def execution_name(base,label=''):
    if label and not re.fullmatch(r'[a-z][a-z0-9_-]{0,31}',label):
        raise ValueError('Execution label must be a short lowercase identifier, not a path')
    return base+('_'+label if label else '')


def replace_with_retry(source, destination, *, max_wait_seconds=2., retry_seconds=.02):
    """Keep the previous complete file while a transient reader denies DELETE.

    Retry only PermissionError. Other failures and persistent permission errors
    propagate; the operation never substitutes a non-atomic write.
    """
    deadline=time.monotonic()+max_wait_seconds
    while True:
        try:
            os.replace(source,destination)
            return
        except PermissionError:
            remaining=deadline-time.monotonic()
            if remaining<=0:raise
            time.sleep(min(retry_seconds,remaining))


def read_bytes_with_retry(path,*,max_wait_seconds=2.,retry_seconds=.02):
    """Read complete committed bytes after a transient Windows sharing lock.

    Retry only PermissionError. Missing files, invalid content and identity or
    JSON failures remain the caller's errors; persistent locks still propagate.
    """
    deadline=time.monotonic()+max_wait_seconds
    while True:
        try:return Path(path).read_bytes()
        except PermissionError:
            remaining=deadline-time.monotonic()
            if remaining<=0:raise
            time.sleep(min(retry_seconds,remaining))
