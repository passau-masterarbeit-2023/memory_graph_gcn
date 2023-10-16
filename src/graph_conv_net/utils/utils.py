from datetime import datetime, timedelta
import json
import os
from enum import Enum
from typing import Type, TypeVar

# utils constants
DATETIME_FORMAT = "%Y_%m_%d_%H_%M_%S_%f"


def str2bool(string: str) -> bool:
    """
    Convert a string to a boolean.
    """
    return json.loads(string.lower())

T = TypeVar('T', bound=Enum)

def str2enum(s: str, enum_cls: Type[T]) -> T:
    """
    Convert a string to an enum member.
    """
    return enum_cls[s.upper()]

def datetime2str(datetime: datetime):
    """
    Return a string representation of the given datetime.
    NB: The %f is the microseconds.
    """
    return datetime.strftime(DATETIME_FORMAT)

def check_and_create_directory(dir_path: str):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Directory {dir_path} created.")
    else:
        print(f"Directory {dir_path} already exists.")

def datetime_to_human_readable_str(duration: timedelta) -> str:
    """
    Return a human readable string representation of the given datetime.
    """
    duration_in_seconds = duration.total_seconds()
    duration_hour_min_sec = "{} total sec ({:02d}h {:02d}m {:02d}s)".format(
        duration_in_seconds,
        int(duration_in_seconds // 3600),
        int((duration_in_seconds % 3600) // 60),
        int(duration_in_seconds % 60),
    )
    return duration_hour_min_sec
