from datetime import datetime, timedelta
import json
import os
from enum import Enum
from typing import Type, TypeVar
import psutil

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
    enum_upper_to_key = {e.value.upper(): e for e in enum_cls}
    print(f"enum_upper_to_key: {enum_upper_to_key}")

    if s.upper() not in enum_upper_to_key.keys():
        raise ValueError(f"Invalid value for enum {enum_cls}: {s}")
    else:
        return enum_upper_to_key[s.upper()]

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

def check_memory():
    # Get the memory usage in GB
    memory_info = psutil.virtual_memory()
    used_memory_gb = (memory_info.total - memory_info.available) / (1024 ** 3)
    return used_memory_gb