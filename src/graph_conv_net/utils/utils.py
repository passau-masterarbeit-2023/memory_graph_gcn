from datetime import datetime
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
