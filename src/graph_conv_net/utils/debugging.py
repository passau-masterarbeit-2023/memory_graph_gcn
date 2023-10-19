import os

def dp(*args, **kwargs):
    """
    Debug print.
    """
    if os.environ.get('DEBUG') == '1':
        print(*args, **kwargs)