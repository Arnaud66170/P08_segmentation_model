# src/utils/logger.py

from functools import wraps

def log_step(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"[LOG] ➤ {func.__name__} appelé")
        return func(*args, **kwargs)
    return wrapper
