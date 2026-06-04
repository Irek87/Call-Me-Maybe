import time
from functools import wraps
from typing import Any, Callable


def timer(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator that measures and prints a function's execution time.

    Args:
        func: Function to be wrapped.

    Returns:
        A wrapped callable with the same signature as `func` that prints the
        elapsed wall-clock time after execution.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_time = time.perf_counter() - start_time
        print("\n" + "="*42 + "\nElapsed time: " +
              f"{elapsed_time / 60:.0f} min " +
              f"{(elapsed_time % 60):.0f} sec\n")
        return result
    return wrapper
