import random
import time
from typing import Any, Callable, Iterable, Optional, Tuple, Type


def retry_call(
	func: Callable[..., Any],
	*args: Any,
	max_retries: int = 2,
	backoff_base_seconds: float = 0.5,
	backoff_multiplier: float = 2.0,
	max_backoff_seconds: float = 8.0,
	retry_on_exceptions: Tuple[Type[BaseException], ...] = (Exception,),
	jitter_fraction: float = 0.15,
	before_retry: Optional[Callable[[int, BaseException], None]] = None,
	**kwargs: Any,
) -> Any:
	"""Call a function with simple exponential backoff retries.

	Args:
		func: The callable to execute.
		*args: Positional arguments for the callable.
		max_retries: Number of retry attempts on failure (not counting the first attempt).
		backoff_base_seconds: Initial backoff delay.
		backoff_multiplier: Backoff multiplier per retry.
		max_backoff_seconds: Upper bound on backoff delay.
		retry_on_exceptions: Exception types that should trigger a retry.
		jitter_fraction: Random jitter to reduce thundering herd.
		before_retry: Optional callback invoked before each retry with (attempt_index, exception).
		**kwargs: Keyword arguments for the callable.

	Returns:
		The callable's return value.

	Raises:
		The last exception if all retries fail.
	"""

	attempt = 0
	while True:
		try:
			return func(*args, **kwargs)
		except retry_on_exceptions as exc:  # type: ignore[misc]
			if attempt >= max_retries:
				raise
			if before_retry is not None:
				before_retry(attempt + 1, exc)
			delay = min(
				backoff_base_seconds * (backoff_multiplier ** attempt),
				max_backoff_seconds,
			)
			jitter = 1.0 + random.uniform(-jitter_fraction, jitter_fraction)
			time.sleep(delay * jitter)
			attempt += 1

