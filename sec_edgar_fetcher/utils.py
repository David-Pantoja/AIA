# Utility functions (rate limiting, error handling) 
import time
from functools import wraps
from ratelimit import limits, sleep_and_retry
from requests.exceptions import HTTPError
from backoff import on_exception, expo

# Rate limiting decorator: 10 requests per second
@sleep_and_retry
@limits(calls=10, period=1)
def rate_limited(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

# Exponential backoff retry decorator for HTTP errors
@on_exception(expo, HTTPError, max_tries=3)
def retry_on_http_error(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper 