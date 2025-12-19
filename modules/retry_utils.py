"""
Retry & Error Handling Utilities
Bulletproof error handling for production
"""

import time
import logging
import functools
import asyncio
from typing import Callable, TypeVar, Any, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

T = TypeVar('T')


class RetryConfig:
    """Retry configuration"""
    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: Tuple = (Exception,),
    ):
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions


def retry(config: Optional[RetryConfig] = None):
    """
    Retry decorator with exponential backoff
    
    Usage:
        @retry()
        def my_function():
            ...
        
        @retry(RetryConfig(max_retries=5))
        def my_function():
            ...
    """
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(config.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except config.retryable_exceptions as e:
                    last_exception = e
                    
                    if attempt == config.max_retries:
                        logger.error(f"[RETRY] {func.__name__} failed after {config.max_retries + 1} attempts: {e}")
                        raise
                    
                    # Calculate delay with exponential backoff
                    delay = min(
                        config.initial_delay * (config.exponential_base ** attempt),
                        config.max_delay
                    )
                    
                    # Add jitter
                    if config.jitter:
                        import random
                        delay = delay * (0.5 + random.random())
                    
                    logger.warning(f"[RETRY] {func.__name__} attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s...")
                    time.sleep(delay)
            
            raise last_exception
        
        return wrapper
    return decorator


def async_retry(config: Optional[RetryConfig] = None):
    """
    Async retry decorator with exponential backoff
    """
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(config.max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except config.retryable_exceptions as e:
                    last_exception = e
                    
                    if attempt == config.max_retries:
                        logger.error(f"[ASYNC_RETRY] {func.__name__} failed after {config.max_retries + 1} attempts: {e}")
                        raise
                    
                    delay = min(
                        config.initial_delay * (config.exponential_base ** attempt),
                        config.max_delay
                    )
                    
                    if config.jitter:
                        import random
                        delay = delay * (0.5 + random.random())
                    
                    logger.warning(f"[ASYNC_RETRY] {func.__name__} attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s...")
                    await asyncio.sleep(delay)
            
            raise last_exception
        
        return wrapper
    return decorator


class CircuitBreaker:
    """
    Circuit breaker pattern for fault tolerance
    
    States:
    - CLOSED: Normal operation
    - OPEN: Failing, reject all calls
    - HALF_OPEN: Testing if service recovered
    """
    
    CLOSED = 'closed'
    OPEN = 'open'
    HALF_OPEN = 'half_open'
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 3,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        
        self.state = self.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.half_open_calls = 0
    
    def can_execute(self) -> bool:
        """Check if execution is allowed"""
        if self.state == self.CLOSED:
            return True
        
        if self.state == self.OPEN:
            # Check if recovery timeout passed
            if self.last_failure_time:
                elapsed = (datetime.now() - self.last_failure_time).total_seconds()
                if elapsed >= self.recovery_timeout:
                    self.state = self.HALF_OPEN
                    self.half_open_calls = 0
                    logger.info("[CIRCUIT] State: HALF_OPEN - Testing recovery")
                    return True
            return False
        
        if self.state == self.HALF_OPEN:
            return self.half_open_calls < self.half_open_max_calls
        
        return False
    
    def record_success(self):
        """Record successful execution"""
        if self.state == self.HALF_OPEN:
            self.half_open_calls += 1
            if self.half_open_calls >= self.half_open_max_calls:
                self.state = self.CLOSED
                self.failure_count = 0
                logger.info("[CIRCUIT] State: CLOSED - Service recovered")
        else:
            self.failure_count = 0
    
    def record_failure(self):
        """Record failed execution"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.state == self.HALF_OPEN:
            self.state = self.OPEN
            logger.warning("[CIRCUIT] State: OPEN - Failed during recovery")
        elif self.failure_count >= self.failure_threshold:
            self.state = self.OPEN
            logger.warning(f"[CIRCUIT] State: OPEN - {self.failure_count} failures")


def circuit_breaker(breaker: CircuitBreaker):
    """
    Circuit breaker decorator
    
    Usage:
        breaker = CircuitBreaker()
        
        @circuit_breaker(breaker)
        def my_function():
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            if not breaker.can_execute():
                raise CircuitBreakerOpenError(f"Circuit breaker is OPEN for {func.__name__}")
            
            try:
                result = func(*args, **kwargs)
                breaker.record_success()
                return result
            except Exception as e:
                breaker.record_failure()
                raise
        
        return wrapper
    return decorator


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open"""
    pass


def safe_execute(func: Callable[..., T], *args, default: T = None, **kwargs) -> T:
    """
    Execute function safely, return default on error
    
    Usage:
        result = safe_execute(risky_function, arg1, arg2, default="fallback")
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.warning(f"[SAFE] {func.__name__} failed: {e}. Returning default.")
        return default


class ErrorTracker:
    """
    Track errors for monitoring
    """
    
    def __init__(self, max_errors: int = 100):
        self.errors = []
        self.max_errors = max_errors
        self.error_counts = {}
    
    def record(self, error: Exception, context: str = ""):
        """Record an error"""
        error_type = type(error).__name__
        
        self.errors.append({
            'time': datetime.now().isoformat(),
            'type': error_type,
            'message': str(error)[:200],
            'context': context,
        })
        
        # Trim old errors
        if len(self.errors) > self.max_errors:
            self.errors = self.errors[-self.max_errors:]
        
        # Count by type
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
    
    def get_summary(self) -> dict:
        """Get error summary"""
        return {
            'total_errors': len(self.errors),
            'error_counts': self.error_counts,
            'recent_errors': self.errors[-10:],
        }
    
    def clear(self):
        """Clear all errors"""
        self.errors = []
        self.error_counts = {}


# Global error tracker
error_tracker = ErrorTracker()


def track_errors(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to track errors
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> T:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_tracker.record(e, context=func.__name__)
            raise
    
    return wrapper


# === EXCHANGE-SPECIFIC RETRY CONFIGS ===

# For Binance API calls
BINANCE_RETRY = RetryConfig(
    max_retries=3,
    initial_delay=1.0,
    max_delay=30.0,
    retryable_exceptions=(
        ConnectionError,
        TimeoutError,
        Exception,  # ccxt throws generic exceptions
    ),
)

# For database operations
DATABASE_RETRY = RetryConfig(
    max_retries=5,
    initial_delay=0.5,
    max_delay=10.0,
)

# For external APIs (OpenAI, etc)
API_RETRY = RetryConfig(
    max_retries=3,
    initial_delay=2.0,
    max_delay=60.0,
)


# === EXAMPLE USAGE ===
if __name__ == "__main__":
    # Test retry decorator
    @retry(BINANCE_RETRY)
    def test_function():
        import random
        if random.random() < 0.7:
            raise ConnectionError("Simulated failure")
        return "Success!"
    
    try:
        result = test_function()
        print(f"Result: {result}")
    except Exception as e:
        print(f"Final error: {e}")
    
    # Test circuit breaker
    breaker = CircuitBreaker(failure_threshold=3)
    
    @circuit_breaker(breaker)
    def failing_function():
        raise ValueError("Always fails")
    
    for i in range(5):
        try:
            failing_function()
        except (ValueError, CircuitBreakerOpenError) as e:
            print(f"Attempt {i+1}: {e}")
    
    # Test safe execute
    def risky():
        raise RuntimeError("Oops")
    
    result = safe_execute(risky, default="fallback")
    print(f"Safe result: {result}")
    
    # Show error summary
    print(f"\nError summary: {error_tracker.get_summary()}")
