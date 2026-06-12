import redis
import json
import functools
from recommender.config import REDIS_HOST, REDIS_PORT

# Create a connection
try:
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True)
    r.ping() # Check connection
    REDIS_AVAILABLE = True
except Exception:
    REDIS_AVAILABLE = False
    print("Redis not available. Caching disabled.")

def cache_response(ttl_seconds=300):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            if not REDIS_AVAILABLE:
                return await func(*args, **kwargs)
            
            # Simple Key Generation
            key_parts = [func.__name__]
            key_parts.extend([str(a) for a in args])
            key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
            key = ":".join(key_parts)
            
            # Check Cache
            cached = r.get(key)
            if cached:
                return json.loads(cached)
            
            # Run Function
            result = await func(*args, **kwargs)
            
            # Set Cache
            r.setex(key, ttl_seconds, json.dumps(result))
            return result
        return wrapper
    return decorator
