import redis
import json
from app.core.config import settings

class Cache:
    def __init__(self):
        try:
            self.client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB,
                decode_responses=True
            )
            self.client.ping()
            self.available = True
            print("Redis connected successfully.")
        except Exception as e:
            self.available = False
            print(f"Redis not available: {e}. Gracefully disabling cache.")

    def get(self, key):
        if not self.available:
            return None
        data = self.client.get(key)
        if data:
            return json.loads(data)
        return None

    def set(self, key, value, expire=3600):
        if not self.available:
            return
        self.client.setex(key, expire, json.dumps(value))

cache = Cache()
