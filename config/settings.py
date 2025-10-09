from dotenv import dotenv_values
import os

conf = dotenv_values()

OPENAI_API_KEY = conf.get('OPENAI_API_KEY', '')
INDEX_PERSISTENCE_STORAGE_PATH = "volumes/indexes"
FILE_STORAGE_PATH = "volumes/storage"

EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "local")
MODEL_NAME = os.getenv("EMBED_MODEL", "text-embedding-3-small")

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
