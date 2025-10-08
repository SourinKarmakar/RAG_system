# Scalable RAG System for Technical documents

## Overview

This Project implements a Retrieval Augmented Generation (RAG) backend pipeline capable of ingesting PDF, DOCX and TXT files, performing context aware chunking, supporting dual mode search - Quick search and Deep Search.

### Core features

- Context-aware chunking (header-based)
- Multi-format document ingestion
- Local and OpenAI embedding support
- FAISS Vector search with persistence
- BM 25 Keyword Ranking
- Redis caching with request coalescing
- Modular FASTAPI endpoints

## Setup Instructions

1. Clone this repository `git clone <repoURL>`
2. `cd path_to_folder`
3. `pipenv --python 3.x`
4. `pipenv install`
5. `pipenv run python app.py`
6. Check if server is running by checking `http://127.0.0.1:8000/docs/` (Swagger UI in FastAPI)

### Setup Redis

1. If you are using windows, download Redis for Windows from `https://github.com/tporadowski/redis/releases`
2. Extract the downloaded ZIP to a convenient location like: `C:\redis`
3. You should see this directory structure inside the extracted folder

    ```shell
    C:\\Redis
        |-- redis-server.exe
        |-- redis-cli.exe
        |-- redis.windows.conf
    ```

4. Navigate to redis `cd C:\Redis` and run server `redis-server.exe`
5. Test Redis by opening another command prompt and then navigate to redis folder `cd C:\Redis`. Run `redis-cli.exe` and then type `ping`, it should respond `PONG`.
6. WSL based installation guide here - `https://redis.io/docs/latest/operate/oss_and_stack/install/archive/install-redis/install-redis-on-windows/`
7. For linux setup refer here -- `https://redis.io/docs/latest/operate/oss_and_stack/install/archive/install-redis/install-redis-on-linux/`

## TODO: Setting up using docker compose

1. Install `docker` and `docker compose`
2. Run `sudo docker compose up --build`
3. This will setup all the required services including `redis`, `weaviate` or `faiss`, `nginx` etc. with required volumes and dependencies.
