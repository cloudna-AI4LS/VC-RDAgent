# VCAP-RDAgent · rare-disease-chat

MCP (Model Context Protocol) server for rare disease diagnosis, phenotype analysis, and related tools (chat / Web UI subproject).

> **Note**: In this document, `$PROJECT_ROOT` refers to the project root directory (this directory `rare-disease-chat`). Set these environment variables before use:
> ```bash
> export PROJECT_ROOT=<path-to-project-root>
> export MCP_SIMPLE_TOOL_DIR=$PROJECT_ROOT/mcp-server/mcp_simple_tool
> ```

## Quick Start

Choose one of two installation methods: **Option A (Docker)** or **Option B (Local)**. Both support terminal Chat; **Web UI** is documented separately and works with either.

---

### Option A: Docker install and run

In Docker mode, MCP server and Chat run in containers; Chat supports terminal (CLI) only—no Web UI.

The project uses two separate Docker images:
- **MCP server image** (`rarellm-mcp-server:latest`) – provides tool services
- **Chat system image** (`rarellm-chat-system:latest`) – intelligent chat system

#### Dockerfile overview

**`Dockerfile`** in the `mcp-server` directory:

- **Base image**: `python:3.12-slim`
- **Features**:
  - Build from scratch, copy entire project directory (`COPY . .`)
  - Install project dependencies with `uv pip install -e .`, using Tsinghua/Aliyun mirrors for speed
  - Install CPU-only torch to avoid CUDA and reduce image size
  - Install system dependencies (build-essential, curl, git, etc.)
- **Use case**: Suitable for local development and production; Docker Compose uses this Dockerfile by default

For MCP server API usage and troubleshooting, see [mcp-server/README.md](mcp-server/README.md).

#### Build images

```bash
cd $PROJECT_ROOT

# Build MCP server image
docker-compose build mcp-server

# Build chat system image
docker-compose build chat-system

# Or build all images at once
docker-compose build
```

#### Using Docker Compose (recommended)

Docker Compose will:
- Build images automatically (if missing or code has changed)
- Run containers with the built images
- Mount local directories via volumes so code changes take effect immediately

```bash
cd $PROJECT_ROOT

# Force rebuild (e.g. after code or dependency updates)
docker-compose build mcp-server
docker-compose build chat-system
# Or
docker-compose build --no-cache

# Start MCP server (builds image if missing)
docker-compose up -d mcp-server

# View MCP server logs
docker-compose logs -f mcp-server

# Start chat system (interactive)
docker-compose run --rm chat-system

# Or run chat system in background (not recommended; it needs interaction)
# docker-compose up -d chat-system

# Stop all services
docker-compose down
```

#### Using Docker commands (with pre-built images)

If images are already built, you can start them with `docker run`:

**Notes**:
- With `docker run`, create the network first; both containers must join the same network to communicate
- **Important**: `MCP_ENDPOINT` must use the container name `rarellm-mcp-server` (not the service name `mcp-server`), because Docker resolves container names for DNS
- Confirm the MCP server is running before starting the chat system (e.g. `docker logs rarellm-mcp-server`)

```bash
# 1. Create network (if it does not exist)
docker network create rarellm-network

# 2. Start MCP server (using built image, attach to network)
docker run -d \
  --name rarellm-mcp-server \
  --network rarellm-network \
  -p 3000:3000 \
  -e MCP_SIMPLE_TOOL_DIR=/app/mcp_simple_tool \
  -e PORT=3000 \
  -e HF_ENDPOINT=https://hf-mirror.com \
  -v $PROJECT_ROOT/mcp-server/mcp_simple_tool/data/model_weight:/app/mcp-server/mcp_simple_tool/data/model_weight \
  rarellm-mcp-server:latest

# 3. View MCP server logs
docker logs -f rarellm-mcp-server

# 4. Wait for MCP server to be ready (optional but recommended)
# Method 1: Check if container is running
docker ps | grep rarellm-mcp-server && echo "✓ MCP server container running"

# Method 2: Check server (tools/list)
# If response contains "result", server is OK
docker exec rarellm-mcp-server curl -s -X POST http://localhost:3000/mcp/ \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' | head -c 100
echo ""  # newline

# 5. Run chat system (interactive; MCP server must be running, same network)
# Note: MCP_ENDPOINT must use container name rarellm-mcp-server, not service name
docker run -it --rm \
  --name rarellm-chat-system \
  --network rarellm-network \
  -e MCP_ENDPOINT=http://rarellm-mcp-server:3000/mcp/ \
  -e MCP_TIMEOUT=600 \
  -e HF_ENDPOINT=https://hf-mirror.com \
  rarellm-chat-system:latest

# 6. Stop and clean up
docker stop rarellm-mcp-server rarellm-chat-system 
docker rm rarellm-mcp-server rarellm-chat-system 
docker network rm rarellm-network  # optional: remove network
```

**CLI commands**: type `quit` to exit, `clear` to clear chat history.

---

### Option B: Local install and run

In local mode, MCP server and Chat run on the host; Chat supports both terminal (CLI) and Web UI.

#### Start the server

> **Note**: In this document, `$PROJECT_ROOT` refers to the project root (this directory `rare-disease-chat`). Set these environment variables before use:
> ```bash
> export PROJECT_ROOT=<path-to-project-root>
> export MCP_SIMPLE_TOOL_DIR=$PROJECT_ROOT/mcp-server/mcp_simple_tool
> ```

1. **Go to project directory**
   ```bash
   cd $PROJECT_ROOT
   ```

2. **Go to MCP server directory**
   ```bash
   cd $PROJECT_ROOT/mcp-server
   ```

3. **Activate virtual environment** (`.venv` is under `mcp-server/`)
   ```bash
   source .venv/bin/activate
   ```
   
   If the venv does not exist, create it and install dependencies from `mcp-server/pyproject.toml`. For **CPU-only** (no GPU), install PyTorch CPU first so `sentence-transformers` does not pull CUDA:
   ```bash
   uv venv
   source .venv/bin/activate
   uv pip install torch -i https://download.pytorch.org/whl/cpu
   uv pip install -e .
   ```
   
   If the venv already exists but dependencies are missing, for CPU-only run:
   ```bash
   uv pip install torch -i https://download.pytorch.org/whl/cpu
   uv pip install -e .
   ```

4. **Run the start script**
   ```bash
   ./start_server.sh
   ```

The server runs at `http://0.0.0.0:3000` by default.

### Stop the server

From the `mcp-server` directory:
```bash
./stop_server.sh
```

### Start the chat system

Ensure the MCP server is running first (see "Start the server" above).

1. **Go to chat-system and activate virtual environment**
   ```bash
   cd $PROJECT_ROOT/chat-system
   source .venv/bin/activate
   ```
   
   If the venv does not exist, create it and install dependencies (`uv` uses `pyproject.toml` in the current directory):
   ```bash
   cd $PROJECT_ROOT/chat-system
   uv venv
   source .venv/bin/activate
   uv pip install -e .
   ```

2. **Choose one of two modes**:

   **Option A: Web UI** (recommended)
   ```bash
   ./start_web_ui.sh
   ```
   Or: `python web_ui_api.py` (Chinese) / `python web_ui_api_en.py` (English).  
   Open http://localhost:8080 in your browser. See [chat-system/WEB_UI_README.md](chat-system/WEB_UI_README.md) for details.

   **Option B: CLI (terminal chat)**
   ```bash
   python phenotype_to_disease_controller_langchain_stream_api.py
   ```
   Type `quit` to exit, `clear` to clear chat history.

After startup, the system supports:
- Phenotype extraction and symptom analysis
- Disease diagnosis and case extraction
- Disease information retrieval and normalization
- Multi-turn conversation memory

---

### Web UI

Web UI runs on your machine and connects to MCP server at `http://localhost:3000`. **Prerequisites**: MCP server must be running (Docker or local, port 3000).  
See "Start the chat system" > Option A above for Web UI setup and run instructions.

## Requirements

- Python >= 3.10
- `uv` is recommended as the package manager

## Environment variables

Set the following before using the project:

### Required

- **`PROJECT_ROOT`**: Path to the project root (this directory `rare-disease-chat`)
  ```bash
  export PROJECT_ROOT=<path-to-project-root>
  ```

- **`MCP_SIMPLE_TOOL_DIR`**: Full path to the `mcp_simple_tool` directory
  ```bash
  export MCP_SIMPLE_TOOL_DIR=$PROJECT_ROOT/mcp-server/mcp_simple_tool
  ```

### Optional

- **`PORT`**: MCP server port (default: 3000)
  ```bash
  export PORT=3000
  ```

- **`MCP_ENDPOINT`**: MCP server URL (default: `http://localhost:3000/mcp/`)
  ```bash
  export MCP_ENDPOINT=http://localhost:3000/mcp/
  ```

### Variables in config files

The config file `prompt_config_forKG.json` supports environment variable substitution:
- `${VAR_NAME}`: use env var (recommended)
- `$VAR_NAME`: use env var (short form)

Example:
```json
{
  "base_path": "${MCP_SIMPLE_TOOL_DIR}"
}
```

Environment variables are replaced with actual paths when the config is loaded.

### LLM configuration (start scripts)

Both `mcp-server/start_server.sh` and `chat-system/start_web_ui.sh` write LLM settings into their config files before starting. Edit the variables at the top of each script to point to your inference server:

| Variable | Description | Example |
|----------|-------------|---------|
| `LLM_MODEL` | Model name | `Qwen/Qwen3-8B` |
| `LLM_BASE_URL` | Inference API base URL | `http://192.168.0.127:8000/v1` |
| `LLM_API_KEY` | API key (use `EMPTY` for local) | `EMPTY` |
| `LLM_MODEL_PROVIDER` | Provider (empty for OpenAI-compatible API) | *(empty)* |
| `LLM_TEMPERATURE` | Sampling temperature | `0.1` |
| `LLM_TOP_P` | Top-p sampling | `0.95` |
| `LLM_STREAMING` | Enable streaming | `true` |

- **MCP server** (local run): values are written by `start_server.sh` to `mcp_simple_tool/scripts/rare_disease_diagnose/prompt_config_forKG.json`
- **Web UI** (local run): values are written by `start_web_ui.sh` to `chat-system/inference_config.json`

**When not using the start scripts** (e.g. Docker or direct run), you must edit the config files manually:
- **Docker MCP server**: edit `prompt_config_forKG.json` before building/starting the container
- **Terminal chat** (Docker or local): edit `chat-system/inference_config.json` before building/starting the chat system

## Project structure

```
rare-disease-chat/               # Project root (this README)
├── README.md                    # This file
├── docker-compose.yml           # Docker Compose config
├── mcp-server/                  # MCP server
│   ├── README.md                # MCP server API and usage
│   ├── Dockerfile               # MCP server image
│   ├── mcp_simple_tool/         # Core tool module
│   │   ├── tools/               # Tool implementations
│   │   ├── scripts/             # Helper scripts
│   │   │   └── rare_disease_diagnose/  # Rare disease diagnosis scripts
│   │   │       ├── vc_ranker.py                      # Ensemble disease ranking (Z-statistics)
│   │   │       ├── query_kg.py                       # KG query & phenotype-to-disease prompt generation
│   │   │       ├── disease_scraper/                  # Disease info scraper
│   │   │       └── prompt_config_forKG.json         # KG prompt config
│   │   └── data/                # Data files
│   ├── start_server.sh          # Start script
│   └── stop_server.sh           # Stop script
├── chat-system/                 # Chat system
│   ├── README.md                # Chat system overview; see parent README for full overview
│   ├── Dockerfile               # Chat system image
│   ├── phenotype_to_disease_controller_langchain_stream_api.py  # Core controller (CLI chat)
│   ├── web_ui_api.py            # Web UI backend (Chinese)
│   ├── web_ui_api_en.py         # Web UI backend (English)
│   ├── start_web_ui.sh          # Web UI startup script
│   ├── inference_config.json    # LLM inference config
│   ├── main.py                  # Entry point
│   ├── pyproject.toml           # Dependencies
│   ├── rdagent/                 # Web UI frontend
│   │   ├── index.html           # Chinese frontend
│   │   └── index_en.html        # English frontend
│   └── WEB_UI_README.md         # Web UI guide
└── ...
```

## Scripts

The `mcp_simple_tool/scripts/` directory contains helper scripts for rare disease diagnosis:

### rare_disease_diagnose/

Tools for rare disease diagnosis:

- **`vc_ranker.py`**: Ensemble script that uses multiple case-extraction strategies to get candidate diseases, aggregates and ranks them with Z-statistics, and outputs the top 50 candidates.

- **`query_kg.py`**: Knowledge-graph query and phenotype-to-disease prompt generation; supports various case-extraction methods (overlap, embedding, or both).

- **`disease_scraper/`**: Disease information scraper
  - `disease_scraper.py`: Scrapes disease info from HPO (Human Phenotype Ontology)
  - `batch_disease_scraper.py`: Batch scraping script

- **`prompt_config_forKG.json`**: Knowledge-graph prompt configuration

These scripts are mainly for data processing, model training, and evaluation; they are not required for normal server operation.

## Logs

Server logs are written to `mcp-server/api_server.log`. To tail them from the project root:

```bash
tail -f mcp-server/api_server.log
```

## Notes

### MCP server
- Ensure port 3000 is free, or set `PORT` to another port
- The server runs in the background; its PID is stored in `api_server.pid`
- To restart, run `./stop_server.sh` first, then start again

### Chat system
- **Start the MCP server first**; the chat system depends on its tool API
- Default MCP URL is `http://localhost:3000/mcp/`; override with `MCP_ENDPOINT`
- The system uses a LangGraph-based multi-agent setup with autonomous tool selection and invocation
- Chat history is managed automatically; old messages are trimmed when token limits are exceeded
