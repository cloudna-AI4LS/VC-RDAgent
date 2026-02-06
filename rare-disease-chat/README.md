# VC-RDAgent · rare-disease-chat

MCP (Model Context Protocol) server for rare disease diagnosis, phenotype analysis, and related tools (chat / Web UI subproject).

## Installation

**Requirements:** Python >= 3.10; `uv` is recommended.

**MCP server:**

```bash
cd mcp-server
uv venv && source .venv/bin/activate
uv pip install torch -i https://download.pytorch.org/whl/cpu   # for CPU-only
uv pip install -e .
```

**Chat system:**

```bash
cd chat-system
uv venv && source .venv/bin/activate
uv pip install -e .
```

## Configuration

**`set_config.sh`** has two sections: **MCP** (`MCP_LLM_*`) and **chat-system** (`CHAT_LLM_*`). They can use different values.

At a minimum, you should customize:

- **`MCP_LLM_MODEL` / `CHAT_LLM_MODEL`**: which model to call (e.g. local Qwen or a remote API model)
- **`MCP_LLM_BASE_URL` / `CHAT_LLM_BASE_URL`**: inference API base URL (your local gateway or cloud endpoint)
- **`MCP_LLM_API_KEY` / `CHAT_LLM_API_KEY`**: API key (`EMPTY` if your endpoint does not require a key)

Optionally, you can also tune **temperature**, **top_p**, and **streaming**:

- `MCP_LLM_TEMPERATURE` / `CHAT_LLM_TEMPERATURE`
- `MCP_LLM_TOP_P` / `CHAT_LLM_TOP_P`
- `MCP_LLM_STREAMING` / `CHAT_LLM_STREAMING`

After editing the variables at the top of `set_config.sh`, run **`./set_config.sh`** to write both config files, or **`MCP_ONLY=1 ./set_config.sh`** / **`CHAT_ONLY=1 ./set_config.sh`** to update only one:

```bash
export PROJECT_ROOT=<path-to-rare-disease-chat>
export MCP_SIMPLE_TOOL_DIR=$PROJECT_ROOT/mcp-server/mcp_simple_tool

cd $PROJECT_ROOT
./set_config.sh
```


## Quick Start

One-click start (MCP server in background + Web UI in foreground). Run once **Configuration** and **Installation** first.

```bash
cd rare-disease-chat
./auto_start.sh
```

Then open http://localhost:8080/rdagent/ in your browser. Press Ctrl+C to stop the dashboard (MCP server keeps running; stop with `mcp-server/stop_server.sh`).

For separate steps (start_server.sh / start_dashboard.sh), CLI mode, and Docker, see **More information** below.

---

## More information

### Docker install and run

In Docker mode you **must** have the two config files set (e.g. run **`./set_config.sh`** once from this directory, or edit the two JSON files manually). Then build and run with Docker. Chat in Docker is terminal (CLI) only—no Web UI.

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

### Local install and run

In local mode, run **`./set_config.sh`** once (after editing its top variables) to write both config files. Then run `start_server.sh` and `start_dashboard.sh`. Chat supports **two modes**: Web UI (RDAgent Dashboard) and CLI (terminal).

#### Start the server

> **Note**: Set `PROJECT_ROOT` and `MCP_SIMPLE_TOOL_DIR` before use (see top of this README).

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

2. **Choose one of two modes** (from `chat-system/` with venv activated):

   **Mode (a): Web UI (RDAgent Dashboard)** (recommended)

   Run either:
   - `./start_dashboard.sh`
   - or `python rdagent_dashboard_api.py`

   Then open http://localhost:8080/rdagent/ in your browser. See [chat-system/WEB_UI_README.md](chat-system/WEB_UI_README.md) for details.

   **Mode (b): CLI (terminal chat)**

   Run:
   - `python phenotype_to_disease_controller_langchain_stream_api.py`

   Type `quit` to exit, `clear` to clear chat history.

After startup (either option), the system supports:
- Phenotype extraction and symptom analysis
- Disease diagnosis and case extraction
- Disease information retrieval and normalization
- Multi-turn conversation memory

---

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

## Project structure

```
rare-disease-chat/               # Project root (this README)
├── README.md                    # This file
├── set_config.sh                # Edit MCP_* and CHAT_* vars; run to write one or both configs
├── docker-compose.yml           # Docker Compose config
├── mcp-server/                  # MCP server
│   ├── README.md                # MCP server API and usage
│   ├── Dockerfile               # MCP server image
│   ├── start_server.sh          # Start script
│   ├── stop_server.sh           # Stop script
│   ├── mcp_simple_tool/         # Core tool module
│   │   ├── tools/               # Tool implementations
│   │   ├── scripts/             # Helper scripts
│   │   │   └── rare_disease_diagnose/  # Rare disease diagnosis scripts
│   │   │       ├── vc_ranker.py                      # Ensemble disease ranking (Z-statistics)
│   │   │       ├── query_kg.py                       # KG query & phenotype-to-disease prompt generation
│   │   │       ├── disease_scraper/                  # Disease info scraper
│   │   │       └── prompt_config_forKG.json         # KG prompt config
│   │   └── data/                # Data files (source: VC-RDAgent repo, see below)
│   │       ├── disease_annotations/   # ← VC-RDAgent/general_cases/disease_ids_names.json
│   │       ├── disease_phenotype_kg/  # ← VC-RDAgent/disease_phenotype_kg/ (*.csv)
│   │       └── hpo_annotations/       # ← VC-RDAgent/hpo_annotations/ (e.g. hp.obo)
├── chat-system/                 # Chat system
│   ├── README.md                # Chat system overview; see parent README for full overview
│   ├── Dockerfile               # Chat system image
│   ├── phenotype_to_disease_controller_langchain_stream_api.py  # Core controller (CLI chat)
│   ├── rdagent_dashboard_api.py # RDAgent Dashboard backend (recommended Web UI)
│   ├── start_dashboard.sh      # Start RDAgent Dashboard
│   ├── inference_config.json    # LLM config (written by set_config.sh)
│   ├── main.py                  # Entry point
│   ├── pyproject.toml           # Dependencies
│   ├── rdagent/                 # Web UI frontend
│   │   └── rdagent_dashboard.html  # RDAgent Dashboard frontend
│   └── WEB_UI_README.md         # Web UI guide
└── ...
```

**Data source:** The files under `mcp-server/mcp_simple_tool/data/` are provided by or derived from the parent repository **VC-RDAgent**. Corresponding paths:

| Local path (`mcp_simple_tool/data/`) | Source in VC-RDAgent |
|--------------------------------------|------------------------|
| `disease_annotations/disease_ids_names.json` | `general_cases/disease_ids_names.json` |
| `disease_phenotype_kg/*.csv` | `disease_phenotype_kg/` |
| `hpo_annotations/hp.obo` | `hpo_annotations/` |

Copy or symlink these from the VC-RDAgent root if you are building from a fresh clone.

**First run:** The file `hp.index` is built under `mcp_simple_tool/data/hpo_annotations/` on first use (HPO phenotype index for search). This one-time build can take a long time; please wait for it to finish.

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
