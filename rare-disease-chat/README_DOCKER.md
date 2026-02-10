# Docker One-Click Startup Guide

This document explains how to use Docker to start MCP Server and Dashboard WebUI with one command.

## Build Image

Before using the one-click script or running the container manually, build the Docker image once. Run from the **rare-disease-chat** directory (where the root `Dockerfile` is located):

```bash
cd rare-disease-chat
docker build -t rarellm-all-in-one:latest .
```

Build may take several minutes (installing dependencies, etc.). After the build completes, use the Quick Start below or the `docker run` examples in Advanced Configuration.

## Quick Start

The script can be copied to any location and run from there. Since it uses Docker, the container is independent of where the script is executed:

```bash
# Run from any location 
./auto_run_docker.sh start
```

The script will automatically perform the following operations:
1. Check that the Docker image exists (if not, build it first—see [Build Image](#build-image) above)
2. Extract configuration files to the `docker_configs/` directory
3. Start the container with mounted configuration files
4. Display service access addresses

**Access Services**:
- **MCP Server**: http://localhost:3000
- **Dashboard WebUI**: http://localhost:8080/rdagent/

**Other Common Commands**:
```bash
# View logs
./auto_run_docker.sh logs

# Check status
./auto_run_docker.sh status

# Stop container
./auto_run_docker.sh stop

# Restart container
./auto_run_docker.sh restart

# Remove container
./auto_run_docker.sh rm
```

## Advanced Configuration

### Custom Ports

```bash
docker run -d \
  --name rarellm-all \
  -p 3001:3001 \
  -p 8081:8081 \
  -e MCP_PORT=3001 \
  -e DASHBOARD_PORT=8081 \
  rarellm-all-in-one:latest
```

### Mount Configuration Files

To customize configurations, you can mount configuration files. **Important**: You must first extract these files from the container before mounting them. If the files don't exist on your host, Docker will create directories instead, causing mount errors.

**Step 1: Extract files from container (run from any directory)**

```bash
# Create a temporary container to extract files
docker create --name temp-config rarellm-all-in-one:latest

# Extract configuration files to current directory
docker cp temp-config:/app/chat-system/inference_config.json ./inference_config.json
docker cp temp-config:/app/mcp-server/mcp_simple_tool/scripts/rare_disease_diagnose/prompt_config_forKG.json ./prompt_config_forKG.json

# Remove temporary container
docker rm temp-config
```

**Step 2: Mount the extracted files**

```bash
# Now mount them (run from the directory where you extracted the files)
docker run -d \
  --name rarellm-all \
  -p 3000:3000 \
  -p 8080:8080 \
  -v $(pwd)/inference_config.json:/app/chat-system/inference_config.json \
  -v $(pwd)/prompt_config_forKG.json:/app/mcp-server/mcp_simple_tool/scripts/rare_disease_diagnose/prompt_config_forKG.json \
  rarellm-all-in-one:latest
```

### Mount Data Directory (Persistence)

```bash
docker run -d \
  --name rarellm-all \
  -p 3000:3000 \
  -p 8080:8080 \
  -v $(pwd)/mcp-server/mcp_simple_tool/data:/app/mcp-server/mcp_simple_tool/data \
  rarellm-all-in-one:latest
```

### Environment Variables

Available environment variables:

- `MCP_PORT`: MCP Server port (default: 3000)
- `DASHBOARD_PORT`: Dashboard port (default: 8080)
- `MCP_ENDPOINT`: Endpoint for Dashboard to connect to MCP Server (default: http://localhost:3000/mcp/)
- `MCP_TIMEOUT`: MCP request timeout in seconds (default: 600)
- `PROJECT_ROOT`: Project root directory (default: /app)
- `HF_ENDPOINT`: HuggingFace mirror endpoint (default: https://hf-mirror.com)

Example:

```bash
docker run -d \
  --name rarellm-all \
  -p 3000:3000 \
  -p 8080:8080 \
  -e MCP_TIMEOUT=1200 \
  -e HF_ENDPOINT=https://hf-mirror.com \
  rarellm-all-in-one:latest
```

## Troubleshooting

### 1. Container Fails to Start

Check container logs:

```bash
docker logs rarellm-all
```

### 2. Port Already in Use

If the port is already in use, you can modify the port mapping:

```bash
docker run -d \
  --name rarellm-all \
  -p 3001:3000 \
  -p 8081:8080 \
  rarellm-all-in-one:latest
```

Then access:
- MCP Server: http://localhost:3001
- Dashboard: http://localhost:8081/rdagent/

### 3. Service Startup Failure

Enter the container to check:

```bash
docker exec -it rarellm-all bash
```

Inside the container, check:
- MCP Server logs: `cat /tmp/mcp.log`
- Dashboard logs: `cat /tmp/dashboard.log`
- Process status: `ps aux`

### 4. Dashboard Cannot Connect to MCP Server

Ensure the `MCP_ENDPOINT` environment variable is set correctly. Inside the container, MCP Server runs on `localhost`, so the default configuration should work.

If using a custom network, you may need to modify `MCP_ENDPOINT`:

```bash
docker run -d \
  --name rarellm-all \
  --network mynetwork \
  -p 3000:3000 \
  -p 8080:8080 \
  -e MCP_ENDPOINT=http://localhost:3000/mcp/ \
  rarellm-all-in-one:latest
```

## Development Mode

If you need to see code changes in real-time during development, you can mount the source code directories:

```bash
docker run -d \
  --name rarellm-all \
  -p 3000:3000 \
  -p 8080:8080 \
  -v $(pwd)/mcp-server:/app/mcp-server \
  -v $(pwd)/chat-system:/app/chat-system \
  rarellm-all-in-one:latest
```

**Note**: You need to restart the container after code changes for them to take effect.

