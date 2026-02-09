# Docker One-Click Startup Guide

This document explains how to use Docker to start MCP Server and Dashboard WebUI with one command.

## Quick Start

### 1. Build Image

Execute in the project root directory (`rare-disease-chat/`):

```bash
docker build -t rarellm-all-in-one:latest .
```

### 2. Run Container

```bash
docker run -d \
  --name rarellm-all \
  -p 3000:3000 \
  -p 8080:8080 \
  rarellm-all-in-one:latest
```

### 3. Access Services

- **MCP Server**: http://localhost:3000
- **Dashboard WebUI**: http://localhost:8080/rdagent/

### 4. View Logs

```bash
# View container logs
docker logs -f rarellm-all

# View MCP Server logs
docker exec rarellm-all tail -f /tmp/mcp.log

# View Dashboard logs
docker exec rarellm-all tail -f /tmp/dashboard.log
```

### 5. Stop Services

```bash
docker stop rarellm-all
docker rm rarellm-all
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

To customize configurations, you can mount configuration files:

```bash
docker run -d \
  --name rarellm-all \
  -p 3000:3000 \
  -p 8080:8080 \
  -v $(pwd)/chat-system/inference_config.json:/app/chat-system/inference_config.json \
  -v $(pwd)/mcp-server/mcp_simple_tool/scripts/rare_disease_diagnose/prompt_config_forKG.json:/app/mcp-server/mcp_simple_tool/scripts/rare_disease_diagnose/prompt_config_forKG.json \
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

## Using docker-compose

You can also create a simple `docker-compose.all.yml` file:

```yaml
version: '3.8'

services:
  rarellm-all:
    build:
      context: .
      dockerfile: Dockerfile
    image: rarellm-all-in-one:latest
    container_name: rarellm-all
    ports:
      - "3000:3000"
      - "8080:8080"
    environment:
      - MCP_PORT=3000
      - DASHBOARD_PORT=8080
      - MCP_ENDPOINT=http://localhost:3000/mcp/
      - MCP_TIMEOUT=600
    volumes:
      - ./mcp-server/mcp_simple_tool/data:/app/mcp-server/mcp_simple_tool/data
    restart: unless-stopped
```

Then run:

```bash
docker-compose -f docker-compose.all.yml up -d
```
