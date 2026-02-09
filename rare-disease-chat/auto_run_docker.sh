#!/bin/bash
# Docker container management script
# Usage: ./auto_start_docker.sh [start|stop|restart|rm|logs|status|extract-config]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

IMAGE_NAME="rarellm-all-in-one:latest"
CONTAINER_NAME="rarellm-all"
MCP_PORT="${MCP_PORT:-3000}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8080}"

# Configuration files
CONFIG_DIR="$SCRIPT_DIR/docker_configs"
INFERENCE_CONFIG="$CONFIG_DIR/inference_config.json"
PROMPT_CONFIG="$CONFIG_DIR/prompt_config_forKG.json"

# Function to check if container exists
container_exists() {
    docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"
}

# Function to check if container is running
container_running() {
    docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"
}

# Function to extract configuration files
extract_configs() {
    echo "Extracting configuration files from container..."
    
    # Create config directory
    mkdir -p "$CONFIG_DIR"
    
    # Check if image exists
    if ! docker images --format '{{.Repository}}:{{.Tag}}' | grep -q "^${IMAGE_NAME}$"; then
        echo "Error: Docker image ${IMAGE_NAME} not found"
        echo "Please build the image first: docker build -t ${IMAGE_NAME} ."
        return 1
    fi
    
    # Create temporary container
    TEMP_CONTAINER="temp-config-$$"
    docker create --name "$TEMP_CONTAINER" "$IMAGE_NAME" > /dev/null 2>&1
    
    # Extract files
    if [ ! -f "$INFERENCE_CONFIG" ]; then
        echo "  Extracting inference_config.json..."
        docker cp "$TEMP_CONTAINER:/app/chat-system/inference_config.json" "$INFERENCE_CONFIG"
    else
        echo "  inference_config.json already exists, skipping..."
    fi
    
    if [ ! -f "$PROMPT_CONFIG" ]; then
        echo "  Extracting prompt_config_forKG.json..."
        docker cp "$TEMP_CONTAINER:/app/mcp-server/mcp_simple_tool/scripts/rare_disease_diagnose/prompt_config_forKG.json" "$PROMPT_CONFIG"
    else
        echo "  prompt_config_forKG.json already exists, skipping..."
    fi
    
    # Remove temporary container
    docker rm "$TEMP_CONTAINER" > /dev/null 2>&1
    
    echo "✓ Configuration files extracted to $CONFIG_DIR"
}

# Function to start container
start_container() {
    echo "=== Starting Docker Container ==="
    echo ""

    # Check if image exists
    if ! docker images --format '{{.Repository}}:{{.Tag}}' | grep -q "^${IMAGE_NAME}$"; then
        echo "Error: Docker image ${IMAGE_NAME} not found"
        echo "Please build the image first: docker build -t ${IMAGE_NAME} ."
        return 1
    fi
    
    # Extract configuration files if they don't exist
    if [ ! -f "$INFERENCE_CONFIG" ] || [ ! -f "$PROMPT_CONFIG" ]; then
        extract_configs
        echo ""
    fi
    
    # Stop and remove existing container if it exists
    if container_exists; then
        echo "Stopping existing container..."
        docker stop "$CONTAINER_NAME" > /dev/null 2>&1 || true
        docker rm "$CONTAINER_NAME" > /dev/null 2>&1 || true
    fi
    
    # Start container with mounted configuration files
    echo "Starting container with mounted configuration files..."
    echo ""
    
    docker run -d \
        --name "$CONTAINER_NAME" \
        -p "${MCP_PORT}:${MCP_PORT}" \
        -p "${DASHBOARD_PORT}:${DASHBOARD_PORT}" \
        -e MCP_PORT="${MCP_PORT}" \
        -e DASHBOARD_PORT="${DASHBOARD_PORT}" \
        -v "$INFERENCE_CONFIG:/app/chat-system/inference_config.json" \
        -v "$PROMPT_CONFIG:/app/mcp-server/mcp_simple_tool/scripts/rare_disease_diagnose/prompt_config_forKG.json" \
        "$IMAGE_NAME"
    
    if [ $? -eq 0 ]; then
        echo "✓ Container started successfully"
        echo ""
        echo "MCP Server: http://localhost:${MCP_PORT}"
        echo "Dashboard: http://localhost:${DASHBOARD_PORT}/rdagent/"
        echo ""
        echo "Configuration files location: $CONFIG_DIR"
        echo "  - Edit $INFERENCE_CONFIG to customize Dashboard settings"
        echo "  - Edit $PROMPT_CONFIG to customize MCP Server settings"
        echo "  - After modifying configuration files, run './auto_run_docker.sh restart' to apply changes"
        echo ""
        echo "Use './auto_start_docker.sh logs' to view logs"
    else
        echo "✗ Failed to start container"
        return 1
    fi
}

# Function to stop container
stop_container() {
    echo "=== Stopping Docker Container ==="
    echo ""
    
    if container_running; then
        echo "Stopping container..."
        docker stop "$CONTAINER_NAME"
        echo "✓ Container stopped"
    elif container_exists; then
        echo "Container is already stopped"
    else
        echo "Container does not exist"
    fi
}

# Function to remove container
remove_container() {
    echo "=== Removing Docker Container ==="
    echo ""
    
    if container_exists; then
        if container_running; then
            echo "Stopping container first..."
            docker stop "$CONTAINER_NAME" > /dev/null 2>&1
        fi
        echo "Removing container..."
        docker rm "$CONTAINER_NAME"
        echo "✓ Container removed"
    else
        echo "Container does not exist"
    fi
}

# Function to restart container
restart_container() {
    echo "=== Restarting Docker Container ==="
    echo ""
    
    if container_exists; then
        echo "Restarting container..."
        docker restart "$CONTAINER_NAME"
        echo "✓ Container restarted"
        echo ""
        echo "MCP Server: http://localhost:${MCP_PORT}"
        echo "Dashboard: http://localhost:${DASHBOARD_PORT}/rdagent/"
    else
        echo "Container does not exist, starting new container..."
        start_container
    fi
}

# Function to show logs
show_logs() {
    if container_exists; then
        echo "=== Container Logs ==="
        echo ""
        docker logs -f "$CONTAINER_NAME"
    else
        echo "Container does not exist"
        exit 1
    fi
}

# Function to show status
show_status() {
    echo "=== Container Status ==="
    echo ""
    
    if container_exists; then
        if container_running; then
            echo "Status: Running"
            echo ""
            docker ps --filter "name=$CONTAINER_NAME" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
            echo ""
            echo "MCP Server: http://localhost:${MCP_PORT}"
            echo "Dashboard: http://localhost:${DASHBOARD_PORT}/rdagent/"
        else
            echo "Status: Stopped"
            echo ""
            docker ps -a --filter "name=$CONTAINER_NAME" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
        fi
    else
        echo "Status: Container does not exist"
    fi
}

# Main command handler
case "${1:-start}" in
    start)
        start_container
        ;;
    stop)
        stop_container
        ;;
    restart)
        restart_container
        ;;
    rm|remove)
        remove_container
        ;;
    logs)
        show_logs
        ;;
    status)
        show_status
        ;;
    extract-config)
        extract_configs
        ;;
    *)
        echo "Usage: $0 [start|stop|restart|rm|logs|status|extract-config]"
        echo ""
        echo "Commands:"
        echo "  start          - Start container (default)"
        echo "  stop           - Stop container"
        echo "  restart        - Restart container"
        echo "  rm, remove     - Remove container"
        echo "  logs           - View container logs (follow mode)"
        echo "  status         - Show container status"
        echo "  extract-config - Extract configuration files from image"
        echo ""
        echo "Environment variables:"
        echo "  MCP_PORT       - MCP Server port (default: 3000)"
        echo "  DASHBOARD_PORT - Dashboard port (default: 8080)"
        echo ""
        echo "Examples:"
        echo "  ./auto_start_docker.sh start"
        echo "  ./auto_start_docker.sh stop"
        echo "  ./auto_start_docker.sh restart"
        echo "  MCP_PORT=3001 DASHBOARD_PORT=8081 ./auto_start_docker.sh start"
        exit 1
        ;;
esac
