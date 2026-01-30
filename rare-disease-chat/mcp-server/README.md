# Rare Disease Diagnosis API Server

A HTTP API server for accessing rare disease diagnosis tools. The server uses MCP (Model Context Protocol) framework and runs based on `mcp_simple_tool/server.py`.

## Quick Start

### 1. Install Dependencies

See the project root [README.md](../README.md) section **"Start the server"** for the full install flow (venv, CPU-only PyTorch, `uv pip install -e .`).

### 2. Start Server (Background)

```bash
# Start with default configuration (port 3000)
./start_server.sh

# Or use custom port
PORT=3001 ./start_server.sh
```

### 3. Check Server Status

```bash
# Check if server is running
ps -p $(cat api_server.pid) 2>/dev/null && echo "Server is running" || echo "Server is not running"

# View logs
tail -f api_server.log
```

### 4. Stop Server

```bash
./stop_server.sh
```

## Configuration

Server can be configured via environment variables:

- `PORT`: Server port (default: 3000)
- `LOG_FILE`: Log file path (default: api_server.log)
- `PID_FILE`: PID file path (default: api_server.pid)

## API Usage

### Important Notes

1. **URL Path**: Must use `/mcp/` (with trailing slash), not `/mcp`
2. **Accept Header**: Must include `Accept: application/json, text/event-stream`
3. **Timeout**: Tool execution may take a long time, recommend setting timeout (e.g., 300 seconds)

### 1. List All Tools

```bash
curl -X POST http://localhost:3000/mcp/ \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/list",
    "params": {}
  }'
```

### 2. Call Tool - Phenotype Extractor

```bash
curl -X POST http://localhost:3000/mcp/ \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
      "name": "phenotype-extractor",
      "arguments": {
        "query": "Patient presents with intellectual disability, developmental delay, and hypotonia"
      }
    }
  }'
```

### 3. Call Tool - Extract Disease Cases

```bash
curl -X POST http://localhost:3000/mcp/ \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/call",
    "params": {
      "name": "disease-case-extractor",
      "arguments": {
        "query": "Patient has seizures, microcephaly, and intellectual disability",
        "top_k": 100,
        "final_top_k": 50
      }
    }
  }'
```

### 4. Call Tool - Disease Information Retrieval

```bash
curl -X POST http://localhost:3000/mcp/ \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{
    "jsonrpc": "2.0",
    "id": 3,
    "method": "tools/call",
    "params": {
      "name": "disease-information-retrieval",
      "arguments": {
        "query": "Patient may have Down syndrome or Trisomy 21",
        "use_model": true
      }
    }
  }'
```

### 5. Call Tool - Disease Diagnosis

**Note:** This tool requires outputs from `phenotype-extractor` and `disease-case-extractor` tools. First run those tools, then use their outputs here.

```bash
curl -X POST http://localhost:3000/mcp/ \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{
    "jsonrpc": "2.0",
    "id": 4,
    "method": "tools/call",
    "params": {
      "name": "disease-diagnosis",
      "arguments": {
        "original_query": "Patient presents with intellectual disability, developmental delay, and hypotonia",
        "extracted_phenotypes": {
          "HP:0001250,HP:0001249,HP:0001252": {
            "Intellectual disability": {
              "hpo_id": "HP:0001250",
              "phenotype abnormal category": "Abnormality of the nervous system",
              "synonyms": [],
              "parent categories": [],
              "detailed information": ""
            },
            "Global developmental delay": {
              "hpo_id": "HP:0001249",
              "phenotype abnormal category": "Abnormality of the nervous system",
              "synonyms": [],
              "parent categories": [],
              "detailed information": ""
            },
            "Hypotonia": {
              "hpo_id": "HP:0001252",
              "phenotype abnormal category": "Abnormality of the nervous system",
              "synonyms": [],
              "parent categories": [],
              "detailed information": ""
            }
          }
        },
        "disease_cases": {
          "Case 1": {
            "Disease name": "Down syndrome",
            "Disease id": "OMIM:190685",
            "Disease category": "Syndrome",
            "Disease description": "A chromosomal disorder"
          },
          "Case 2": {
            "Disease name": "Rett syndrome",
            "Disease id": "OMIM:312750",
            "Disease category": "Syndrome",
            "Disease description": "A neurodevelopmental disorder"
          }
        }
      }
    }
  }'
```

## Python Client Example

```python
import requests
import json

# Call tool
response = requests.post(
    "http://localhost:3000/mcp/",  # Note the trailing slash
    json={
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "phenotype-extractor",
            "arguments": {
                "query": "Patient presents with intellectual disability"
            }
        }
    },
    headers={
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream"  # Required Accept header
    },
    timeout=300  # Tool execution may take a long time
)

result = response.json()
print(json.dumps(result, indent=2, ensure_ascii=False))
```

## Available Tools

- `phenotype-extractor` - Extract phenotype information from text
- `disease-case-extractor` - Extract related disease cases based on phenotypes
- `disease-information-retrieval` - Retrieve detailed disease information
- `disease-diagnosis` - Perform disease diagnosis
- `judge-symbol` - Judge symbol type (gene or protein)
- `reports` - Manage report prompt templates

## Manual Start (Development Mode)

For debugging, you can start manually:

```bash
# Development mode (foreground)
python3 mcp_simple_tool/server.py --port 3000 --log-level DEBUG

# Enable JSON response format
python3 mcp_simple_tool/server.py --port 3000 --json-response

# View all options
python3 mcp_simple_tool/server.py --help
```

## Troubleshooting

### 1. Port Already in Use

**Error:**
```
ERROR: [Errno 98] error while attempting to bind on address ('0.0.0.0', 3000): address already in use
```

**Solution:**
```bash
# Stop the process using the port
./stop_server.sh

# Or use a different port
PORT=3001 ./start_server.sh
```

### 2. 307 Temporary Redirect

**Error:**
```
INFO: "POST /mcp HTTP/1.1" 307 Temporary Redirect
```

**Cause:** URL path does not end with a slash

**Solution:** Use `/mcp/` instead of `/mcp`

### 3. 406 Not Acceptable

**Error:**
```
INFO: "POST /mcp/ HTTP/1.1" 406 Not Acceptable
{"error": "Not Acceptable: Client must accept both application/json and text/event-stream"}
```

**Cause:** Missing required Accept header

**Solution:** Add `Accept: application/json, text/event-stream` to request headers

### 4. anyio.ClosedResourceError (Log Noise)

**Error:**
```
ERROR - Error in message router
anyio.ClosedResourceError
```

**Note:**
- ✅ **Does not affect functionality** - This is log noise during session cleanup
- Request has already returned successfully (200 OK)
- This error can be ignored, or reduce log level to WARNING

**Solution (Optional):**
```bash
# Edit start_server.sh, change --log-level INFO to --log-level WARNING
```

### 5. Tool Loading Failed (NEBULA Environment Variables)

**Error:**
```
ERROR - Cannot load tool ... : Missing required environment variables: NEBULA_HOST, NEBULA_PORT, NEBULA_USERNAME, NEBULA_PASSWORD, NEBULA_SPACE
```

**Note:**
- Some tools require Nebula graph database connection
- These tools are optional and **do not affect core rare disease diagnosis functionality**
- Core tools (phenotype-extractor, disease-case-extractor, etc.) work normally

**Solution (if you need these tools):**
Create a `.env` file and configure Nebula connection:
```bash
NEBULA_HOST=your_nebula_host
NEBULA_PORT=9669
NEBULA_USERNAME=your_username
NEBULA_PASSWORD=your_password
NEBULA_SPACE=your_space
```

### 6. Request Timeout

**Cause:** Tool execution takes a long time (may need 10-30 seconds)

**Solution:** Increase timeout setting (e.g., 300 seconds)

## View Logs

```bash
# View logs in real-time
tail -f api_server.log

# View last 100 lines
tail -n 100 api_server.log

# Filter out noise errors
tail -f api_server.log | grep -v "ClosedResourceError" | grep -v "Error in message router"
```

## File Description

- `mcp_simple_tool/server.py` - MCP server main program
- `start_server.sh` - Server startup script
- `stop_server.sh` - Server stop script
- `api_server.log` - Server log file
- `api_server.pid` - Server process ID file

## Testing

```bash
# Test if API is working
curl -X POST http://localhost:3000/mcp/ \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}'
```

If it returns a tool list, the server is running normally.
