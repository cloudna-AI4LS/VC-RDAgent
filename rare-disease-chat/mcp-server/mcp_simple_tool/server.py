import contextlib
import inspect
import logging
import importlib
import pathlib
import sys
from collections.abc import AsyncIterator

import click
import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.applications import Starlette
from starlette.routing import Mount
from starlette.types import Receive, Scope, Send
from starlette.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)

def load_tools_from_directory(directory: pathlib.Path):
    """Scan directory and load tools; return {tool_name: {...}} dict."""
    tools = {}
    if not directory.is_dir():
        logger.warning("Tools directory not found: %s", directory)
        return tools

    for path in directory.glob("*.py"):
        if path.name.startswith("_"):
            continue
        module_name = path.stem
        tool_name = module_name.replace("_", "-")
        try:
            spec = importlib.util.spec_from_file_location(
                f"{__package__}.tools.{module_name}", path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            if not (
                inspect.iscoroutinefunction(getattr(module, "call_tool", None))
                and callable(getattr(module, "get_tool_definition", None))
            ):
                logger.warning("Skipping %s: missing required functions", path)
                continue

            tools[tool_name] = {
                "definition": module.get_tool_definition(),
                "call_tool_func": module.call_tool,
            }
        except Exception as exc:
            logger.error("Cannot load tool %s: %s", path, exc)

    if not tools:
        logger.warning("No tools loaded from %s", directory)
    return tools

@click.command()
@click.option("--port", default=3000, help="Port to listen on for HTTP")
@click.option(
    "--log-level",
    default="INFO",
    help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
)
@click.option(
    "--json-response",
    is_flag=True,
    default=False,
    help="Enable JSON responses instead of SSE streams",
)
def main(port: int, log_level: str, json_response: bool) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    app = Server("mcp-streamable-http-stateless-demo")

    tools_dir = pathlib.Path(__file__).with_suffix("").parent / "tools"
    loaded_tools = load_tools_from_directory(tools_dir)

    @app.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[types.ContentBlock]:
        if name not in loaded_tools:
            raise ValueError(f"Tool '{name}' not found")
        return await loaded_tools[name]["call_tool_func"](
            app.request_context, arguments
        )

    @app.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [t["definition"] for t in loaded_tools.values()]

    session_manager = StreamableHTTPSessionManager(
        app=app, event_store=None, json_response=json_response, stateless=True
    )

    async def handle_streamable_http(
        scope: Scope, receive: Receive, send: Send
    ) -> None:
        await session_manager.handle_request(scope, receive, send)

    @contextlib.asynccontextmanager
    async def lifespan(_: Starlette) -> AsyncIterator[None]:
        async with session_manager.run():
            logger.info("Application started")
            yield
        logger.info("Application shutting down")

    starlette_app = Starlette(
        debug=True,
        routes=[Mount("/mcp", app=handle_streamable_http)],
        lifespan=lifespan,
    )

    import uvicorn
    
    # Add CORS middleware: allow all origins, methods, headers
    starlette_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    uvicorn.run(starlette_app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    sys.exit(main())