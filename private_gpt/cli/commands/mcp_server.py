import logging

import typer

from private_gpt.di import create_application_injector, set_global_injector
from private_gpt.initialize import initialize_globals
from private_gpt.launcher import apply_migrations, eager_loading
from private_gpt.mcp_server.server import create_mcp_server

logger = logging.getLogger(__name__)


def mcp_server_command(
    host: str = typer.Option("0.0.0.0", help="Bind address for the MCP server"),
    port: int = typer.Option(8765, help="Port for the MCP server (default: 8765)"),
    log_level: str = typer.Option(
        "info", "--log-level", help="debug | info | warn | error"
    ),
) -> None:
    """Start the MCP server exposing privateGPT tools via SSE transport."""
    logger.info(
        "Initializing privateGPT for MCP server on %s:%s",
        host,
        port,
    )

    initialize_globals()
    injector = create_application_injector()
    set_global_injector(injector)
    apply_migrations(injector)
    eager_loading(injector)

    logger.info("Creating MCP server with privateGPT tools")
    mcp = create_mcp_server(injector)

    logger.info(
        "Starting MCP SSE server on http://%s:%s",
        host,
        port,
    )
    mcp.run(transport="sse", host=host, port=port, log_level=log_level)
