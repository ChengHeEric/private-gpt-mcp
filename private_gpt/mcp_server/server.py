import logging
from pathlib import Path

from injector import Injector
from mcp.server.fastmcp import FastMCP

from private_gpt.chat.extensions.context_filter import ContextFilter
from private_gpt.chat.input_models import MessageInput
from private_gpt.di import get_global_injector
from private_gpt.events.models import TextBlock
from private_gpt.server.chat.chat_models import ChatBody
from private_gpt.server.chat.chat_request_mapper import ChatRequestMapper
from private_gpt.server.chat.chat_service import ChatService
from private_gpt.server.ingest.ingest_service import IngestService
from private_gpt.server.primitives.semantic_search_service import (
    SemanticSearchService,
)

logger = logging.getLogger(__name__)


def create_mcp_server(injector: Injector | None = None) -> FastMCP:
    """Create a FastMCP instance exposing privateGPT functionality as MCP tools.

    Args:
        injector: Optional injector with pre-initialized services.
                  If None, uses the global injector.
    """
    if injector is None:
        injector = get_global_injector()

    mcp = FastMCP("PrivateGPT")

    ingest_service = injector.get(IngestService)
    semantic_search_service = injector.get(SemanticSearchService)
    chat_service = injector.get(ChatService)
    chat_request_mapper = injector.get(ChatRequestMapper)

    @mcp.tool()
    def query_documents(query: str, limit: int = 5) -> str:
        """Search documents in the local vector store using semantic search.

        Args:
            query: The search query text to find relevant documents
            limit: Maximum number of results to return (default: 5)
        """
        try:
            context_filter = ContextFilter(collection="pgpt_collection")
            chunks = semantic_search_service.retrieve_semantic_relevant(
                text=query,
                context_filter=context_filter,
                limit=limit,
                score_threshold=0.0,
                expand=True,
            )
            if not chunks:
                return "No documents found matching the query."

            results = []
            for chunk in chunks:
                doc_ref = f"[Document: {chunk.document.artifact}]"
                text = chunk.text or ""
                score = chunk.score or 0.0
                results.append(f"{doc_ref} (score: {score:.3f})\n{text}")

            return "\n\n---\n\n".join(results)
        except Exception as e:
            logger.error("Error querying documents: %s", e)
            return f"Error querying documents: {e}"

    @mcp.tool()
    def ingest_document(file_path: str) -> str:
        """Ingest a document file into privateGPT's local vector store.

        Args:
            file_path: Absolute or relative path to the file to ingest
        """
        try:
            path = Path(file_path).resolve()
            if not path.exists():
                return f"File not found: {path}"
            if not path.is_file():
                return f"Path is not a file: {path}"

            artifact = path.stem
            collection = "pgpt_collection"

            ingest_service.initialize_artifact_indices(
                collection=collection,
                artifact=artifact,
            )

            ingested_docs = ingest_service.populate_vector_index(
                collection=collection,
                artifact=artifact,
                file_data=path,
                file_metadata={"file_name": path.name},
            )

            return (
                f"Successfully ingested {len(ingested_docs)} document(s) "
                f"from {path.name}"
            )
        except Exception as e:
            logger.error("Error ingesting document: %s", e)
            return f"Error ingesting document: {e}"

    @mcp.tool()
    def list_documents() -> str:
        """List all indexed documents in the local vector store."""
        try:
            ingested_files = list(
                ingest_service.get_ingested_files("pgpt_collection")
            )
            if not ingested_files:
                return "No documents have been ingested yet."

            results = []
            for doc in ingested_files:
                file_name = (doc.doc_metadata or {}).get("file_name", "unknown")
                results.append(f"- {doc.artifact} ({file_name})")

            return f"Found {len(results)} document(s):\n" + "\n".join(results)
        except Exception as e:
            logger.error("Error listing documents: %s", e)
            return f"Error listing documents: {e}"

    @mcp.tool()
    async def chat(message: str) -> str:
        """Chat with the privateGPT LLM using the configured model.

        Args:
            message: The user message to send to the LLM
        """
        try:
            chat_body = ChatBody(
                messages=[MessageInput(role="user", content=message)],
                stream=False,
            )
            chat_request = await chat_request_mapper.create_request_from_body(
                chat_body
            )
            completion = await chat_service.chat(chat_request)

            if completion.response:
                return completion.response
            if completion.content:
                text_parts = [
                    block.text
                    for block in completion.content
                    if isinstance(block, TextBlock)
                ]
                return "".join(text_parts) if text_parts else "No response generated."
            return "No response generated."
        except Exception as e:
            logger.error("Error in chat: %s", e)
            return f"Error in chat: {e}"

    return mcp
