"""
RAG Index Builder
=================

Stage 4A — RAG Foundation

Pipeline:

    Markdown knowledge documents
        ↓
    document loading
        ↓
    semantic chunking
        ↓
    local SentenceTransformer embeddings
        ↓
    persistent ChromaDB collection

Embedding model:
    sentence-transformers/all-MiniLM-L6-v2

The source documents are the project-derived Markdown files generated
by knowledge_base.py.

No external business knowledge is introduced by this module.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]
)

KNOWLEDGE_BASE_DIR = (
    PROJECT_ROOT
    / "data"
    / "knowledge_base"
)

VECTOR_STORE_DIR = (
    PROJECT_ROOT
    / "data"
    / "vector_store"
)

COLLECTION_NAME = (
    "forecasting_business_context"
)

EMBEDDING_MODEL_NAME = (
    "all-MiniLM-L6-v2"
)


# ============================================================================
# CHUNK CONFIGURATION
# ============================================================================

# Character-based chunking is sufficient for this small Markdown corpus.
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

# Avoid indexing extremely tiny fragments.
MIN_CHUNK_LENGTH = 80


# ============================================================================
# DOCUMENT DISCOVERY
# ============================================================================

def discover_markdown_documents() -> list[Path]:
    """Return all Markdown knowledge documents."""

    if not KNOWLEDGE_BASE_DIR.is_dir():
        raise FileNotFoundError(
            "Knowledge-base directory not found:\n"
            f"{KNOWLEDGE_BASE_DIR}\n\n"
            "Run knowledge_base.py first."
        )

    documents = sorted(
        KNOWLEDGE_BASE_DIR.rglob(
            "*.md"
        )
    )

    if not documents:
        raise FileNotFoundError(
            "No Markdown knowledge documents found in:\n"
            f"{KNOWLEDGE_BASE_DIR}"
        )

    return documents


# ============================================================================
# DOCUMENT LOADING
# ============================================================================

def load_document(
    path: Path,
) -> str:
    """Load one Markdown document."""

    content = path.read_text(
        encoding="utf-8"
    ).strip()

    if not content:
        raise ValueError(
            f"Knowledge document is empty:\n{path}"
        )

    return content


# ============================================================================
# TEXT NORMALIZATION
# ============================================================================

def normalize_text(
    text: str,
) -> str:
    """Normalize whitespace while preserving Markdown structure."""

    text = text.replace(
        "\r\n",
        "\n",
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    return text.strip()


# ============================================================================
# MARKDOWN-AWARE CHUNKING
# ============================================================================

def split_large_block(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Split an oversized Markdown section into overlapping character chunks.

    Preference order:
        1. paragraph boundaries
        2. sentence boundaries
        3. hard character split
    """

    text = text.strip()

    if len(text) <= chunk_size:
        return [text]

    paragraphs = re.split(
        r"\n\s*\n",
        text,
    )

    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:

        paragraph = paragraph.strip()

        if not paragraph:
            continue

        candidate = (
            paragraph
            if not current
            else f"{current}\n\n{paragraph}"
        )

        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(
                current.strip()
            )

        # Paragraph itself is too large.
        if len(paragraph) > chunk_size:

            sentences = re.split(
                r"(?<=[.!?])\s+",
                paragraph,
            )

            sentence_chunk = ""

            for sentence in sentences:

                sentence = sentence.strip()

                if not sentence:
                    continue

                candidate_sentence = (
                    sentence
                    if not sentence_chunk
                    else (
                        f"{sentence_chunk} "
                        f"{sentence}"
                    )
                )

                if (
                    len(candidate_sentence)
                    <= chunk_size
                ):
                    sentence_chunk = (
                        candidate_sentence
                    )

                else:
                    if sentence_chunk:
                        chunks.append(
                            sentence_chunk
                        )

                    sentence_chunk = (
                        sentence
                    )

            if sentence_chunk:
                chunks.append(
                    sentence_chunk
                )

            current = ""

        else:
            current = paragraph

    if current:
        chunks.append(
            current.strip()
        )

    # Add overlap between neighboring chunks.
    if overlap <= 0 or len(chunks) <= 1:
        return chunks

    overlapped: list[str] = []

    for index, chunk in enumerate(
        chunks
    ):

        if index == 0:
            overlapped.append(
                chunk
            )
            continue

        previous = chunks[
            index - 1
        ]

        overlap_text = previous[
            -overlap:
        ]

        combined = (
            f"{overlap_text}\n\n{chunk}"
        )

        overlapped.append(
            combined
        )

    return overlapped


def chunk_markdown(
    text: str,
    source_path: Path,
) -> list[dict]:
    """
    Create Markdown-aware chunks.

    Headings are retained with their section content so retrieved
    chunks carry useful context.
    """

    text = normalize_text(
        text
    )

    # Split on Markdown headings.
    sections = re.split(
        r"(?m)(?=^#{1,6}\s+)",
        text,
    )

    chunks: list[dict] = []

    current_heading = "document"

    for section_index, section in enumerate(
        sections
    ):

        section = section.strip()

        if not section:
            continue

        heading_match = re.match(
            r"^(#{1,6})\s+(.+?)(?:\n|$)",
            section,
        )

        if heading_match:
            current_heading = (
                heading_match.group(
                    2
                ).strip()
            )

        section_chunks = (
            split_large_block(
                section
            )
        )

        for chunk_index, chunk in enumerate(
            section_chunks
        ):

            chunk = chunk.strip()

            if len(chunk) < MIN_CHUNK_LENGTH:
                continue

            chunks.append(
                {
                    "text": chunk,
                    "source":
                        str(
                            source_path.relative_to(
                                PROJECT_ROOT
                            )
                        ),
                    "filename":
                        source_path.name,
                    "section":
                        current_heading,
                    "section_index":
                        section_index,
                    "chunk_index":
                        chunk_index,
                }
            )

    return chunks


# ============================================================================
# DETERMINISTIC CHUNK IDS
# ============================================================================

def create_chunk_id(
    source: str,
    section_index: int,
    chunk_index: int,
    text: str,
) -> str:
    """Create a deterministic ID so repeated indexing is idempotent."""

    raw = (
        f"{source}|"
        f"{section_index}|"
        f"{chunk_index}|"
        f"{text}"
    )

    digest = hashlib.sha256(
        raw.encode(
            "utf-8"
        )
    ).hexdigest()

    return f"chunk_{digest}"


# ============================================================================
# EMBEDDING MODEL
# ============================================================================

def load_embedding_model() -> SentenceTransformer:
    """
    Load the local embedding model.

    The first execution downloads the model from Hugging Face.
    Subsequent executions use the local cache.
    """

    print(
        f"Loading embedding model: "
        f"{EMBEDDING_MODEL_NAME}"
    )

    model = SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )

    print(
        "Embedding model loaded."
    )

    return model


# ============================================================================
# CHROMADB
# ============================================================================

def create_chroma_client() -> chromadb.PersistentClient:
    """Create the persistent ChromaDB client."""

    VECTOR_STORE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    return chromadb.PersistentClient(
        path=str(
            VECTOR_STORE_DIR
        )
    )


def get_collection(
    client: chromadb.PersistentClient,
):
    """Get or create the project knowledge collection."""

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "description":
                (
                    "Project-derived business "
                    "context for the Intelligent "
                    "Forecasting Agent"
                ),
            "embedding_model":
                EMBEDDING_MODEL_NAME,
        },
    )


# ============================================================================
# INDEXING
# ============================================================================

def build_chunks() -> list[dict]:
    """Load and chunk every Markdown knowledge document."""

    documents = (
        discover_markdown_documents()
    )

    all_chunks: list[dict] = []

    print(
        "\n=== DOCUMENT DISCOVERY ==="
    )

    for path in documents:

        content = load_document(
            path
        )

        chunks = chunk_markdown(
            content,
            path,
        )

        print(
            f"{path.relative_to(PROJECT_ROOT)} "
            f"→ {len(chunks)} chunks"
        )

        all_chunks.extend(
            chunks
        )

    if not all_chunks:
        raise RuntimeError(
            "No valid knowledge chunks were generated."
        )

    return all_chunks


def index_chunks(
    chunks: list[dict],
    model: SentenceTransformer,
    collection,
) -> None:
    """Embed and persist all chunks in ChromaDB."""

    ids = [
        create_chunk_id(
            source=chunk["source"],
            section_index=chunk[
                "section_index"
            ],
            chunk_index=chunk[
                "chunk_index"
            ],
            text=chunk["text"],
        )
        for chunk in chunks
    ]

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    metadatas = [
        {
            "source":
                chunk["source"],
            "filename":
                chunk["filename"],
            "section":
                chunk["section"],
            "section_index":
                int(
                    chunk[
                        "section_index"
                    ]
                ),
            "chunk_index":
                int(
                    chunk[
                        "chunk_index"
                    ]
                ),
        }
        for chunk in chunks
    ]

    print(
        f"\nGenerating embeddings for "
        f"{len(texts)} chunks..."
    )

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = embeddings.tolist()

    # Use upsert so repeated runs remain idempotent.
    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print(
        "Chunks indexed successfully."
    )


# ============================================================================
# VALIDATION
# ============================================================================

def validate_index(
    collection,
) -> None:
    """Validate that ChromaDB contains the expected documents."""

    result = collection.get(
        include=[
            "documents",
            "metadatas",
        ]
    )

    document_count = len(
        result["documents"]
    )

    if document_count == 0:
        raise RuntimeError(
            "ChromaDB collection is empty."
        )

    print(
        "\n=== INDEX VALIDATION ==="
    )

    print(
        f"Stored chunks: "
        f"{document_count}"
    )

    print(
        f"Collection name: "
        f"{COLLECTION_NAME}"
    )

    print(
        f"Vector store: "
        f"{VECTOR_STORE_DIR}"
    )

    sample_documents = (
        result["documents"][
            :3
        ]
    )

    print(
        "\nSample indexed chunks:"
    )

    for index, document in enumerate(
        sample_documents,
        start=1,
    ):

        preview = (
            document
            .replace(
                "\n",
                " ",
            )
        )

        if len(preview) > 180:
            preview = (
                preview[:180]
                + "..."
            )

        print(
            f"{index}. {preview}"
        )


# ============================================================================
# STANDALONE RETRIEVAL SMOKE TEST
# ============================================================================

def run_retrieval_test(
    collection,
    model: SentenceTransformer,
) -> None:
    """Verify semantic retrieval before the LangGraph layer exists."""

    test_queries = [
        (
            "What features are driving the "
            "overall revenue forecast?"
        ),
        (
            "Which category forecasts have "
            "the strongest recent performance?"
        ),
        (
            "What is the current model "
            "monitoring status?"
        ),
    ]

    print(
        "\n=== RETRIEVAL SMOKE TEST ==="
    )

    for query in test_queries:

        query_embedding = model.encode(
            [query],
            normalize_embeddings=True,
        )

        result = collection.query(
            query_embeddings=(
                query_embedding
                .tolist()
            ),
            n_results=3,
        )

        print(
            f"\nQuery: {query}"
        )

        documents = result.get(
            "documents",
            [[]],
        )[0]

        metadatas = result.get(
            "metadatas",
            [[]],
        )[0]

        for rank, (
            document,
            metadata,
        ) in enumerate(
            zip(
                documents,
                metadatas,
            ),
            start=1,
        ):

            preview = (
                document
                .replace(
                    "\n",
                    " ",
                )
            )

            if len(preview) > 160:
                preview = (
                    preview[:160]
                    + "..."
                )

            print(
                f"{rank}. "
                f"[{metadata.get('source')}] "
                f"{preview}"
            )


# ============================================================================
# MAIN
# ============================================================================

def run_rag_indexing() -> None:
    """Build the complete persistent RAG index."""

    print(
        "=== RAG INDEX BUILDER ==="
    )

    print(
        f"Knowledge base: "
        f"{KNOWLEDGE_BASE_DIR}"
    )

    print(
        f"Vector store: "
        f"{VECTOR_STORE_DIR}"
    )

    print(
        f"Embedding model: "
        f"{EMBEDDING_MODEL_NAME}"
    )

    # ---------------------------------------------------------------
    # Build chunks
    # ---------------------------------------------------------------

    chunks = build_chunks()

    print(
        f"\nTotal chunks generated: "
        f"{len(chunks)}"
    )

    # ---------------------------------------------------------------
    # Load embedding model
    # ---------------------------------------------------------------

    model = load_embedding_model()

    # ---------------------------------------------------------------
    # Create ChromaDB collection
    # ---------------------------------------------------------------

    client = create_chroma_client()

    collection = get_collection(
        client
    )

    # ---------------------------------------------------------------
    # Index
    # ---------------------------------------------------------------

    index_chunks(
        chunks=chunks,
        model=model,
        collection=collection,
    )

    # ---------------------------------------------------------------
    # Validate
    # ---------------------------------------------------------------

    validate_index(
        collection
    )

    # ---------------------------------------------------------------
    # Retrieval smoke test
    # ---------------------------------------------------------------

    run_retrieval_test(
        collection=collection,
        model=model,
    )

    print(
        "\n=== RAG INDEXING COMPLETE ==="
    )

    print(
        f"Persistent ChromaDB path:\n"
        f"{VECTOR_STORE_DIR}"
    )


if __name__ == "__main__":
    run_rag_indexing()