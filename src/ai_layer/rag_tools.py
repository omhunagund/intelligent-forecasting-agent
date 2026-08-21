"""
RAG Data Access Tools
=====================

Deterministic semantic retrieval over the project's persistent
ChromaDB knowledge base.

Vector store:
    data/vector_store/

Collection:
    forecasting_business_context

Embedding model:
    all-MiniLM-L6-v2

This module contains no LLM reasoning.
"""

from __future__ import annotations

from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]
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
# LOADERS
# ============================================================================

def load_embedding_model() -> SentenceTransformer:
    """Load the same embedding model used to build the index."""

    return SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )


def load_collection():
    """Load the persistent ChromaDB collection."""

    if not VECTOR_STORE_DIR.is_dir():
        raise FileNotFoundError(
            "Persistent vector store not found:\n"
            f"{VECTOR_STORE_DIR}\n\n"
            "Run rag_index.py first."
        )

    client = chromadb.PersistentClient(
        path=str(
            VECTOR_STORE_DIR
        )
    )

    try:
        collection = client.get_collection(
            name=COLLECTION_NAME
        )

    except Exception as exc:
        raise RuntimeError(
            f"ChromaDB collection "
            f"'{COLLECTION_NAME}' "
            "was not found."
        ) from exc

    return collection


# ============================================================================
# RESULT NORMALIZATION
# ============================================================================

def normalize_retrieval_results(
    result: dict,
) -> list[dict]:
    """Convert ChromaDB query output into a clean agent-facing format."""

    documents = (
        result.get(
            "documents",
            [[]],
        )[0]
    )

    metadatas = (
        result.get(
            "metadatas",
            [[]],
        )[0]
    )

    distances = (
        result.get(
            "distances",
            [[]],
        )[0]
    )

    normalized: list[dict] = []

    for index, document in enumerate(
        documents
    ):

        metadata = (
            metadatas[index]
            if index < len(metadatas)
            else {}
        )

        distance = (
            distances[index]
            if index < len(distances)
            else None
        )

        normalized.append(
            {
                "rank":
                    index + 1,
                "text":
                    str(
                        document
                    ),
                "source":
                    str(
                        metadata.get(
                            "source",
                            "unknown",
                        )
                    ),
                "filename":
                    str(
                        metadata.get(
                            "filename",
                            "unknown",
                        )
                    ),
                "section":
                    str(
                        metadata.get(
                            "section",
                            "unknown",
                        )
                    ),
                "distance":
                    None
                    if distance is None
                    else float(
                        distance
                    ),
            }
        )

    return normalized


# ============================================================================
# PUBLIC RETRIEVAL FUNCTION
# ============================================================================

def retrieve_business_context_data(
    query: str,
    top_k: int = 5,
) -> dict:
    """
    Retrieve the most relevant project-derived knowledge chunks.

    Parameters
    ----------
    query:
        Natural-language business/context question.

    top_k:
        Number of chunks to retrieve.

    Returns
    -------
    dict
        Structured retrieval result with documents and sources.
    """

    query = query.strip()

    if not query:
        raise ValueError(
            "query must not be empty."
        )

    if top_k < 1:
        raise ValueError(
            "top_k must be at least 1."
        )

    # Prevent unnecessarily large retrievals.
    top_k = min(
        top_k,
        10,
    )

    model = load_embedding_model()

    collection = load_collection()

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    result = collection.query(
        query_embeddings=(
            query_embedding
            .tolist()
        ),
        n_results=top_k,
    )

    documents = normalize_retrieval_results(
        result
    )

    sources = list(
        dict.fromkeys(
            document["source"]
            for document in documents
        )
    )

    return {
        "query":
            query,
        "results":
            documents,
        "sources":
            sources,
        "collection":
            COLLECTION_NAME,
        "embedding_model":
            EMBEDDING_MODEL_NAME,
    }