"""ChromaDB persistent client wrapper."""

from pathlib import Path
import chromadb


class NannyChroma:
    """Thin wrapper over chromadb.PersistentClient with one collection."""

    def __init__(self, persist_dir: str | Path, collection_name: str = "nannies"):
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._client.get_or_create_collection(name=collection_name)

    def add(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict],
        embeddings: list[list[float]],
    ) -> None:
        self._collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    def query(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        where: dict | None = None,
    ) -> dict:
        return self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
        )

    def count(self) -> int:
        return self._collection.count()
