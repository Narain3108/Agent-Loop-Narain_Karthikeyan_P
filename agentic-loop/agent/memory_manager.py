import uuid
import chromadb
from typing import Dict, Any, List

class MemoryManager:
    """
    Manages both short-term session memory (Python list) and 
    long-term semantic memory (ChromaDB Vector Store).
    """
    def __init__(self):
        # 1. Ephemeral session memory (holds traces from previous iteration)
        self.session_mem: List[Dict[str, Any]] = []
        
        # 2. Vector memory (runs entirely in-memory as requested)
        self.chroma_client = chromadb.EphemeralClient()
        
        # Create or get a collection for this session
        self.collection_name = "agentic_summaries"
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name
        )
        print("[MemoryManager] Initialized Ephemeral ChromaDB and Session Memory.")

    def save(self, iteration_data: Dict[str, Any], summary_text: str = None) -> None:
        """
        Saves the complete context of an iteration.
        If a new summary was generated, it embeds it into ChromaDB.
        """
        # Save to short-term session memory
        self.session_mem.append(iteration_data)
        
        # Keep only the last 3 iterations in short term memory to avoid context bloat
        if len(self.session_mem) > 3:
            self.session_mem = self.session_mem[-3:]

        # Save generated summary to vector database for semantic recall
        if summary_text:
            doc_id = str(uuid.uuid4())
            self.collection.add(
                documents=[summary_text],
                metadatas=[{"iteration": len(self.session_mem)}],
                ids=[doc_id]
            )
            print(f"[MemoryManager] Saved 1 summary to ChromaDB with ID {doc_id}")

    def recall(self, query: str = None, n_results: int = 2) -> Dict[str, Any]:
        """
        Recalls the session memory and performs a semantic search on ChromaDB.
        Returns a combined dictionary.
        """
        recalled_memory = {
            "recent_session_history": self.session_mem,
            "relevant_past_summaries": []
        }

        # If we have a specific query and documents exist in the collection
        if query and self.collection.count() > 0:
            # We must not request more results than we have in the DB
            num_docs = min(self.collection.count(), n_results)
            results = self.collection.query(
                query_texts=[query],
                n_results=num_docs
            )
            
            # Extract the raw document texts from the Chroma response
            if results and results.get("documents") and len(results["documents"]) > 0:
                recalled_memory["relevant_past_summaries"] = results["documents"][0]
                print(f"[MemoryManager] Recalled {len(recalled_memory['relevant_past_summaries'])} past summaries.")

        return recalled_memory

    def clear(self) -> None:
        """
        Wipes both the short-term and long-term memory for the next session.
        """
        # Clear list
        self.session_mem.clear()
        
        # Delete vector collection
        try:
            self.chroma_client.delete_collection(name=self.collection_name)
            # Recreate an empty one just in case the agent is reused immediately
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name
            )
            print("[MemoryManager] Cleared session memory and deleted ChromaDB collection.")
        except Exception as e:
            print(f"[MemoryManager] Error clearing ChromaDB: {e}")
