import json
import os
import heapq
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Set, List, Optional, Tuple, Any, Dict
from loguru import logger
import gin

@gin.configurable
class QueueManager:
    def __init__(self, state_file: str = "queue_state.json", embedding_file: str = "queue_embeddings.json"):
        self.state_file = state_file
        self.embedding_file = embedding_file
        # Heap of [priority, query_string]. Python's heapq sorts by the first element.
        # Lower priority number = Higher importance.
        self.queue: List[Tuple[int, str]] = []
        
        self.seen_queries: Set[str] = set()
        self.seen_videos: Set[str] = set()
        self.seen_channels: Set[str] = set()
        
        # Map normalized_query -> embedding_vector
        self.query_embeddings: Dict[str, List[float]] = {}
        
        self.load_state()

    def _normalize(self, query: str) -> str:
        """
        Normalizes a query string for deduplication.
        1. Lowercase
        2. Remove special chars
        3. Sort words alphabetically (handles 'Tango Festival' vs 'Festival Tango')
        """
        if not query:
            return ""
        # Remove non-alphanumeric (keep spaces)
        clean = re.sub(r'[^a-z0-9\s]', '', query.lower())
        # Sort tokens to handle permutations
        tokens = sorted(clean.split())
        return " ".join(tokens)

    async def add_query(self, query: str, priority: int = 10, llm_client=None):
        """
        Adds a new search query to the queue with optional semantic deduplication.
        """
        norm_q = self._normalize(query)
        if not norm_q:
            return

        if norm_q in self.seen_queries:
            # Already processed or pending
            return
            
        # Semantic Deduplication
        if llm_client:
            # Check if we already have a semantic match in history
            if self.query_embeddings:
                # Generate embedding for the NEW query
                new_embedding = await llm_client.get_embedding(norm_q)
                
                if new_embedding:
                    # Check similarity against all seen queries
                    # Converting to matrix is fast enough for <50k items
                    matrix = np.array(list(self.query_embeddings.values()))
                    # Reshape for sklearn: (1, n_features) vs (n_samples, n_features)
                    sims = cosine_similarity([new_embedding], matrix)[0]
                    
                    # Threshold 0.90 covers "Festival Lunar" vs "Lunar Festival" 
                    # without blocking "Mundial 2023" vs "Mundial 2024"
                    if np.max(sims) > 0.90:
                        logger.info(f"Skipping '{query}' (Semantic match found)")
                        # Mark as seen to prevent re-checking
                        self.seen_queries.add(norm_q)
                        return
                    
                    # No match, so store this embedding
                    self.query_embeddings[norm_q] = new_embedding

        # Mark as seen immediately
        self.seen_queries.add(norm_q)
        
        # Push to heap
        heapq.heappush(self.queue, (priority, query))
        # logger.debug(f"Added query (p={priority}): {query}")

    def pop_query(self) -> Optional[str]:
        """Get the highest priority query (lowest int value) to process."""
        if self.queue:
            priority, query = heapq.heappop(self.queue)
            return query
        return None

    def mark_video_seen(self, video_id: str):
        self.seen_videos.add(video_id)

    def is_video_seen(self, video_id: str) -> bool:
        return video_id in self.seen_videos

    def save_state(self):
        """Persists the current queue and seen lists to disk."""
        state = {
            "queue": self.queue, # list of [prio, query]
            "seen_queries": list(self.seen_queries),
            "seen_videos": list(self.seen_videos),
            "seen_channels": list(self.seen_channels)
        }
        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2)
            
        # Save embeddings separately
        if self.query_embeddings:
            with open(self.embedding_file, "w") as f:
                json.dump(self.query_embeddings, f)
                
        logger.debug(f"Queue saved. {len(self.queue)} pending, {len(self.seen_videos)} videos seen.")

    def load_state(self):
        """Loads state from disk if available."""
        # 1. Load Main State
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    state = json.load(f)
                    
                    self.seen_queries = set(state.get("seen_queries", []))
                    self.seen_videos = set(state.get("seen_videos", []))
                    self.seen_channels = set(state.get("seen_channels", []))
                    
                    raw_queue = state.get("queue", [])
                    self.queue = []
                    
                    if raw_queue:
                        first = raw_queue[0]
                        if isinstance(first, str):
                            logger.info("Migrating legacy queue format...")
                            for q in raw_queue:
                                norm = self._normalize(q)
                                if norm not in self.seen_queries:
                                    heapq.heappush(self.queue, (10, q))
                                    self.seen_queries.add(norm)
                        else:
                            for item in raw_queue:
                                heapq.heappush(self.queue, tuple(item))

                logger.info(f"Loaded queue state: {len(self.queue)} queries pending.")
            except Exception as e:
                logger.error(f"Failed to load queue state: {e}")
                
        # 2. Load Embeddings
        if os.path.exists(self.embedding_file):
            try:
                with open(self.embedding_file, "r") as f:
                    self.query_embeddings = json.load(f)
                logger.info(f"Loaded {len(self.query_embeddings)} semantic vectors.")
            except Exception as e:
                logger.error(f"Failed to load embeddings: {e}")