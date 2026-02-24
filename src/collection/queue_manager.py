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
        
        # Track the currently executing query to ensure resilience against crashes
        self.active_query: Optional[Tuple[int, str]] = None
        
        self.seen_queries: Set[str] = set()
        
        # Map video_id -> count of times seen in search results
        # Used for Chao1 population estimation (Capture-Recapture)
        self.video_sightings: Dict[str, int] = {}
        
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

    async def add_query(self, query: str, priority: int = 10, llm_client=None) -> bool:
        """
        Adds a new search query to the queue with optional semantic deduplication.
        Returns True if the query was added, False if it was a duplicate or skipped.
        """
        norm_q = self._normalize(query)
        if not norm_q:
            return False

        if norm_q in self.seen_queries:
            # Already processed or pending
            return False
            
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
                        return False
                    
                    # No match, so store this embedding
                    self.query_embeddings[norm_q] = new_embedding

        # Mark as seen immediately
        self.seen_queries.add(norm_q)
        
        # Push to heap
        heapq.heappush(self.queue, (priority, query))
        # logger.debug(f"Added query (p={priority}): {query}")
        return True

    def pop_query(self) -> Optional[str]:
        """
        Get the highest priority query (lowest int value) to process.
        Moves it to 'active_query' so it isn't lost if the script crashes.
        """
        if self.queue:
            priority, query = heapq.heappop(self.queue)
            self.active_query = (priority, query)
            return query
        return None

    def complete_query(self):
        """Marks the current active query as fully processed."""
        self.active_query = None

    def record_sighting(self, video_id: str):
        """Records a sighting of a video ID. Increments count for stats."""
        self.video_sightings[video_id] = self.video_sightings.get(video_id, 0) + 1

    def mark_video_seen(self, video_id: str):
        """Legacy alias for record_sighting to maintain compatibility."""
        self.record_sighting(video_id)

    def is_video_seen(self, video_id: str) -> bool:
        return video_id in self.video_sightings

    def get_population_estimate(self) -> Dict[str, Any]:
        """
        Uses the Chao1 estimator to predict the total number of relevant videos
        based on the collision rate (frequency of seeing the same video).
        """
        f1 = 0 # Singletons (seen once)
        f2 = 0 # Doubletons (seen twice)
        s_obs = len(self.video_sightings)
        
        for count in self.video_sightings.values():
            if count == 1: f1 += 1
            elif count == 2: f2 += 1
            
        # Chao1 Estimator
        if f2 > 0:
            est = s_obs + (f1 ** 2) / (2 * f2)
        else:
            # Bias-corrected form or fallback
            est = s_obs + (f1 * (f1 - 1)) / (2 * (f2 + 1))
            
        return {
            "observed": s_obs,
            "estimated_total": int(est),
            "coverage_percent": round((s_obs / est * 100), 1) if est > 0 else 0.0,
            "singletons": f1,
            "doubletons": f2
        }

    def save_state(self):
        """Persists the current queue and seen lists to disk."""
        state = {
            "queue": self.queue, # list of [prio, query]
            "active_query": self.active_query,
            "seen_queries": list(self.seen_queries),
            "video_sightings": self.video_sightings,
            "seen_channels": list(self.seen_channels)
        }
        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2)
            
        # Save embeddings separately
        if self.query_embeddings:
            with open(self.embedding_file, "w") as f:
                json.dump(self.query_embeddings, f)
                
        logger.debug(f"Queue saved. {len(self.queue)} pending, {len(self.video_sightings)} videos seen.")

    def load_state(self):
        """Loads state from disk if available."""
        # 1. Load Main State
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    state = json.load(f)
                    
                    self.seen_queries = set(state.get("seen_queries", []))
                    self.seen_channels = set(state.get("seen_channels", []))
                    
                    # Handle Migration: seen_videos (list) -> video_sightings (dict)
                    if "video_sightings" in state:
                        self.video_sightings = state["video_sightings"]
                    else:
                        legacy_seen = state.get("seen_videos", [])
                        self.video_sightings = {vid: 1 for vid in legacy_seen}
                        logger.info(f"Migrated {len(legacy_seen)} videos to new sighting counter.")
                    
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
                    
                    # Recovery: If there was an active query when we last saved, put it back.
                    # This handles interruptions/crashes mid-batch.
                    active = state.get("active_query")
                    if active:
                        logger.warning(f"Recovering interrupted query: '{active[1]}'")
                        heapq.heappush(self.queue, tuple(active))
                        self.active_query = None

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