import json
import os
import heapq
import re
from typing import Set, List, Optional, Tuple, Any
from loguru import logger
import gin

@gin.configurable
class QueueManager:
    def __init__(self, state_file: str = "queue_state.json"):
        self.state_file = state_file
        # Heap of [priority, query_string]. Python's heapq sorts by the first element.
        # Lower priority number = Higher importance.
        self.queue: List[Tuple[int, str]] = []
        
        self.seen_queries: Set[str] = set()
        self.seen_videos: Set[str] = set()
        self.seen_channels: Set[str] = set()
        
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

    def add_query(self, query: str, priority: int = 10):
        """
        Adds a new search query to the queue.
        Priority 1 = High (Exploit new entity)
        Priority 10 = Low (Explore/Suggestions)
        """
        norm_q = self._normalize(query)
        if not norm_q:
            return

        if norm_q in self.seen_queries:
            # Already processed or pending
            return
            
        # Mark as seen immediately so we don't add duplicates to the queue
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
        logger.debug(f"Queue saved. {len(self.queue)} pending, {len(self.seen_videos)} videos seen.")

    def load_state(self):
        """Loads state from disk if available."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    state = json.load(f)
                    
                    # Load History
                    self.seen_queries = set(state.get("seen_queries", []))
                    self.seen_videos = set(state.get("seen_videos", []))
                    self.seen_channels = set(state.get("seen_channels", []))
                    
                    # Load Queue with Migration Support
                    raw_queue = state.get("queue", [])
                    self.queue = []
                    
                    if raw_queue:
                        # Check format of first item
                        first = raw_queue[0]
                        if isinstance(first, str):
                            # LEGACY MIGRATION: List of strings -> List of [10, string]
                            logger.info("Migrating legacy queue format...")
                            for q in raw_queue:
                                # Dedupe against seen_queries during migration
                                norm = self._normalize(q)
                                if norm not in self.seen_queries:
                                    heapq.heappush(self.queue, (10, q))
                                    self.seen_queries.add(norm)
                        else:
                            # Standard Format: List of [prio, string]
                            # We must re-heapify because JSON load returns a plain list
                            for item in raw_queue:
                                # item is [priority, query]
                                heapq.heappush(self.queue, tuple(item))

                logger.info(f"Loaded queue state: {len(self.queue)} queries pending.")
            except Exception as e:
                logger.error(f"Failed to load queue state: {e}")