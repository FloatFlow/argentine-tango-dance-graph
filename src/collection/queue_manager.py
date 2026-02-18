import json
import os
from collections import deque
from typing import Set, List, Optional
from loguru import logger
import gin

@gin.configurable
class QueueManager:
    def __init__(self, state_file: str = "queue_state.json"):
        self.state_file = state_file
        self.search_queue: deque = deque()
        self.seen_videos: Set[str] = set()
        self.seen_channels: Set[str] = set()
        
        self.load_state()

    def add_query(self, query: str):
        """Adds a new search query to the back of the queue if not already present."""
        if query not in self.search_queue:
            self.search_queue.append(query)
            # logger.debug(f"Added query to queue: {query}")

    def pop_query(self) -> Optional[str]:
        """Get the next query to process."""
        if self.search_queue:
            return self.search_queue.popleft()
        return None

    def mark_video_seen(self, video_id: str):
        self.seen_videos.add(video_id)

    def is_video_seen(self, video_id: str) -> bool:
        return video_id in self.seen_videos

    def save_state(self):
        """Persists the current queue and seen lists to disk."""
        state = {
            "queue": list(self.search_queue),
            "seen_videos": list(self.seen_videos),
            "seen_channels": list(self.seen_channels)
        }
        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2)
        logger.debug(f"Queue state saved. {len(self.seen_videos)} videos in history.")

    def load_state(self):
        """Loads state from disk if available."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    state = json.load(f)
                    self.search_queue = deque(state.get("queue", []))
                    self.seen_videos = set(state.get("seen_videos", []))
                    self.seen_channels = set(state.get("seen_channels", []))
                logger.info(f"Loaded queue state: {len(self.search_queue)} queries, {len(self.seen_videos)} videos seen.")
            except Exception as e:
                logger.error(f"Failed to load queue state: {e}")