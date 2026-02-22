import json
import os
import re
from loguru import logger

def normalize_query(query: str) -> str:
    """
    Normalizes a query string for deduplication.
    Matches logic in QueueManager.
    """
    if not query:
        return ""
    # Remove non-alphanumeric (keep spaces)
    clean = re.sub(r'[^a-z0-9\s]', '', query.lower())
    # Sort tokens to handle permutations
    tokens = sorted(clean.split())
    return " ".join(tokens)

def reset_system():
    # 1. Delete Corrupted Graph & Aliases
    for file in ["graph_db.json", "aliases.json"]:
        if os.path.exists(file):
            os.remove(file)
            logger.info(f"Deleted {file}")
        else:
            logger.info(f"{file} not found (clean)")

    # 2. Handle Queue History
    queue_choice = input("Do you want to WIPE the search queue and start fresh from seeds? (yes/no): ")
    
    if queue_choice.lower() == "yes":
        for file in ["queue_state.json", "queue_embeddings.json"]:
            if os.path.exists(file):
                os.remove(file)
                logger.info(f"Deleted {file}")
        logger.success("Queue wiped. System will restart with default seeds.")
    else:
        # Preserve and Clean Queue
        queue_file = "queue_state.json"
        if os.path.exists(queue_file):
            with open(queue_file, "r") as f:
                state = json.load(f)
            
            raw_queue = state.get("queue", [])
            
            # Deduplicate Queue based on Normalized String
            # Map: normalized_str -> [best_priority, original_query]
            deduped_map = {}
            
            for item in raw_queue:
                if isinstance(item, list) and len(item) == 2:
                    p, q = item
                else:
                    # Legacy string support
                    p, q = 10, item
                
                norm = normalize_query(q)
                if not norm: continue
                
                if norm not in deduped_map:
                    deduped_map[norm] = [p, q]
                else:
                    # Keep the version with higher priority (lower number)
                    if p < deduped_map[norm][0]:
                        deduped_map[norm] = [p, q]
            
            final_queue = list(deduped_map.values())
            state["queue"] = final_queue
            
            # Wipe history so videos are re-processed
            state["seen_videos"] = []
            state["seen_channels"] = [] 
            
            with open(queue_file, "w") as f:
                json.dump(state, f, indent=2)
                
            logger.success(f"Reset queue history. Reduced {len(raw_queue)} items to {len(final_queue)} unique queries.")
        else:
            logger.warning("No queue state found.")

if __name__ == "__main__":
    confirmation = input("This will delete your graph and alias data. Type 'yes' to proceed: ")
    if confirmation.lower() == "yes":
        reset_system()
    else:
        print("Operation cancelled.")