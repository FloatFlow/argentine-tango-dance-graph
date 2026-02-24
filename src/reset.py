import os
import json
from loguru import logger

def get_system_stats():
    stats = {
        "videos": 0,
        "dancers": 0,
        "queue": 0,
        "seen": 0
    }
    
    # Check Graph
    if os.path.exists("graph_db.json"):
        try:
            with open("graph_db.json", "r", encoding="utf-8") as f:
                data = json.load(f)
                stats["videos"] = len(data.get("videos", {}))
                stats["dancers"] = len(data.get("dancers", {}))
        except Exception as e:
            logger.warning(f"Could not read graph_db.json: {e}")

    # Check Queue
    if os.path.exists("queue_state.json"):
        try:
            with open("queue_state.json", "r") as f:
                state = json.load(f)
                stats["queue"] = len(state.get("queue", []))
                # Handle legacy vs new formats for seen
                seen = state.get("video_sightings", state.get("seen_videos", []))
                stats["seen"] = len(seen)
        except Exception as e:
            logger.warning(f"Could not read queue_state.json: {e}")
            
    return stats

def full_reset():
    files_to_wipe = [
        "graph_db.json",
        "aliases.json",
        "queue_state.json",
        "queue_embeddings.json"
    ]
    
    deleted_count = 0
    for file in files_to_wipe:
        if os.path.exists(file):
            os.remove(file)
            logger.info(f"Deleted {file}")
            deleted_count += 1
        else:
            logger.info(f"{file} not found (already clean)")
            
    if deleted_count > 0:
        logger.success("System fully reset. Run 'python run.py' to start fresh with default seeds.")
    else:
        logger.warning("Nothing to delete. System is already clean.")

if __name__ == "__main__":
    stats = get_system_stats()
    
    print("!!! WARNING !!!")
    print("This will PERMANENTLY DELETE all collected data:")
    print(f"- {stats['videos']} Videos collected")
    print(f"- {stats['dancers']} Dancers profiled")
    print(f"- {stats['queue']} Queries pending in queue")
    print(f"- {stats['seen']} Video IDs in history")
    print("\nYou will be starting from Day 0.")
    
    confirm = input("Type 'NUKE' to confirm full system reset: ")
    if confirm == "NUKE":
        full_reset()
    else:
        print("Reset cancelled.")