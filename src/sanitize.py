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

def is_safe_text(text: str) -> bool:
    forbidden = ["salsa", "bachata", "kizomba", "zouk", "west coast swing", "ballroom"]
    if not text:
        return True
    text_lower = text.lower()
    return not any(word in text_lower for word in forbidden)

def clean_query_text(query: str) -> str:
    """
    Removes ordinals and years from event names.
    """
    # Remove 1st, 2nd, 3rd, 4th, 10th, 21st, etc.
    cleaned = re.sub(r'\b\d+(?:st|nd|rd|th)\b', '', query, flags=re.IGNORECASE)
    # Collapse multiple spaces
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def rebuild_dancers_index(videos_dict):
    """
    Reconstructs the dancers dictionary from a clean list of videos.
    """
    dancers = {}
    for vid_id, data in videos_dict.items():
        performances = data.get("performances", [])
        tags = data.get("tags", [])
        event = data.get("event")
        event_name = event.get("name") if event else None
        
        # 1. Global Stats
        all_dancers_in_video = []
        for p in performances:
            # p is a dict here
            p_dancers = p.get("dancers", [])
            all_dancers_in_video.extend(p_dancers)
            
        for dancer_data in all_dancers_in_video:
            name = dancer_data.get("name")
            role = dancer_data.get("role")
            
            if name not in dancers:
                dancers[name] = {
                    "videos": [], 
                    "partners": {}, 
                    "events": {}, 
                    "tags": {}, 
                    "roles": [], # Use list here, will be converted to set in GraphStore load if needed
                    "similar_dancers": [] 
                }
            
            entry = dancers[name]
            if vid_id not in entry["videos"]:
                entry["videos"].append(vid_id)
            
            if role and role not in entry["roles"]:
                entry["roles"].append(role)

            if event_name:
                entry["events"][event_name] = entry["events"].get(event_name, 0) + 1

            for tag in tags:
                clean_tag = tag.lower().strip()
                entry["tags"][clean_tag] = entry["tags"].get(clean_tag, 0) + 1

        # 2. Partnerships
        for p in performances:
            p_dancers = p.get("dancers", [])
            p_names = [d.get("name") for d in p_dancers]
            
            if len(p_names) < 2: continue
            
            for d_data in p_dancers:
                d_name = d_data.get("name")
                entry = dancers[d_name]
                for partner_name in p_names:
                    if partner_name == d_name: continue
                    entry["partners"][partner_name] = entry["partners"].get(partner_name, 0) + 1
                    
    return dancers

def reset_system():
    # 1. Sanitize Graph
    kept_videos = {}
    dropped_videos = 0
    
    if os.path.exists("graph_db.json"):
        with open("graph_db.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        
        videos = data.get("videos", {})
        logger.info(f"Scanning {len(videos)} videos for non-Tango content...")
        
        for vid_id, v_data in videos.items():
            title = v_data.get("title", "")
            tags = " ".join(v_data.get("tags", []))
            
            if is_safe_text(title) and is_safe_text(tags):
                kept_videos[vid_id] = v_data
            else:
                dropped_videos += 1
        
        logger.info(f"Rebuilding dancer index from {len(kept_videos)} clean videos...")
        kept_dancers = rebuild_dancers_index(kept_videos)
        
        new_db = {"videos": kept_videos, "dancers": kept_dancers}
        with open("graph_db.json", "w", encoding="utf-8") as f:
            json.dump(new_db, f, indent=2, ensure_ascii=False)
            
        logger.success(f"Graph sanitized. Removed {dropped_videos} bad videos. Kept {len(kept_videos)} videos and {len(kept_dancers)} dancers.")
    else:
        logger.warning("graph_db.json not found. Starting with empty graph.")

    # 2. Reset Aliases (Force re-learning with new logic)
    if os.path.exists("aliases.json"):
        os.remove("aliases.json")
        logger.info("Deleted aliases.json to force fresh entity resolution.")

    # 3. Clean Queue
    queue_file = "queue_state.json"
    if os.path.exists(queue_file):
        with open(queue_file, "r") as f:
            state = json.load(f)
        
        raw_queue = state.get("queue", [])
        
        # Deduplicate AND Filter Queue
        deduped_map = {}
        skipped_count = 0
        cleaned_count = 0
        
        for item in raw_queue:
            if isinstance(item, list) and len(item) == 2:
                p, q = item
            else:
                p, q = 10, item
            
            # Filter Content (Salsa/Bachata)
            if not is_safe_text(q):
                skipped_count += 1
                continue

            # Heuristic Cleaning
            clean_q = clean_query_text(q)
            if clean_q != q:
                cleaned_count += 1
                q = clean_q

            norm = normalize_query(q)
            if not norm: continue
            
            if norm not in deduped_map:
                deduped_map[norm] = [p, q]
            else:
                if p < deduped_map[norm][0]:
                    deduped_map[norm] = [p, q]
        
        final_queue = list(deduped_map.values())
        state["queue"] = final_queue
        
        # CRITICAL: Sync seen videos with what we kept in the graph
        # This prevents re-processing the videos we already have
        state["video_sightings"] = {vid: 1 for vid in kept_videos.keys()}
        state["seen_videos"] = [] # Legacy clear
        state["seen_channels"] = []
        
        # Clear seen_queries to allow re-discovery of seeds if needed
        state["seen_queries"] = []
        
        with open(queue_file, "w") as f:
            json.dump(state, f, indent=2)
            
        logger.success(f"Queue groomed. Removed {skipped_count} salsa queries. Retained {len(final_queue)} pending tasks.")
        logger.info(f"Synced {len(kept_videos)} seen videos to queue state.")
    else:
        logger.warning("No queue state found.")

if __name__ == "__main__":
    confirmation = input("This will SANITIZE your graph (remove salsa) and CLEAN your queue. Type 'yes' to proceed: ")
    if confirmation.lower() == "yes":
        reset_system()
    else:
        print("Operation cancelled.")