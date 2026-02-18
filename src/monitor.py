import json
import os
from collections import Counter

def load_json(filepath: str):
    if not os.path.exists(filepath):
        return {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return {}

def generate_report():
    print("\n=== TANGO GRAPH MONITOR ===\n")
    
    # 1. Load Data
    graph_db = load_json("graph_db.json")
    queue_state = load_json("queue_state.json")
    aliases = load_json("aliases.json")
    
    videos = graph_db.get("videos", {})
    dancers = graph_db.get("dancers", {})
    
    if not videos:
        print("No data found in graph_db.json yet.")
        return

    # 2. Corpus Stats
    total_videos = len(videos)
    total_dancers = len(dancers)
    
    print(f"--- Corpus Stats ---")
    print(f"Total Videos Processed: {total_videos}")
    print(f"Total Unique Dancers:   {total_dancers}")
    
    # 3. Extraction Health
    # Check how many videos successfully yielded dancer entities
    videos_with_dancers = sum(1 for v in videos.values() if v.get('dancers'))
    videos_with_event = sum(1 for v in videos.values() if v.get('event') and v['event'].get('name'))
    
    if total_videos > 0:
        dancer_coverage = (videos_with_dancers / total_videos) * 100
        event_coverage = (videos_with_event / total_videos) * 100
    else:
        dancer_coverage = 0
        event_coverage = 0
        
    print(f"Videos with Dancers:    {videos_with_dancers} ({dancer_coverage:.1f}%)")
    print(f"Videos with Events:     {videos_with_event} ({event_coverage:.1f}%)")
    
    # 4. Top Entities (Sanity Check)
    # If musicians appear here, the extraction prompt needs tuning
    print(f"\n--- Top Dancers (by video count) ---")
    # Sort dancers by length of their 'videos' list
    # The graph store structure for dancers is {name: {videos: [], ...}}
    sorted_dancers = sorted(dancers.items(), key=lambda x: len(x[1].get('videos', [])), reverse=True)
    
    for name, data in sorted_dancers[:15]:
        count = len(data.get('videos', []))
        print(f"{count:<4} | {name}")
        
    # 5. Top Events
    print(f"\n--- Top Events ---")
    event_counter = Counter()
    for v in videos.values():
        if v.get('event') and v['event'].get('name'):
            event_counter[v['event']['name']] += 1
            
    for name, count in event_counter.most_common(10):
        print(f"{count:<4} | {name}")

    # 6. Queue Health
    queue = queue_state.get("queue", [])
    seen_videos = queue_state.get("seen_videos", [])
    seen_channels = queue_state.get("seen_channels", [])
    
    print(f"\n--- Discovery Engine ---")
    print(f"Queue Length:           {len(queue)}")
    print(f"Total History (Seen):   {len(seen_videos)}")
    
    # 7. Alias Health
    print(f"\n--- Entity Resolution ---")
    # Check if aliases is the new nested format or old flat format
    if "dancers" in aliases and isinstance(aliases["dancers"], dict):
        print(f"Dancer Aliases:         {len(aliases['dancers'])}")
        print(f"Event Aliases:          {len(aliases.get('events', {}))}")
        print(f"Videographer Aliases:   {len(aliases.get('videographers', {}))}")
    else:
        # Fallback/Legacy
        print(f"Aliases defined:        {len(aliases)}")

if __name__ == "__main__":
    generate_report()