import json
import os
from typing import Dict, Any, List, Optional
from loguru import logger
import gin
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import Normalizer
from src.core.schemas import VideoContent

@gin.configurable
class GraphStore:
    def __init__(self, db_file: str = "graph_db.json"):
        self.db_file = db_file
        # Maps video_id -> VideoContent dict
        self.videos: Dict[str, Dict] = {} 
        # Maps dancer_name -> { "videos": [id], "partners": {name: count} }
        self.dancers: Dict[str, Dict] = {}
        
        self.load()

    def add_video(self, video_id: str, content: VideoContent, title: str, duration: int = 0, thumbnail: str = ""):
        if video_id in self.videos:
            return

        # Store video metadata
        # We merge the LLM extraction with the raw metadata (Title, URL)
        data = content.model_dump()
        data['title'] = title
        data['url'] = f"https://www.youtube.com/watch?v={video_id}"
        data['duration'] = duration
        data['thumbnail'] = thumbnail
        
        self.videos[video_id] = data

        # Update Dancer Graph
        dancer_names = [d.name for d in content.dancers]
        
        # Register dancers and update basic stats
        for dancer in content.dancers:
            if dancer.name not in self.dancers:
                self.dancers[dancer.name] = {"videos": [], "partners": {}, "roles": set()}
            
            entry = self.dancers[dancer.name]
            entry["videos"].append(video_id)
            if dancer.role:
                entry["roles"].add(dancer.role)

            # Update partners (everyone else in this video is a partner)
            for partner_name in dancer_names:
                if partner_name != dancer.name:
                    current_count = entry["partners"].get(partner_name, 0)
                    entry["partners"][partner_name] = current_count + 1

        self.save()

    def get_stats(self):
        return {
            "total_videos": len(self.videos),
            "total_dancers": len(self.dancers)
        }

    def get_dancer_profile(self, name: str) -> Optional[Dict]:
        if name not in self.dancers:
            return None
        
        # Enrich with similarity data
        profile = self.dancers[name].copy()
        profile["similar_dancers"] = self.get_similar_dancers(name)
        return profile

    def get_similar_dancers(self, dancer_name: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Finds dancers who share the same partners.
        Uses Jaccard Similarity on the sets of partners.
        """
        if dancer_name not in self.dancers:
            return []

        target_partners = set(self.dancers[dancer_name]["partners"].keys())
        if not target_partners:
            return []

        scores = []
        for other_name, data in self.dancers.items():
            if other_name == dancer_name:
                continue
            
            other_partners = set(data["partners"].keys())
            # Jaccard Similarity: Intersection / Union
            intersection = target_partners.intersection(other_partners)
            union = target_partners.union(other_partners)
            
            if not union:
                continue
                
            similarity = len(intersection) / len(union)
            
            # Simple threshold to filter noise
            if similarity > 0.0:
                scores.append({"name": other_name, "score": round(similarity, 3), "shared_partners": list(intersection)})

        # Sort desc
        scores.sort(key=lambda x: x["score"], reverse=True)
        return scores[:limit]
    
    def update_style_embeddings(self):
        """
        Pre-computes 2D t-SNE embeddings for all active dancers.
        Includes L2 normalization to account for popularity differences.
        """
        all_dancers = list(self.dancers.keys())
        # Filter for active dancers to reduce noise (must have > 1 video)
        active_dancers = [d for d in all_dancers if len(self.dancers[d].get('videos', [])) > 1]

        if len(active_dancers) < 5:
            # t-SNE needs a few samples to run meaningfully
            return
        
        try:
            # Build Matrix: Rows = Dancers, Cols = Partners (Features)
            data_matrix = []
            for dancer in active_dancers:
                row = []
                partners = self.dancers[dancer].get('partners', {})
                for possible_partner in all_dancers:
                    row.append(partners.get(possible_partner, 0))
                data_matrix.append(row)
            
            # 1. Normalize Rows (L2 Norm)
            # This ensures a dancer with 100 partners doesn't overpower a dancer with 10 partners
            # if the *distribution* of partners is similar.
            normalizer = Normalizer()
            normalized_data = normalizer.fit_transform(data_matrix)
            
            # 2. t-SNE Calculation
            # Perplexity is roughly related to number of neighbors to consider.
            # For small datasets, 5-30 is typical. We cap it at len(samples) - 1.
            n_samples = len(active_dancers)
            perplexity = min(30, n_samples - 1)
            
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init='pca', learning_rate='auto')
            components = tsne.fit_transform(normalized_data)
            
            # Store back to DB
            for idx, dancer in enumerate(active_dancers):
                self.dancers[dancer]["style_embedding"] = {
                    "x": float(components[idx][0]),
                    "y": float(components[idx][1])
                }
            
            self.save()
            logger.info(f"Updated style embeddings (t-SNE) for {len(active_dancers)} dancers.")
        except Exception as e:
            logger.error(f"Failed to update style embeddings: {e}")

    def merge_dancers(self, source_name: str, target_name: str):
        """
        Merges two dancer nodes. 
        Moves all videos/partners from source to target, then deletes source.
        Updates all references in the graph.
        """
        if source_name not in self.dancers or source_name == target_name:
            return
        
        # 1. Ensure target exists
        if target_name not in self.dancers:
            self.dancers[target_name] = {"videos": [], "partners": {}, "roles": set()}

        source = self.dancers[source_name]
        target = self.dancers[target_name]

        # 2. Merge Videos (set union to avoid duplicates)
        target_videos = set(target["videos"]) | set(source["videos"])
        target["videos"] = list(target_videos)

        # 3. Merge Roles
        target["roles"] = target["roles"] | source["roles"]

        # 4. Merge Partners logic
        # Part A: Add source's partners to target
        for partner, count in source["partners"].items():
            if partner == target_name: continue 
            target["partners"][partner] = target["partners"].get(partner, 0) + count

        # Part B: Update external references (The O(N) operation)
        # Go through every other dancer. If they have 'source' as a partner, move that count to 'target'.
        for other_dancer_name in self.dancers:
            if other_dancer_name == source_name: continue
            
            other_dancer = self.dancers[other_dancer_name]
            if source_name in other_dancer["partners"]:
                # Remove reference to old name
                count = other_dancer["partners"].pop(source_name)
                # Add reference to new name (unless it's self-referential)
                if other_dancer_name != target_name:
                    other_dancer["partners"][target_name] = other_dancer["partners"].get(target_name, 0) + count

        # 5. Delete source
        del self.dancers[source_name]
        self.save()
        logger.info(f"Graph Merge: '{source_name}' -> '{target_name}'")

    def save(self):
        # Convert sets to lists for JSON serialization
        dancers_serializable = {}
        for k, v in self.dancers.items():
            dancers_serializable[k] = {
                "videos": v["videos"],
                "partners": v["partners"],
                "roles": list(v["roles"])
            }
            
        data = {
            "videos": self.videos,
            "dancers": dancers_serializable
        }
        with open(self.db_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load(self):
        if os.path.exists(self.db_file):
            try:
                with open(self.db_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.videos = data.get("videos", {})
                    self.dancers = data.get("dancers", {})
                    # Restore sets
                    for d in self.dancers.values():
                        d["roles"] = set(d.get("roles", []))
                logger.info(f"Loaded Graph: {len(self.videos)} videos, {len(self.dancers)} dancers.")
            except Exception as e:
                logger.error(f"Failed to load graph DB: {e}")