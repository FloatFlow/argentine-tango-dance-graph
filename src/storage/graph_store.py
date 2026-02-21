import json
import os
from typing import Dict, Any, List, Optional
from loguru import logger
import gin
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from src.core.schemas import VideoContent

@gin.configurable
class GraphStore:
    def __init__(self, db_file: str = "graph_db.json"):
        self.db_file = db_file
        # Maps video_id -> VideoContent dict
        self.videos: Dict[str, Dict] = {} 
        # Maps dancer_name -> { 
        #   "videos": [id], 
        #   "partners": {name: count}, 
        #   "events": {name: count}, 
        #   "tags": {tag: count}, 
        #   "style_embedding": {x, y},
        #   "similar_dancers": [{name, score}]  <-- Pre-computed neighbors
        # }
        self.dancers: Dict[str, Dict] = {}
        
        self.load()

    def add_video(self, video_id: str, content: VideoContent, title: str, duration: int = 0, thumbnail: str = "", tags: List[str] = None):
        if video_id in self.videos:
            return

        # Store video metadata
        data = content.model_dump()
        data['title'] = title
        data['url'] = f"https://www.youtube.com/watch?v={video_id}"
        data['duration'] = duration
        data['thumbnail'] = thumbnail
        data['tags'] = tags or []
        
        self.videos[video_id] = data

        # Update Dancer Graph
        dancer_names = [d.name for d in content.dancers]
        event_name = content.event.name if content.event else None
        
        # Register dancers and update basic stats
        for dancer in content.dancers:
            if dancer.name not in self.dancers:
                self.dancers[dancer.name] = {
                    "videos": [], 
                    "partners": {}, 
                    "events": {}, 
                    "tags": {}, 
                    "roles": set(),
                    "similar_dancers": [] 
                }
            
            entry = self.dancers[dancer.name]
            entry["videos"].append(video_id)
            if dancer.role:
                entry["roles"].add(dancer.role)

            # Update partners
            for partner_name in dancer_names:
                if partner_name != dancer.name:
                    entry["partners"][partner_name] = entry["partners"].get(partner_name, 0) + 1
            
            # Update Event Attendance
            if event_name:
                entry["events"][event_name] = entry["events"].get(event_name, 0) + 1

            # Update Tags
            if tags:
                for tag in tags:
                    clean_tag = tag.lower().strip()
                    entry["tags"][clean_tag] = entry["tags"].get(clean_tag, 0) + 1

        self.save()

    def get_stats(self):
        return {
            "total_videos": len(self.videos),
            "total_dancers": len(self.dancers)
        }

    def get_dancer_profile(self, name: str) -> Optional[Dict]:
        if name not in self.dancers:
            return None
        return self.dancers[name]

    def update_style_embeddings(self):
        """
        Runs the heavy math (TF-IDF, Cosine Similarity, t-SNE) in batch.
        Updates both the 2D coordinates AND the nearest-neighbor lists.
        """
        all_dancers = list(self.dancers.keys())
        # Filter for active dancers to reduce noise (must have > 1 video)
        active_dancers = [d for d in all_dancers if len(self.dancers[d].get('videos', [])) > 1]

        if len(active_dancers) < 5:
            return
        
        try:
            # 1. Build Feature Space
            # We combine Partners + Events into a single feature set
            all_features = set()
            
            # Collect all possible features first
            for d in active_dancers:
                all_features.update(self.dancers[d].get("partners", {}).keys())
                all_features.update(self.dancers[d].get("events", {}).keys())
            
            feature_list = sorted(list(all_features))
            feature_map = {name: i for i, name in enumerate(feature_list)}
            
            # Build Sparse Matrix (using list of lists for simplicity)
            data_matrix = []
            for dancer in active_dancers:
                row = [0] * len(feature_list)
                d_data = self.dancers[dancer]
                
                # Partners
                for p, count in d_data.get("partners", {}).items():
                    if p in feature_map:
                        row[feature_map[p]] = count
                
                # Events (Weight x2)
                for e, count in d_data.get("events", {}).items():
                    if e in feature_map:
                        row[feature_map[e]] = count * 2
                        
                data_matrix.append(row)
            
            # 2. TF-IDF Transformation
            # Down-weights features that appear for everyone (e.g. "Mundial")
            tfidf = TfidfTransformer()
            tfidf_matrix = tfidf.fit_transform(data_matrix)
            
            # 3. Compute Pairwise Similarity (Cosine on TF-IDF vectors)
            # This replaces the runtime Jaccard calculation
            sim_matrix = cosine_similarity(tfidf_matrix)
            
            # Store Top 5 Neighbors for each dancer
            for i, dancer_name in enumerate(active_dancers):
                scores = []
                for j, other_name in enumerate(active_dancers):
                    if i == j: continue
                    score = sim_matrix[i][j]
                    if score > 0.05:
                        scores.append({"name": other_name, "score": float(round(score, 3))})
                
                # Sort and slice
                scores.sort(key=lambda x: x["score"], reverse=True)
                self.dancers[dancer_name]["similar_dancers"] = scores[:5]

            # 4. t-SNE Calculation (for Visualization)
            # Use the TF-IDF matrix as input
            n_samples = len(active_dancers)
            perplexity = min(30, n_samples - 1)
            
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init='pca', learning_rate='auto')
            components = tsne.fit_transform(tfidf_matrix.toarray())
            
            # Store Coordinates
            for idx, dancer in enumerate(active_dancers):
                self.dancers[dancer]["style_embedding"] = {
                    "x": float(components[idx][0]),
                    "y": float(components[idx][1])
                }
            
            self.save()
            logger.info(f"Updated style embeddings & similarities for {len(active_dancers)} dancers.")
        except Exception as e:
            logger.error(f"Failed to update style embeddings: {e}")

    def merge_dancers(self, source_name: str, target_name: str):
        if source_name not in self.dancers or source_name == target_name:
            return
        
        if target_name not in self.dancers:
            self.dancers[target_name] = {
                "videos": [], "partners": {}, "events": {}, "tags": {}, "roles": set(), "similar_dancers": []
            }

        source = self.dancers[source_name]
        target = self.dancers[target_name]

        # Merge simple sets/lists
        target["videos"] = list(set(target["videos"]) | set(source["videos"]))
        target["roles"] = target.get("roles", set()) | source.get("roles", set())

        # Merge Dictionaries (Partners, Events, Tags)
        for dict_name in ["partners", "events", "tags"]:
            target_dict = target.get(dict_name, {})
            source_dict = source.get(dict_name, {})
            for k, v in source_dict.items():
                if dict_name == "partners" and k == target_name: continue
                target_dict[k] = target_dict.get(k, 0) + v
            target[dict_name] = target_dict

        # Update external references
        for other_dancer_name in self.dancers:
            if other_dancer_name == source_name: continue
            
            other_dancer = self.dancers[other_dancer_name]
            if source_name in other_dancer.get("partners", {}):
                count = other_dancer["partners"].pop(source_name)
                if other_dancer_name != target_name:
                    other_dancer["partners"][target_name] = other_dancer["partners"].get(target_name, 0) + count

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
                "events": v.get("events", {}),
                "tags": v.get("tags", {}),
                "roles": list(v.get("roles", [])),
                "style_embedding": v.get("style_embedding"),
                "similar_dancers": v.get("similar_dancers", [])
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