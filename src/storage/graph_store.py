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

        # 0. Disambiguate Dancers based on Partnerships in this video
        # We attempt to resolve short names (e.g. "Lorena") to full names ("Lorena Tarantino")
        # by checking if the specific pair exists in our graph history.
        for p in content.performances:
            if len(p.dancers) == 2:
                d1, d2 = p.dancers[0], p.dancers[1]
                new_n1, new_n2 = self.infer_partnership(d1.name, d2.name)
                
                if new_n1 != d1.name:
                    logger.debug(f"Inferred: {d1.name} -> {new_n1} (via partner {d2.name})")
                    d1.name = new_n1
                if new_n2 != d2.name:
                    logger.debug(f"Inferred: {d2.name} -> {new_n2} (via partner {d1.name})")
                    d2.name = new_n2

        # Store video metadata
        data = content.model_dump()
        data['title'] = title
        data['url'] = f"https://www.youtube.com/watch?v={video_id}"
        data['duration'] = duration
        data['thumbnail'] = thumbnail
        data['tags'] = tags or []
        
        self.videos[video_id] = data

        # Index the video into the dancer graph
        self._index_video(video_id, content, tags)
        self.save()

    def _index_video(self, video_id: str, content: VideoContent, tags: List[str]):
        """
        Helper to update self.dancers from a video record.
        Used by add_video and normalize_graph.
        """
        event_name = content.event.name if content.event else None
        
        # 1. Gather all dancers for global stats (Appearance in Video / Event)
        all_dancers_in_video = []
        for p in content.performances:
            all_dancers_in_video.extend(p.dancers)
            
        for dancer in all_dancers_in_video:
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
            if video_id not in entry["videos"]:
                entry["videos"].append(video_id)
            
            if dancer.role:
                entry["roles"].add(dancer.role)

            # Update Event Attendance
            if event_name:
                entry["events"][event_name] = entry["events"].get(event_name, 0) + 1

            # Update Tags
            if tags:
                for tag in tags:
                    clean_tag = tag.lower().strip()
                    entry["tags"][clean_tag] = entry["tags"].get(clean_tag, 0) + 1

        # 2. Update Partnerships (Scoped to Performance Unit)
        for p in content.performances:
            p_names = [d.name for d in p.dancers]
            if len(p_names) < 2:
                continue
                
            for d_obj in p.dancers:
                entry = self.dancers[d_obj.name]
                for partner_name in p_names:
                    if partner_name == d_obj.name:
                        continue
                    entry["partners"][partner_name] = entry["partners"].get(partner_name, 0) + 1

    def normalize_graph(self, resolver=None):
        """
        Maintenance routine to:
        1. Migrate legacy schema (flat dancers -> performances)
        2. Retroactively apply partnership inference to old videos
        3. Apply Entity Resolution (Aliases) if resolver provided
        4. Rebuild the dancer index from scratch to ensure consistency
        """
        logger.info("--- Running Graph Normalization ---")
        updates_count = 0
        
        # 1. Iterate all videos to fix Schema & Disambiguate
        for vid_id, data in self.videos.items():
            # Load into Pydantic to access validator logic (auto-migration)
            try:
                content = VideoContent(**data)
            except Exception as e:
                logger.warning(f"Skipping malformed video {vid_id}: {e}")
                continue
            
            changed = False

            # A. Schema Migration is handled by Pydantic model_validator on load
            if "dancers" in data and not data.get("performances"):
                changed = True

            # B. Entity Resolution (Aliases)
            if resolver:
                # Events
                if content.event and content.event.name:
                    orig_evt = content.event.name
                    content.event.name = resolver.resolve_event(orig_evt)
                    if content.event.name != orig_evt:
                        changed = True

                # Videographers
                if content.videographer:
                    orig_vid = content.videographer
                    content.videographer = resolver.resolve(orig_vid, "videographers")
                    if content.videographer != orig_vid:
                        changed = True

                # Dancers (Explicit Alias Check)
                for p in content.performances:
                    for d in p.dancers:
                        orig_d = d.name
                        d.name = resolver.resolve_dancer(orig_d)
                        if d.name != orig_d:
                            changed = True

            # C. Disambiguation (Partnership Inference)
            for p in content.performances:
                if len(p.dancers) == 2:
                    d1, d2 = p.dancers[0], p.dancers[1]
                    # Try to infer better names using the CURRENT graph state
                    # (Note: This uses the existing self.dancers index before we rebuild it)
                    new_n1, new_n2 = self.infer_partnership(d1.name, d2.name)
                    
                    if new_n1 != d1.name:
                        d1.name = new_n1
                        changed = True
                    if new_n2 != d2.name:
                        d2.name = new_n2
                        changed = True
            
            if changed:
                updates_count += 1
                # Merge back into the raw dict
                dumped = content.model_dump()
                data.update(dumped)
                # Remove legacy key if it exists to keep DB clean
                if "dancers" in data and "performances" in data:
                    data.pop("dancers", None)

        logger.info(f"Normalized {updates_count} videos.")

        # 2. Rebuild Dancer Index from scratch
        # This ensures all counts/partnerships are clean after name changes
        self.dancers = {}
        for vid_id, data in self.videos.items():
            try:
                content = VideoContent(**data)
                tags = data.get("tags", [])
                self._index_video(vid_id, content, tags)
            except Exception as e:
                logger.error(f"Failed to re-index video {vid_id}: {e}")
        
        self.save()
        logger.info("Graph Normalization Complete.")

    def get_stats(self):
        return {
            "total_videos": len(self.videos),
            "total_dancers": len(self.dancers)
        }

    def infer_partnership(self, name_a: str, name_b: str) -> tuple[str, str]:
        """
        Attempts to resolve full names for a pair of dancers by checking 
        if any known partnership in the graph matches the provided names.
        """
        # Helper to find candidates in the DB
        def get_candidates(query_name):
            query_norm = query_name.lower().strip()
            candidates = []
            for known_name in self.dancers.keys():
                # Exact match
                if query_norm == known_name.lower():
                    return [known_name] # Found exact, stop looking
                
                # Token match (e.g. 'Lorena' in 'Lorena Tarantino')
                # We require the query to be a significant prefix or word match
                known_norm = known_name.lower()
                known_parts = known_norm.split()
                if query_norm in known_parts:
                    candidates.append(known_name)
            return candidates

        cands_a = get_candidates(name_a)
        cands_b = get_candidates(name_b)

        # If no candidates found for either, we can't infer anything -> return originals
        if not cands_a and not cands_b:
            return name_a, name_b

        # If one has no candidates, assume the extracted name is a new/unknown dancer
        if not cands_a: cands_a = [name_a]
        if not cands_b: cands_b = [name_b]

        best_score = 0
        best_pair = (name_a, name_b)

        for ca in cands_a:
            data_a = self.dancers.get(ca)
            if not data_a: continue
            
            partners_a = data_a.get("partners", {})
            
            for cb in cands_b:
                # Check if they have danced together
                if cb in partners_a:
                    score = partners_a[cb]
                    if score > best_score:
                        best_score = score
                        best_pair = (ca, cb)
        
        # Only update if we found a known link (score > 0)
        if best_score > 0:
            return best_pair
            
        return name_a, name_b

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