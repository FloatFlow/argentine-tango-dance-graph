import json
import os
import asyncio
from typing import Dict, Any, List, Optional
from loguru import logger
import gin
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from src.core.schemas import VideoContent
from scipy.sparse import csr_matrix, hstack
import umap
from sklearn.cluster import HDBSCAN
import colorsys
import matplotlib.colors

@gin.configurable
class GraphStore:
    def __init__(self, db_file: str = "graph_db.json", feature_cache_file: str = "feature_embeddings.json"):
        self.db_file = db_file
        self.feature_cache_file = feature_cache_file
        # Maps video_id -> VideoContent dict
        self.videos: Dict[str, Dict] = {}
        # Maps dancer_name -> { 
        #   "videos": [id], 
        #   "partners": {name: count}, 
        #   "events": {name: count}, 
        #   "tags": {tag: count}, 
        #   "style_embedding": {x, y, cluster, color},
        #   "similar_dancers": [{name, score}]
        # }
        self.dancers: Dict[str, Dict] = {}
        # Pre-computed stats for dashboard
        self.stats: Dict[str, Any] = {}
        # Cache for semantic embeddings of features (tags/orchestras)
        self.feature_embeddings: Dict[str, List[float]] = {}
        
        self.load()

    def add_video(self, video_id: str, content: VideoContent, title: str, duration: int = 0, thumbnail: str = "", tags: List[str] = None):
        if video_id in self.videos:
            return

        # 0. Disambiguate Dancers based on Partnerships in this video
        for p in content.performances:
            if len(p.dancers) == 2:
                d1, d2 = p.dancers[0], p.dancers[1]
                new_n1, new_n2 = self.infer_partnership(d1.name, d2.name)
                
                if new_n1 != d1.name:
                    d1.name = new_n1
                if new_n2 != d2.name:
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
        event_name = content.event.name if content.event else None
        orchestra = content.music.orchestra if content.music else None
        music_genre = content.music.genre if content.music else None
        v_type = content.video_type
        
        all_dancers_in_video = []
        for p in content.performances:
            all_dancers_in_video.extend(p.dancers)
            
        for dancer in all_dancers_in_video:
            if dancer.name not in self.dancers:
                self.dancers[dancer.name] = {
                    "videos": [], 
                    "partners": {}, 
                    "events": {}, 
                    "orchestras": {}, 
                    "music_genres": {}, 
                    "video_types": {}, 
                    "tags": {}, 
                    "roles": set(),
                    "similar_dancers": [] 
                }
            
            entry = self.dancers[dancer.name]
            if video_id not in entry["videos"]:
                entry["videos"].append(video_id)
            
            if dancer.role:
                entry["roles"].add(dancer.role)

            if event_name:
                entry["events"][event_name] = entry["events"].get(event_name, 0) + 1

            if orchestra:
                if "orchestras" not in entry: entry["orchestras"] = {}
                entry["orchestras"][orchestra] = entry["orchestras"].get(orchestra, 0) + 1

            if music_genre:
                if "music_genres" not in entry: entry["music_genres"] = {}
                entry["music_genres"][music_genre] = entry["music_genres"].get(music_genre, 0) + 1

            if v_type:
                if "video_types" not in entry: entry["video_types"] = {}
                entry["video_types"][v_type] = entry["video_types"].get(v_type, 0) + 1

            if tags:
                for tag in tags:
                    clean_tag = tag.lower().strip()
                    entry["tags"][clean_tag] = entry["tags"].get(clean_tag, 0) + 1

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
        logger.info("--- Running Graph Normalization ---")
        updates_count = 0
        
        ORCHESTRA_TO_GENRE = {
            "golden_age": ["d'arienzo", "di sarli", "troilo", "pugliese", "canaro", "biagi", "tanturi", "d'agostino", "fresedo", "cals", "de angelis", "laurenz", "donato", "rodriguez", "demare", "lomuto"],
            "modern_traditional": ["color tango", "romantica milonguera", "misteriosa", "sexteto milonguero", "cachivache", "sans souci", "juan d'arienzo", "la juan d'arienzo", "andariega"],
            "fantasia": ["piazzolla", "sexteto mayor", "garello", "forever tango", "mariano mores"],
            "electronic_neo": ["gotan", "bajofondo", "tanghetto", "narcotango", "otros aires", "san telmo", "electrocutango", "soltango", "projecto"] 
        }

        for vid_id, data in self.videos.items():
            try:
                content = VideoContent(**data)
            except Exception as e:
                logger.warning(f"Skipping malformed video {vid_id}: {e}")
                continue
            
            changed = False

            if "dancers" in data and not data.get("performances"):
                changed = True
            
            if content.music and content.music.orchestra and not content.music.genre:
                orch_lower = content.music.orchestra.lower()
                for genre, keywords in ORCHESTRA_TO_GENRE.items():
                    if any(k in orch_lower for k in keywords):
                        content.music.genre = genre
                        changed = True
                        break

            if resolver:
                if content.event and content.event.name:
                    orig_evt = content.event.name
                    content.event.name = resolver.resolve_event(orig_evt)
                    if content.event.name != orig_evt:
                        changed = True

                if content.videographer:
                    orig_vid = content.videographer
                    content.videographer = resolver.resolve(orig_vid, "videographers")
                    if content.videographer != orig_vid:
                        changed = True

                for p in content.performances:
                    for d in p.dancers:
                        orig_d = d.name
                        d.name = resolver.resolve_dancer(orig_d)
                        if d.name != orig_d:
                            changed = True

                if "tags" in data:
                    original_tags = data["tags"]
                    new_tags = []
                    tags_changed = False
                    for t in original_tags:
                        resolved = resolver.resolve(t, "tags")
                        new_tags.append(resolved)
                        if resolved != t:
                            tags_changed = True
                    
                    if tags_changed:
                        data["tags"] = new_tags
                        changed = True

            for p in content.performances:
                if len(p.dancers) == 2:
                    d1, d2 = p.dancers[0], p.dancers[1]
                    new_n1, new_n2 = self.infer_partnership(d1.name, d2.name)
                    
                    if new_n1 != d1.name:
                        d1.name = new_n1
                    if new_n2 != d2.name:
                        d2.name = new_n2
            
            if changed:
                updates_count += 1
                dumped = content.model_dump()
                data.update(dumped)
                if "dancers" in data and "performances" in data:
                    data.pop("dancers", None)

        logger.info(f"Normalized {updates_count} videos.")

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
        def get_candidates(query_name):
            query_norm = query_name.lower().strip()
            candidates = []
            for known_name in self.dancers.keys():
                if query_norm == known_name.lower():
                    return [known_name]
                known_norm = known_name.lower()
                known_parts = known_norm.split()
                if query_norm in known_parts:
                    candidates.append(known_name)
            return candidates

        cands_a = get_candidates(name_a)
        cands_b = get_candidates(name_b)

        if not cands_a and not cands_b:
            return name_a, name_b

        if not cands_a: cands_a = [name_a]
        if not cands_b: cands_b = [name_b]

        best_score = 0
        best_pair = (name_a, name_b)

        for ca in cands_a:
            data_a = self.dancers.get(ca)
            if not data_a: continue
            
            partners_a = data_a.get("partners", {})
            
            for cb in cands_b:
                if cb in partners_a:
                    score = partners_a[cb]
                    if score > best_score:
                        best_score = score
                        best_pair = (ca, cb)
        
        if best_score > 0:
            return best_pair
            
        return name_a, name_b

    def get_dancer_profile(self, name: str) -> Optional[Dict]:
        if name not in self.dancers:
            return None
        return self.dancers[name]

    async def update_style_embeddings(self, llm_client=None):
        """
        Hybrid Analytics Pipeline:
        1. Sparse Structural Features (TF-IDF on Partners/Events)
        2. Dense Semantic Features (Weighted Average of Music/Tags Embeddings)
        3. Fusion & UMAP
        """
        all_dancers = list(self.dancers.keys())
        active_dancers = [d for d in all_dancers if len(self.dancers[d].get('videos', [])) > 1]

        if len(active_dancers) < 10:
            return
        
        try:
            logger.info(f"Starting Hybrid Analytics Update for {len(active_dancers)} dancers...")

            # --- PART A: Structural Features (Sparse) ---
            struct_features_set = set()
            for d in active_dancers:
                struct_features_set.update(self.dancers[d].get("partners", {}).keys())
                struct_features_set.update(self.dancers[d].get("events", {}).keys())
            
            struct_features_list = sorted(list(struct_features_set))
            struct_map = {name: i for i, name in enumerate(struct_features_list)}
            n_struct = len(struct_features_list)
            
            rows, cols, data = [], [], []
            for row_idx, dancer in enumerate(active_dancers):
                d_data = self.dancers[dancer]
                for p, count in d_data.get("partners", {}).items():
                    if p in struct_map:
                        rows.append(row_idx)
                        cols.append(struct_map[p])
                        data.append(count * 1.0)
                for e, count in d_data.get("events", {}).items():
                    if e in struct_map:
                        rows.append(row_idx)
                        cols.append(struct_map[e])
                        data.append(count * 1.5)
            
            sparse_struct = csr_matrix((data, (rows, cols)), shape=(len(active_dancers), n_struct))
            tfidf = TfidfTransformer()
            tfidf_struct = tfidf.fit_transform(sparse_struct)
            
            # --- PART B: Semantic Features (Dense) ---
            # Identify unique terms to embed
            all_orchestras = set()
            all_tags = set()
            for d in active_dancers:
                all_orchestras.update(self.dancers[d].get("orchestras", {}).keys())
                all_tags.update(self.dancers[d].get("tags", {}).keys())
            
            semantic_terms = all_orchestras.union(all_tags)
            
            # Check Cache
            terms_to_fetch = [t for t in semantic_terms if t not in self.feature_embeddings]
            
            if terms_to_fetch and llm_client:
                logger.info(f"Fetching embeddings for {len(terms_to_fetch)} new terms...")
                for term in terms_to_fetch:
                    if len(term) < 2: continue
                    try:
                        # Create a semantic label for better embedding (e.g. "Tango Orchestra: D'Arienzo")
                        # But simpler is often better for general tags. Let's stick to raw term.
                        emb = await llm_client.get_embedding(term)
                        if emb:
                            self.feature_embeddings[term] = emb
                    except Exception as e:
                        logger.warning(f"Failed to embed '{term}': {e}")
                self.save_feature_cache()
            
            embedding_dim = 768
            if self.feature_embeddings:
                embedding_dim = len(next(iter(self.feature_embeddings.values())))
            
            # Build Separate Dense Matrices
            music_vectors = []
            tag_vectors = []
            
            for dancer in active_dancers:
                d_data = self.dancers[dancer]
                
                # 1. Music Vector (High Quality)
                vec_m = np.zeros(embedding_dim)
                w_m = 0.0
                for o, count in d_data.get("orchestras", {}).items():
                    if o in self.feature_embeddings:
                        emb = np.array(self.feature_embeddings[o])
                        vec_m += emb * count
                        w_m += count
                if w_m > 0: vec_m /= w_m
                music_vectors.append(vec_m)
                
                # 2. Tag Vector (Noisy)
                vec_t = np.zeros(embedding_dim)
                w_t = 0.0
                for t, count in d_data.get("tags", {}).items():
                    if t in self.feature_embeddings:
                        emb = np.array(self.feature_embeddings[t])
                        vec_t += emb * count
                        w_t += count
                if w_t > 0: vec_t /= w_t
                tag_vectors.append(vec_t)

            mat_music = np.array(music_vectors)
            mat_tags = np.array(tag_vectors)
            
            # Normalize before weighting
            mat_music = normalize(mat_music, axis=1)
            mat_tags = normalize(mat_tags, axis=1)
            
            # --- PART C: Fusion ---
            struct_dense = tfidf_struct.toarray()
            
            # Weighted Stacking
            # Structure: 1.0 (Implicit via TF-IDF scaling)
            # Music: 3.0 (Strong Style Gravity)
            # Tags: 0.5 (Weak context)
            final_matrix = np.hstack([
                struct_dense, 
                mat_music * 3.0, 
                mat_tags * 0.5
            ])
            
            # --- Similarity & UMAP ---
            sim_matrix = cosine_similarity(final_matrix)
            
            for i, dancer_name in enumerate(active_dancers):
                row_sims = sim_matrix[i]
                k = 6
                if len(row_sims) < k: k = len(row_sims)
                top_k_idx = np.argpartition(row_sims, -k)[-k:]
                scores = []
                for idx in top_k_idx:
                    if idx == i: continue
                    score = row_sims[idx]
                    if score > 0.05:
                        scores.append({"name": active_dancers[idx], "score": float(round(score, 3))})
                scores.sort(key=lambda x: x["score"], reverse=True)
                self.dancers[dancer_name]["similar_dancers"] = scores[:5]

            reducer = umap.UMAP(
                n_neighbors=30, 
                n_components=2, 
                min_dist=0.5, 
                spread=5.0,
                metric='cosine',
                init='random',
                random_state=42
            )
            coords = reducer.fit_transform(final_matrix)
            
            hdb = HDBSCAN(min_cluster_size=6, min_samples=2)
            cluster_labels = hdb.fit_predict(coords)
            cluster_labels_str = [str(c) for c in cluster_labels]
            
            unique_clusters = sorted(list(set(cluster_labels_str)))
            if '-1' in unique_clusters: unique_clusters.remove('-1')
            
            cluster_centroids = []
            for c in unique_clusters:
                indices = [i for i, label in enumerate(cluster_labels_str) if label == c]
                if indices:
                    avg_x = np.mean(coords[indices, 0])
                    cluster_centroids.append((c, avg_x))
            cluster_centroids.sort(key=lambda x: x[1])
            sorted_labels = [x[0] for x in cluster_centroids]
            
            palette = {}
            golden_ratio_conjugate = 0.618033988749895
            for i, label in enumerate(sorted_labels):
                h = (0.0 + i * golden_ratio_conjugate) % 1.0
                l, s = (0.45, 0.85) if i % 2 == 0 else (0.65, 0.75)
                rgb = colorsys.hls_to_rgb(h, l, s)
                palette[label] = matplotlib.colors.to_hex(rgb)
            palette['-1'] = '#7f7f7f'
            
            for idx, dancer in enumerate(active_dancers):
                lbl = cluster_labels_str[idx]
                color = palette.get(lbl, '#000000')
                self.dancers[dancer]["style_embedding"] = {
                    "x": float(coords[idx][0]),
                    "y": float(coords[idx][1]),
                    "cluster": lbl,
                    "color": color
                }

            # Stats
            all_events_flat = []
            all_videographers_flat = []
            all_orchestras_flat = []
            
            for v in self.videos.values():
                if v.get("event") and v["event"].get("name"):
                    all_events_flat.append(v["event"]["name"])
                if v.get("videographer"):
                    all_videographers_flat.append(v["videographer"])
                if v.get("music") and v["music"].get("orchestra"):
                    all_orchestras_flat.append(v["music"]["orchestra"])
            
            self.stats = {
                "top_events": all_events_flat, 
                "top_videographers": all_videographers_flat,
                "top_orchestras": all_orchestras_flat,
                "total_unique_orchestras": len(all_orchestras),
                "total_unique_tags": len(all_tags)
            }
            
            self.save()
            logger.info(f"Updated Hybrid Analytics for {len(active_dancers)} dancers.")
        except Exception as e:
            logger.error(f"Failed to update analytics: {e}")

    def merge_dancers(self, source_name: str, target_name: str):
        if source_name not in self.dancers or source_name == target_name:
            return
        if target_name not in self.dancers:
            self.dancers[target_name] = {"videos": [], "partners": {}, "events": {}, "tags": {}, "roles": set(), "similar_dancers": []}
        source = self.dancers[source_name]
        target = self.dancers[target_name]
        target["videos"] = list(set(target["videos"]) | set(source["videos"]))
        target["roles"] = target.get("roles", set()) | source.get("roles", set())
        for dict_name in ["partners", "events", "tags"]:
            target_dict = target.get(dict_name, {})
            source_dict = source.get(dict_name, {})
            for k, v in source_dict.items():
                if dict_name == "partners" and k == target_name: continue
                target_dict[k] = target_dict.get(k, 0) + v
            target[dict_name] = target_dict
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
        dancers_serializable = {}
        for k, v in self.dancers.items():
            dancers_serializable[k] = {
                "videos": v["videos"],
                "partners": v["partners"],
                "events": v.get("events", {}),
                "orchestras": v.get("orchestras", {}),
                "music_genres": v.get("music_genres", {}),
                "video_types": v.get("video_types", {}),
                "tags": v.get("tags", {}),
                "roles": list(v.get("roles", [])),
                "style_embedding": v.get("style_embedding"),
                "similar_dancers": v.get("similar_dancers", [])
            }
        data = {"videos": self.videos, "dancers": dancers_serializable, "stats": self.stats}
        with open(self.db_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def save_feature_cache(self):
        with open(self.feature_cache_file, "w", encoding="utf-8") as f:
            json.dump(self.feature_embeddings, f)

    def load(self):
        if os.path.exists(self.db_file):
            try:
                with open(self.db_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.videos = data.get("videos", {})
                    self.dancers = data.get("dancers", {})
                    self.stats = data.get("stats", {})
                    for d in self.dancers.values():
                        d["roles"] = set(d.get("roles", []))
                logger.info(f"Loaded Graph: {len(self.videos)} videos, {len(self.dancers)} dancers.")
            except Exception as e:
                logger.error(f"Failed to load graph DB: {e}")
        
        if os.path.exists(self.feature_cache_file):
            try:
                with open(self.feature_cache_file, "r", encoding="utf-8") as f:
                    self.feature_embeddings = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load feature cache: {e}")