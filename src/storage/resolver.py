import json
import os
import unicodedata
import asyncio
from typing import Dict, List, Set
from difflib import SequenceMatcher
from loguru import logger
import gin
from pydantic import BaseModel
from src.core.llm_client import Rhizosphere

class ResolutionItem(BaseModel):
    original: str
    canonical: str

class ResolutionList(BaseModel):
    resolutions: List[ResolutionItem]

@gin.configurable
class EntityResolver:
    """
    Handles entity deduplication and canonicalization via String Clustering + LLM Verification.
    Supports multiple entity types (dancers, events, videographers).
    """
    def __init__(self, alias_file: str = "aliases.json", model_id: str = "gpt-4o"):
        self.alias_file = alias_file
        self.model_id = model_id
        # map entity_type -> { normalized_alias -> canonical_name }
        self.aliases: Dict[str, Dict[str, str]] = {
            "dancers": {},
            "events": {},
            "videographers": {}
        }
        self.load()

    def normalize_string(self, s: str) -> str:
        """
        Aggressive normalization for comparison: 
        lowercase, remove accents, remove extra spaces.
        """
        if not s: 
            return ""
        s = s.lower().strip()
        # Remove accents (e.g., 'García' -> 'Garcia')
        s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
        return s

    def resolve(self, name: str, entity_type: str) -> str:
        """
        Generic resolution method.
        """
        norm_name = self.normalize_string(name)
        if entity_type in self.aliases and norm_name in self.aliases[entity_type]:
            return self.aliases[entity_type][norm_name]
        return name.strip()

    def resolve_dancer(self, name: str) -> str:
        return self.resolve(name, "dancers")
        
    def resolve_event(self, name: str) -> str:
        return self.resolve(name, "events")

    def add_alias(self, alias: str, canonical: str, entity_type: str):
        if entity_type not in self.aliases:
            self.aliases[entity_type] = {}
        self.aliases[entity_type][self.normalize_string(alias)] = canonical.strip()
        self.save()

    async def run_deduplication(self, all_names: List[str], entity_type: str, llm_client: Rhizosphere, graph_store=None):
        """
        Main maintenance routine. 
        1. Clusters names by string similarity.
        2. Sends clusters to LLM for resolution.
        3. Updates alias DB.
        4. Merges nodes in GraphStore if provided (only for dancers currently).
        """
        logger.info(f"Starting deduplication for {len(all_names)} {entity_type} names...")
        
        # 1. Filter out names already resolved/aliased to save tokens
        if entity_type not in self.aliases:
            self.aliases[entity_type] = {}
            
        unknown_names = [n for n in all_names if self.normalize_string(n) not in self.aliases[entity_type]]
        if not unknown_names:
            logger.info(f"No new {entity_type} names to resolve.")
            return

        # 2. Cluster similar names
        clusters = self._cluster_names(unknown_names)
        logger.info(f"Found {len(clusters)} clusters to process.")

        # Load Prompt
        try:
            with open("src/prompts/resolve.txt", "r") as f:
                raw_prompt = f.read()
                # Inject entity type into prompt
                system_prompt = raw_prompt.replace("{entity_type}", entity_type)
        except FileNotFoundError:
            logger.error("Resolve prompt not found.")
            return

        # 3. Process Clusters
        for cluster in clusters:
            if len(cluster) < 2:
                continue
                
            try:
                # We ask the LLM to return a ResolutionList
                response = await llm_client.structured_call(
                    chat_history=[{"role": "user", "content": f"Input: {json.dumps(cluster)}"}],
                    system_prompt=system_prompt,
                    model_id=self.model_id,
                    model_kwargs={"temperature": 0.0},
                    pydantic_obj=ResolutionList
                )
                
                # 4. Apply Updates
                for item in response.resolutions:
                    variant = item.original
                    canonical = item.canonical
                    
                    if variant != canonical:
                        self.add_alias(variant, canonical, entity_type)
                        logger.info(f"Resolved {entity_type}: '{variant}' -> '{canonical}'")
                        
                        # 5. Retroactive Merge (Specific logic per type if needed)
                        if graph_store and entity_type == "dancers":
                            graph_store.merge_dancers(variant, canonical)
                
                # Rate limit protection for batch processing
                await asyncio.sleep(0.5)
                        
            except Exception as e:
                logger.error(f"Failed to resolve cluster {cluster}: {e}")

    def _cluster_names(self, names: List[str], threshold: float = 0.8) -> List[List[str]]:
        """
        Groups a list of strings into clusters based on similarity.
        Greedy approach.
        """
        clusters = []
        pool = set(names)
        
        while pool:
            seed = pool.pop()
            current_cluster = [seed]
            
            # Find close matches in the remaining pool
            to_remove = []
            for candidate in pool:
                ratio = SequenceMatcher(None, self.normalize_string(seed), self.normalize_string(candidate)).ratio()
                if ratio > threshold:
                    current_cluster.append(candidate)
                    to_remove.append(candidate)
            
            # Remove matches from pool so they don't get clustered again
            for r in to_remove:
                pool.remove(r)
                
            clusters.append(current_cluster)
            
        return clusters

    def save(self):
        with open(self.alias_file, "w") as f:
            json.dump(self.aliases, f, indent=2)

    def load(self):
        if os.path.exists(self.alias_file):
            try:
                with open(self.alias_file, "r") as f:
                    data = json.load(f)
                    # Migration support: if old format (flat dict), convert to 'dancers'
                    if data and isinstance(list(data.values())[0], str):
                        self.aliases["dancers"] = data
                    else:
                        self.aliases = data
            except Exception as e:
                logger.error(f"Failed to load aliases: {e}")