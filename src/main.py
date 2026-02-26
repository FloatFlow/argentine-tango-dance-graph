import asyncio
import os
import random
from loguru import logger
from dotenv import load_dotenv
import gin

from src.collection.tube_client import TubeClient
from src.collection.queue_manager import QueueManager
from src.core.llm_client import Rhizosphere
from src.core.schemas import VideoContent
from src.storage.graph_store import GraphStore
from src.storage.resolver import EntityResolver

# Load env vars for API keys
load_dotenv()

@gin.configurable
async def process_video(
    video_data: dict, 
    tube_client: TubeClient, 
    llm_client: Rhizosphere, 
    graph_store: GraphStore, 
    queue_manager: QueueManager,
    resolver: EntityResolver,
    extract_prompt: str,
    model_id: str = "gpt-4o" # Configurable via gin
) -> bool:
    """
    Returns True if video was successfully processed and contained relevant content.
    Returns False if skipped (seen/amateur) or failed extraction.
    """
    video_id = video_data['id']
    title = video_data.get('title', '')

    # Filter out strict Amateur categories to focus on Professional/Pro-Am content
    # We allow "Pro-Am" but skip standalone "Amateur"
    if "amateur" in title.lower() and "pro-am" not in title.lower():
        logger.info(f"Skipping Amateur video: {title}")
        # Mark as seen so we don't re-process it in future searches
        queue_manager.mark_video_seen(video_id)
        return False
    
    # Filter out non-Tango dance styles to prevent graph drift
    # This keeps the crawler focused if a channel uploads mixed content
    forbidden = ["salsa", "bachata", "kizomba", "zouk", "west coast swing"]
    if any(style in title.lower() for style in forbidden):
        logger.info(f"Skipping non-Tango video: {title}")
        queue_manager.mark_video_seen(video_id)
        return False
    
    # Check if already processed
    if queue_manager.is_video_seen(video_id):
        return False

    logger.info(f"Processing Video: {title} ({video_id})")

    # 1. Fetch full details (description, tags) - Wrapped in thread to avoid blocking loop
    details = await asyncio.to_thread(tube_client.fetch_video_details, video_id)
    if not details:
        logger.warning(f"Could not fetch details for {video_id}")
        return False

    # 1.5 Deep Filter: Check Description and Tags for non-Tango content
    # This catches videos where the title is generic (e.g. "Social Dance") but metadata reveals it's Salsa.
    description = details.get('description', '').lower()
    tags = [t.lower() for t in details.get('tags', [])]
    combined_text = description + " " + " ".join(tags)
    
    if any(style in combined_text for style in forbidden):
        logger.info(f"Skipping non-Tango video (Deep Filter): {title}")
        queue_manager.mark_video_seen(video_id)
        return False

    # 2. Prepare context for LLM
    # We construct a string representation of the metadata
    metadata_text = f"""
    Title: {details.get('title')}
    Channel: {details.get('uploader')}
    Upload Date: {details.get('upload_date')}
    Description:
    {details.get('description', '')[:2000]} 
    
    Tags: {', '.join(details.get('tags', [])[:20])}
    """
    # Truncated description to save tokens/avoid noise

    try:
        # 3. Extract Intelligence
        # We use a dummy chat history with just the user message
        chat_history = [{"role": "user", "content": metadata_text}]
        
        content: VideoContent = await llm_client.structured_call(
            chat_history=chat_history,
            system_prompt=extract_prompt,
            model_id=model_id, 
            model_kwargs={"temperature": 0.0},
            pydantic_obj=VideoContent
        )

        # 3.5 Canonicalize Entities
        # Resolve dancer names against known aliases
        for performance in content.performances:
            for dancer in performance.dancers:
                original_name = dancer.name
                dancer.name = resolver.resolve_dancer(original_name)
                if dancer.name != original_name:
                    logger.debug(f"Normalized: '{original_name}' -> '{dancer.name}'")
        
        # Resolve Event
        if content.event and content.event.name:
            content.event.name = resolver.resolve_event(content.event.name)
            
        # Resolve Videographer
        if content.videographer:
            content.videographer = resolver.resolve(content.videographer, "videographers")

        logger.success(f"Extracted: {[d.name for d in content.dancers]} @ {content.event.name if content.event else 'Unknown Event'}")

        # 4. Store in Graph
        duration = details.get('duration', 0)
        thumbnail = details.get('thumbnail', '')
        graph_store.add_video(video_id, content, title, duration, thumbnail)

        # 5. Feed the Flywheel
        # Only add new queries if we successfully extracted meaningful data
        if content.performances:
            # A. Exploitation (High Priority): Target newly discovered entities
            
            # 1. Dancers
            for p in content.performances:
                for dancer in p.dancers:
                    profile = graph_store.get_dancer_profile(dancer.name)
                    # Since we just added the video in Step 4, a count of 1 means they are new to the graph
                    if profile and len(profile.get("videos", [])) <= 1:
                        logger.info(f"Exploiting new dancer: {dancer.name}")
                        await queue_manager.add_query(f"{dancer.name} tango performance", priority=5, llm_client=llm_client)
            
            # 2. Videographers
            if content.videographer:
                # QueueManager deduplicates, so this only runs once per unique videographer found
                await queue_manager.add_query(f"{content.videographer} tango", priority=5, llm_client=llm_client)

            # 3. Events (Deterministic Expansion)
            if content.event and content.event.name:
                # Base event search (High Priority)
                await queue_manager.add_query(f"{content.event.name} tango", priority=5, llm_client=llm_client)
                # Recent history expansion (Lower Priority - ensure freshness)
                for year in range(2016, 2027):
                    query_str = f"{content.event.name} {year}"
                    if await queue_manager.add_query(query_str, priority=10, llm_client=llm_client):
                        logger.info(f"Queueing event expansion: '{query_str}'")

        # 6. Mark done
        queue_manager.mark_video_seen(video_id)
        # Note: save_state() moved to outer loop to prevent concurrent write collisions
        
        # Return True if we found actual performances, False if extraction was empty (irrelevant)
        return bool(content.performances)

    except Exception as e:
        logger.error(f"LLM Extraction failed for {video_id}: {e}")
        return False

async def run_maintenance(graph_store: GraphStore, resolver: EntityResolver, llm_client: Rhizosphere, queue_manager: QueueManager):
    """
    Periodically runs deduplication on all entities found in the graph.
    """
    logger.info("--- Running Graph Maintenance ---")
    
    # 0. Graph Normalization (Schema Migration & Contextual Disambiguation)
    # This runs before deduplication to ensure we have the best possible names from context
    graph_store.normalize_graph()
    
    # 1. Deduplicate Dancers (Includes graph merging)
    dancer_names = list(graph_store.dancers.keys())
    if dancer_names:
        await resolver.run_deduplication(dancer_names, "dancers", llm_client, graph_store)

    # 2. Deduplicate Events & Videographers
    # We gather these from the stored video blobs
    event_names = set()
    video_names = set()
    
    for vid_data in graph_store.videos.values():
        if vid_data.get('event') and vid_data['event'].get('name'):
            event_names.add(vid_data['event']['name'])
        if vid_data.get('videographer'):
            video_names.add(vid_data['videographer'])
            
    if event_names:
        await resolver.run_deduplication(list(event_names), "events", llm_client)
    if video_names:
        await resolver.run_deduplication(list(video_names), "videographers", llm_client)

    # 3. Update Style Embeddings (t-SNE)
    # This runs locally and saves coordinates to the JSON for the frontend
    graph_store.update_style_embeddings()
    
    # 4. Population Estimate (Chao1)
    stats = queue_manager.get_population_estimate()
    logger.success(f"--- Population Status ---")
    logger.info(f"Seen Videos: {stats['observed']} | Estimated Total: {stats['estimated_total']}")
    logger.info(f"Market Coverage: {stats['coverage_percent']}% (Singletons: {stats['singletons']}, Doubletons: {stats['doubletons']})")
        
    logger.info("--- Maintenance Complete ---")

@gin.configurable
async def main(video_delay: int = 5, query_delay: int = 10, search_limit: int = 50, maintenance_interval: int = 10, chunk_size: int = 5):
    # DEBUG: Verify Environment / Auth State
    project_id = os.environ.get("VERTEX_PROJECT_ID")
    api_key = os.environ.get("GOOGLE_API_KEY")
    creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    
    logger.info(f"--- Runtime Auth Debug ---")
    logger.info(f"VERTEX_PROJECT_ID: {project_id}")
    logger.info(f"GOOGLE_API_KEY: {'[PRESENT]' if api_key else '[MISSING]'}")
    logger.info(f"GOOGLE_APPLICATION_CREDENTIALS: {creds}")
    
    # Initialize components
    tube_client = TubeClient()
    queue_manager = QueueManager()
    graph_store = GraphStore()
    llm_client = Rhizosphere()
    resolver = EntityResolver()

    # Load System Prompt
    try:
        with open("src/prompts/extract.txt", "r") as f:
            extract_prompt = f.read()
    except FileNotFoundError:
        logger.error("Prompts file not found. Please ensure src/prompts/extract.txt exists.")
        return

    # Seed if empty
    if not queue_manager.queue:
        logger.info("Queue is empty. Seeding with defaults.")
        seeds = [
            "Mundial de Tango 2024",
            "Torino Tango Festival performance",
            "CITA tango festival 2023",
            "Tango Salon champions",
            "Carlitos Espinoza tango"
        ]
        for s in seeds:
            await queue_manager.add_query(s, priority=10, llm_client=llm_client)

    # Startup Backfill: Ensure we have exploited all known dancers
    # This runs every startup but queue_manager deduplicates against 'seen_queries',
    # so it only adds searches for dancers we haven't targeted yet.
    logger.info(f"Checking exploitation coverage for {len(graph_store.dancers)} dancers...")
    for name in graph_store.dancers.keys():
        # Priority 5 (High) ensures we prioritize filling out the graph over random exploration
        await queue_manager.add_query(f"{name} tango performance", priority=5, llm_client=llm_client)

    # Main Loop
    query_counter = 0
    while True:
        # Run maintenance every N queries to keep the graph clean as it grows
        if query_counter > 0 and query_counter % maintenance_interval == 0:
            await run_maintenance(graph_store, resolver, llm_client, queue_manager)

        query = queue_manager.pop_query()
        if not query:
            logger.info("Queue empty. Exiting.")
            break
            
        logger.info(f"--- Running Search Query: '{query}' ---")
        
        # Search (offloaded to thread so we don't block)
        results = await asyncio.to_thread(tube_client.search, query, limit=search_limit)
        
        seen_streak = 0
        irrelevant_streak = 0
        
        # Process videos in chunks
        for i in range(0, len(results), chunk_size):
            chunk = results[i:i+chunk_size]
            tasks = []
            
            for video_summary in chunk:
                video_id = video_summary['id']
                
                # 1. Dynamic Depth: Check if we are retreading old ground
                if queue_manager.is_video_seen(video_id):
                    # Record the collision to correct Chao1 stats
                    queue_manager.record_sighting(video_id)
                    seen_streak += 1
                    continue
                
                seen_streak = 0 # Reset streak if we find a new video
                
                # 2. Add to concurrent processing pool
                tasks.append(process_video(
                    video_summary, 
                    tube_client, 
                    llm_client, 
                    graph_store, 
                    queue_manager,
                    resolver,
                    extract_prompt
                ))

            # Check streaks before running batch
            if seen_streak >= 10:
                logger.info(f"Stopping query '{query}' early due to {seen_streak} consecutive seen videos.")
                break
                
            if not tasks:
                continue
                
            # Run chunk concurrently (Both yt-dlp fetches and LLM calls)
            # We recreate the list of tasks every chunk iteration to ensure fresh coroutines
            chunk_results = await asyncio.gather(*tasks)
            
            # Save state once per chunk to avoid race conditions
            queue_manager.save_state()
            
            # 3. Dynamic Depth: Check if results are drifting into irrelevance
            # chunk_results contains the bool return values from process_video
            for res in chunk_results:
                if res:
                    irrelevant_streak = 0
                else:
                    irrelevant_streak += 1
                
            if irrelevant_streak >= 10:
                logger.info(f"Stopping query '{query}' early due to {irrelevant_streak} consecutive irrelevant/failed videos.")
                break

            # Rate limiting sleep between chunks
            await asyncio.sleep(video_delay)

        # Mark query as done so it doesn't get re-queued on restart
        queue_manager.complete_query()
        queue_manager.save_state()
        query_counter += 1
        
        # Sleep between queries
        await asyncio.sleep(query_delay)
if __name__ == "__main__":
    gin.parse_config_file("src/config.gin")
    asyncio.run(main())