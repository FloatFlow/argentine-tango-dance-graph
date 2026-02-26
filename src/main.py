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
    graph_lock: asyncio.Lock,
    queue_lock: asyncio.Lock,
    model_id: str = "gpt-4o" # Configurable via gin
) -> bool:
    """
    Returns True if video was successfully processed and contained relevant content.
    Returns False if skipped (seen/amateur) or failed extraction.
    """
    video_id = video_data['id']
    title = video_data.get('title', '')

    # Filter out strict Amateur categories to focus on Professional/Pro-Am content
    if "amateur" in title.lower() and "pro-am" not in title.lower():
        async with queue_lock:
            queue_manager.mark_video_seen(video_id)
        return False
    
    # Filter out non-Tango dance styles
    forbidden = ["salsa", "bachata", "kizomba", "zouk", "west coast swing"]
    if any(style in title.lower() for style in forbidden):
        async with queue_lock:
            queue_manager.mark_video_seen(video_id)
        return False
    
    # Check if already processed (Thread-safe check)
    # Note: Main loop also checks this, but we check again in case of race during batching
    # Using lock for read is safer given concurrent writes
    async with queue_lock:
        if queue_manager.is_video_seen(video_id):
            return False

    logger.info(f"Processing Video: {title} ({video_id})")

    # 1. Fetch full details (I/O Bound - runs in thread)
    details = await asyncio.to_thread(tube_client.fetch_video_details, video_id)
    if not details:
        logger.warning(f"Could not fetch details for {video_id}")
        return False

    # 1.5 Deep Filter (Description/Tags)
    description = details.get('description', '').lower()
    tags = [t.lower() for t in details.get('tags', [])]
    combined_text = description + " " + " ".join(tags)
    
    if any(style in combined_text for style in forbidden):
        logger.info(f"Skipping non-Tango video (Deep Filter): {title}")
        async with queue_lock:
            queue_manager.mark_video_seen(video_id)
        return False

    # 2. Prepare context for LLM
    metadata_text = f"""
    Title: {details.get('title')}
    Channel: {details.get('uploader')}
    Upload Date: {details.get('upload_date')}
    Description:
    {details.get('description', '')[:2000]} 
    
    Tags: {', '.join(details.get('tags', [])[:20])}
    """

    try:
        # 3. Extract Intelligence (I/O Bound)
        chat_history = [{"role": "user", "content": metadata_text}]
        
        content: VideoContent = await llm_client.structured_call(
            chat_history=chat_history,
            system_prompt=extract_prompt,
            model_id=model_id, 
            model_kwargs={"temperature": 0.0},
            pydantic_obj=VideoContent
        )

        # 3.5 Canonicalize Entities (Read-Only on Resolver mostly, but safe to keep concurrent)
        for performance in content.performances:
            for dancer in performance.dancers:
                original_name = dancer.name
                dancer.name = resolver.resolve_dancer(original_name)
                if dancer.name != original_name:
                    logger.debug(f"Normalized: '{original_name}' -> '{dancer.name}'")
        
        if content.event and content.event.name:
            content.event.name = resolver.resolve_event(content.event.name)
            
        if content.videographer:
            content.videographer = resolver.resolve(content.videographer, "videographers")

        logger.success(f"Extracted: {[d.name for d in content.dancers]} @ {content.event.name if content.event else 'Unknown Event'}")

        # 4. Store in Graph (CRITICAL SECTION - WRITE)
        duration = details.get('duration', 0)
        thumbnail = details.get('thumbnail', '')
        
        async with graph_lock:
            graph_store.add_video(video_id, content, title, duration, thumbnail)

        # 5. Feed the Flywheel (CRITICAL SECTION - WRITE)
        if content.performances:
            async with queue_lock:
                # A. Exploitation
                for p in content.performances:
                    for dancer in p.dancers:
                        # We need to access graph_store here too, but get_dancer_profile is a read
                        # Since we just wrote to it, it should be fresh.
                        profile = graph_store.get_dancer_profile(dancer.name)
                        if profile and len(profile.get("videos", [])) <= 1:
                            logger.info(f"Exploiting new dancer: {dancer.name}")
                            await queue_manager.add_query(f"{dancer.name} tango performance", priority=5, llm_client=llm_client)
                
                # B. Videographers
                if content.videographer:
                    await queue_manager.add_query(f"{content.videographer} tango", priority=5, llm_client=llm_client)

                # C. Events
                if content.event and content.event.name:
                    await queue_manager.add_query(f"{content.event.name} tango", priority=5, llm_client=llm_client)
                    for year in range(2016, 2027):
                        query_str = f"{content.event.name} {year}"
                        if await queue_manager.add_query(query_str, priority=10, llm_client=llm_client):
                            logger.info(f"Queueing event expansion: '{query_str}'")

        # 6. Mark done (CRITICAL SECTION - WRITE)
        async with queue_lock:
            queue_manager.mark_video_seen(video_id)
        
        return bool(content.performances)

    except Exception as e:
        logger.error(f"LLM Extraction failed for {video_id}: {e}")
        return False

async def run_maintenance(graph_store: GraphStore, resolver: EntityResolver, llm_client: Rhizosphere, queue_manager: QueueManager):
    logger.info("--- Running Graph Maintenance ---")
    graph_store.normalize_graph()
    
    dancer_names = list(graph_store.dancers.keys())
    if dancer_names:
        await resolver.run_deduplication(dancer_names, "dancers", llm_client, graph_store)

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

    graph_store.update_style_embeddings()
    
    stats = queue_manager.get_population_estimate()
    logger.success(f"--- Population Status ---")
    logger.info(f"Seen Videos: {stats['observed']} | Estimated Total: {stats['estimated_total']}")
    logger.info(f"Market Coverage: {stats['coverage_percent']}% (Singletons: {stats['singletons']}, Doubletons: {stats['doubletons']})")
        
    logger.info("--- Maintenance Complete ---")

@gin.configurable
async def main(video_delay: int = 5, query_delay: int = 10, search_limit: int = 50, maintenance_interval: int = 10, chunk_size: int = 5):
    project_id = os.environ.get("VERTEX_PROJECT_ID")
    api_key = os.environ.get("GOOGLE_API_KEY")
    creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    
    logger.info(f"--- Runtime Auth Debug ---")
    logger.info(f"VERTEX_PROJECT_ID: {project_id}")
    logger.info(f"GOOGLE_API_KEY: {'[PRESENT]' if api_key else '[MISSING]'}")
    logger.info(f"GOOGLE_APPLICATION_CREDENTIALS: {creds}")
    
    tube_client = TubeClient()
    queue_manager = QueueManager()
    graph_store = GraphStore()
    llm_client = Rhizosphere()
    resolver = EntityResolver()

    # Locks for Thread Safety
    graph_lock = asyncio.Lock()
    queue_lock = asyncio.Lock()

    try:
        with open("src/prompts/extract.txt", "r") as f:
            extract_prompt = f.read()
    except FileNotFoundError:
        logger.error("Prompts file not found.")
        return

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

    logger.info(f"Checking exploitation coverage for {len(graph_store.dancers)} dancers...")
    for name in graph_store.dancers.keys():
        await queue_manager.add_query(f"{name} tango performance", priority=5, llm_client=llm_client)

    query_counter = 0
    while True:
        if query_counter > 0 and query_counter % maintenance_interval == 0:
            await run_maintenance(graph_store, resolver, llm_client, queue_manager)

        async with queue_lock:
            query = queue_manager.pop_query()
        
        if not query:
            logger.info("Queue empty. Exiting.")
            break
            
        logger.info(f"--- Running Search Query: '{query}' ---")
        
        results = await asyncio.to_thread(tube_client.search, query, limit=search_limit)
        
        seen_streak = 0
        irrelevant_streak = 0
        
        for i in range(0, len(results), chunk_size):
            chunk = results[i:i+chunk_size]
            tasks = []
            
            for video_summary in chunk:
                video_id = video_summary['id']
                
                # Read Lock for safety
                async with queue_lock:
                    already_seen = queue_manager.is_video_seen(video_id)
                    if already_seen:
                        queue_manager.record_sighting(video_id)
                
                if already_seen:
                    seen_streak += 1
                    continue
                
                seen_streak = 0
                
                # Use create_task to properly schedule the coroutine
                task = asyncio.create_task(process_video(
                    video_summary, 
                    tube_client, 
                    llm_client, 
                    graph_store, 
                    queue_manager, 
                    resolver, 
                    extract_prompt,
                    graph_lock,
                    queue_lock
                ))
                tasks.append(task)

            if seen_streak >= 10:
                logger.info(f"Stopping query '{query}' early due to {seen_streak} consecutive seen videos.")
                # Cancel pending tasks if any (though loop order prevents this mostly)
                for t in tasks: t.cancel()
                break
                
            if not tasks:
                continue
                
            chunk_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Save state safely
            async with queue_lock:
                queue_manager.save_state()
            
            # Analyze results (handle exceptions)
            for res in chunk_results:
                if isinstance(res, bool) and res:
                    irrelevant_streak = 0
                else:
                    irrelevant_streak += 1
                
            if irrelevant_streak >= 10:
                logger.info(f"Stopping query '{query}' early due to {irrelevant_streak} consecutive irrelevant/failed videos.")
                break

            await asyncio.sleep(video_delay)

        async with queue_lock:
            queue_manager.complete_query()
            queue_manager.save_state()
            
        query_counter += 1
        await asyncio.sleep(query_delay)
if __name__ == "__main__":
    gin.parse_config_file("src/config.gin")
    asyncio.run(main())