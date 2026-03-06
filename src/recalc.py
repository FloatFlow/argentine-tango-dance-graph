import asyncio
import os
import sys

# Add project root to path to allow imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from loguru import logger
from dotenv import load_dotenv
import gin

from src.storage.graph_store import GraphStore
from src.core.llm_client import Rhizosphere

# Load env vars for API keys
load_dotenv()

async def recalc_embeddings():
    logger.info("Initializing GraphStore and LLM Client...")
    
    # Initialize components
    # We don't need the full pipeline (Queue, TubeClient), just storage and intelligence
    graph_store = GraphStore()
    llm_client = Rhizosphere()
    
    # Run the update
    logger.info("Triggering Style Embedding Update (Late Fusion)...")
    await graph_store.update_style_embeddings(llm_client=llm_client)
    
    logger.success("Recalculation Complete. You can now run the Streamlit app.")

if __name__ == "__main__":
    # Ensure config is loaded if needed, though defaults might suffice
    if os.path.exists("src/config.gin"):
        gin.parse_config_file("src/config.gin")
        
    asyncio.run(recalc_embeddings())