import asyncio
import gin
import sys
import os

# Ensure the project root is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the main module as a library so Gin can verify the namespace match
import src.main

if __name__ == "__main__":
    # Parse the config
    gin.parse_config_file("src/config.gin")
    
    # Run the main function from the imported module, NOT as __main__
    # This ensures 'src.main.process_video' config applies correctly
    asyncio.run(src.main.main())