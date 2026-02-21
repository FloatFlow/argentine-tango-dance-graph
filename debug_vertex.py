import asyncio
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load env variables from .env file
load_dotenv()

async def debug_main():
    project_id = os.environ.get("VERTEX_PROJECT_ID")
    location = os.environ.get("VERTEX_LOCATION", "us-central1")
    creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    api_key = os.environ.get("GOOGLE_API_KEY")

    print(f"--- Debugging Vertex/Gemini Connection ---")
    print(f"Project ID: {project_id}")
    print(f"Location: {location}")
    print(f"Credentials File: {creds}")
    print(f"API Key present: {bool(api_key)}")

    try:
        if project_id:
            print("Initializing in Vertex AI Mode...")
            # This mimics exactly how src/core/llm_client.py initializes
            client = genai.Client(
                vertexai=True,
                project=project_id,
                location=location
            ).aio
        else:
            print("Initializing in AI Studio Mode...")
            client = genai.Client(
                api_key=api_key
            ).aio

        model_id = "gemini-3-flash-preview"
        print(f"Sending request to '{model_id}'...")
        
        response = await client.models.generate_content(
            model=model_id,
            contents='Hello, please verify that you are operational. Reply with a short tango pun.',
        )
        
        print("\nSUCCESS!")
        print(f"Response: {response.text}")

    except Exception as e:
        print("\nFAILURE")
        print(f"Error Type: {type(e)}")
        print(f"Error Message: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_main())