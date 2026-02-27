import json
import os
import asyncio
import re
import base64
from typing import AsyncGenerator, List, Tuple, Dict, Any, Literal, Optional
import traceback
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_fixed, wait_random_exponential, RetryError

import gin
#from aiobotocore.session import get_session
from openai import AsyncOpenAI
from google import genai
from google.genai import types


def convert_history_to_gemini(history: list[dict[str, Any]]) -> list[types.Content]:
    google_history = []
    for msg in history:
        content_text = msg["content"]
        thought_sigs = msg.get("thought_signatures", [])

        # Fallback: Extract hidden thought signature if present in text
        if not thought_sigs:
            sig_match = re.search(r'<thought_signature>(.*?)</thought_signature>', content_text, re.DOTALL)
            if sig_match:
                thought_sigs = [sig_match.group(1)]
                # Remove the tag from the text so the model doesn't see it as user text
                content_text = content_text.replace(sig_match.group(0), "").strip()

        part = types.Part.from_text(text=content_text)

        # Re-attach the signature to the Part object
        if thought_sigs:
            # Try to set it if the SDK supports it (Gemini 3+)
            # We use the last signature as that's the one associated with the final generation
            # stored_sig is Base64 encoded from structured_call_stream to ensure JSON safety.
            stored_sig = thought_sigs[-1]

            try:
                # The SDK's Part object for thought_signature (TYPE_BYTES) requires bytes.
                # stored_sig is a Base64 string from our storage.
                try:
                    part.thought_signature = base64.b64decode(stored_sig, validate=True)
                except Exception:
                    # Fallback for dummy signatures or strings that aren't valid Base64
                    part.thought_signature = stored_sig.encode("utf-8")
            except Exception:
                # If the attribute doesn't exist or assignment fails, safely ignore
                pass

        if msg["role"] == "user":
            google_history.append(
                types.Content(role="user", parts=[part])
            )
        elif msg["role"] == "assistant":
            google_history.append(
                types.Content(role="model", parts=[part])
            )
    return google_history


@gin.configurable
class Rhizosphere:
    def __init__(
        self,
        model_id: str = "gpt-4o",
        model_kwargs: Dict[str, any] = {},
        api_type: str = "message",
        add_system_prompt_to_history: bool = False,
    ):
        self.model_id = model_id
        self.model_kwargs = model_kwargs
        self.api_type = api_type
        self.add_system_prompt_to_history = add_system_prompt_to_history
        self._gemini_client = None

    def _get_gemini_client(self):
        if self._gemini_client:
            return self._gemini_client

        vertex_project = os.environ.get("VERTEX_PROJECT_ID")
        vertex_location = os.environ.get("VERTEX_LOCATION", "us-central1")

        if vertex_project:
            # Vertex AI Mode
            # We do NOT force 'v1' here because 'gemini-3-flash-preview' requires the Beta endpoint.
            logger.info(f"Initializing Gemini Client in Vertex AI Mode (Project: {vertex_project}, Location: {vertex_location})")
            self._gemini_client = genai.Client(
                vertexai=True,
                project=vertex_project,
                location=vertex_location
            )
        else:
            # AI Studio Mode
            logger.info("Initializing Gemini Client in AI Studio Mode")
            self._gemini_client = genai.Client(
                api_key=os.environ.get("GOOGLE_API_KEY")
            )
        
        return self._gemini_client

    def _parse_claude_event(self, chunk: Dict[str, Any]) -> str:
        if self.api_type == "messages":
            if chunk.get("type") == "content_block_delta":
                if chunk.get("delta", {}).get("type") == "text_delta":
                    yield chunk["delta"]["text"]
        else:
            yield chunk["completion"]

    def _configure_gemini_thinking(self, model_kwargs: dict[str, Any]):
        thinking_args = {}
        if "thinking_budget" in model_kwargs:
            thinking_args["thinking_budget"] = model_kwargs.pop("thinking_budget")
        if "thinking_level" in model_kwargs:
            thinking_args["thinking_level"] = model_kwargs.pop("thinking_level")
        if "include_thoughts" in model_kwargs:
            thinking_args["include_thoughts"] = model_kwargs.pop("include_thoughts")

        if thinking_args:
            model_kwargs["thinking_config"] = types.ThinkingConfig(**thinking_args)

    def _merge_kwargs(self, overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        combined = self.model_kwargs.copy()
        if overrides:
            combined.update(overrides)
        return combined

    # Restoring retry with tighter wait (2 seconds fixed) as requested
    @retry(stop=stop_after_attempt(10), wait=wait_random_exponential(multiplier=1, max=60))
    async def stream_call(
        self,
        chat_history: List[Dict[str, str]],
        system_prompt: str,
        model_id: str,
        model_kwargs: Optional[dict[str, Any]] = None,
        prefill: str = "",
    ) -> AsyncGenerator[str, None]:
        
        final_kwargs = self._merge_kwargs(model_kwargs)

        if "gpt" in model_id:

            try:
                client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
                messages = [{'role': 'system', 'content': system_prompt}] + chat_history
                response = await client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    temperature=final_kwargs.get("temperature", 0.1),
                    stream=True,
                )
                content_yielded = False
                async for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        content_yielded = True
                        yield chunk.choices[0].delta.content

                if not content_yielded:
                    raise ValueError("OpenAI stream call returned no content.")

            except Exception as e:
                logger.error(f"Error in OpenAI stream call: {e}")
                raise

        elif "gemini" in model_id:
            client = self._get_gemini_client()
            self._configure_gemini_thinking(final_kwargs)

            converted_convo = convert_history_to_gemini(chat_history)
            content_yielded = False
            
            try:
                async for chunk in await client.aio.models.generate_content_stream(
                    model=model_id,
                    contents=converted_convo,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        **final_kwargs
                    )
                ):
                    if hasattr(chunk, "finish_reason"):
                        break
                    if hasattr(chunk, "text") and chunk.text is not None:
                        content_yielded = True
                        yield chunk.text
            except RetryError as re:
                # Unwrap SDK retry error to see real cause
                cause = re.last_attempt.exception()
                logger.error(f"Gemini Stream RetryError Caused By: {type(cause).__name__} - {cause}")
                raise cause
            except Exception as e:
                logger.error(f"Gemini Stream Error: {type(e).__name__} - {e}")
                raise e

            if not content_yielded:
                raise ValueError("Gemini stream call returned no content.")

        else:
            raise ValueError(
                f"{self.model_id} is not a recognized or supported model."
            )

    # Restoring retry with tighter wait (2 seconds fixed)
    @retry(stop=stop_after_attempt(10), wait=wait_random_exponential(multiplier=1, max=60))
    async def structured_call(
        self,
        chat_history: List[Dict[str, str]],
        system_prompt: str,
        model_id: str,
        pydantic_obj: object,
        model_kwargs: Optional[dict[str, Any]] = None,
    ) -> object:
        #print("Sending Generate Request to OpenAI")
        final_kwargs = self._merge_kwargs(model_kwargs)

        if "gpt" in model_id:
            client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            messages = [{'role': 'system', 'content': system_prompt}] + chat_history

            # a ChatCompletion request
            response = await client.beta.chat.completions.parse(
                model=model_id,
                messages=messages,
                temperature=final_kwargs.get("temperature", 0.1),
                response_format=pydantic_obj,
            )
            parsed_response = response.choices[0].message.parsed
            if parsed_response is None:
                raise ValueError("OpenAI structured call returned None.")
            return parsed_response

        elif "gemini" in model_id:
            self._configure_gemini_thinking(final_kwargs)
            client = self._get_gemini_client()

            converted_convo = convert_history_to_gemini(chat_history)
            
            # DEBUG: Log payload size to check for massive context
            #payload_preview = json.dumps(chat_history)[:500]
            #logger.info(f"Sending Gemini Request: {model_id} | History Count: {len(chat_history)} | System Prompt Len: {len(system_prompt)}")
            #logger.debug(f"Payload Preview: {payload_preview}...")

            try:
                response = await client.aio.models.generate_content(
                    model=model_id,
                    contents=converted_convo,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        response_mime_type="application/json",
                        response_schema=pydantic_obj,
                        **final_kwargs
                    ),
                )
            except RetryError as re:
                # CRITICAL: Unwrap the Tenacity error from the Google SDK if present
                cause = re.last_attempt.exception()
                logger.error(f"Gemini SDK RetryError: {type(cause).__name__} - {cause}")
                raise cause
            except Exception as e:
                logger.error(f"Gemini API Error (RAW): {type(e).__name__} - {str(e)}")
                raise e

            # logger.debug(f"{response.usage_metadata = }")
            parsed_response = response.parsed
            if parsed_response is None:
                raise ValueError("Gemini structured call returned None.")
            return parsed_response

        else:
            raise NotImplementedError(f"Formatted Responses not supported for {self.model_id}.")

    async def structured_call_stream(
        self,
        chat_history: List[Dict[str, str]],
        system_prompt: str,
        model_id: str,
        pydantic_obj: object,
        model_kwargs: Optional[dict[str, Any]] = None,
    ) -> AsyncGenerator[Tuple[str, Any], None]:
        """
        Calls the model and streams a response that is expected to be a JSON object
        parsable by the provided Pydantic model.
        Includes manual retry logic for connection errors (e.g., 503) only if no data has been sent yet.
        """
        final_kwargs = self._merge_kwargs(model_kwargs)
        max_retries = 6 # Increased to allow for ~1 min backoff
        retry_delay = 1 # Start small

        for attempt in range(max_retries):
            has_yielded = False
            try:
                if "gpt" in model_id:
                    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
                    messages = [{'role': 'system', 'content': system_prompt}] + chat_history

                    tool_name = "structured_response"
                    tools = [
                        {
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "description": f"Parses the user's request and responds in the format of the {pydantic_obj.__name__} model.",
                                "parameters": pydantic_obj.model_json_schema()
                            }
                        }
                    ]

                    response_stream = await client.chat.completions.create(
                        model=model_id,
                        messages=messages,
                        temperature=final_kwargs.get("temperature", 0.1),
                        tools=tools,
                        tool_choice={"type": "function", "function": {"name": tool_name}},
                        stream=True,
                        stream_options={"include_usage": True}
                    )

                    async for chunk in response_stream:
                        if chunk.usage:
                            yield ("usage", {
                                "prompt_tokens": chunk.usage.prompt_tokens,
                                "completion_tokens": chunk.usage.completion_tokens,
                                "total_tokens": chunk.usage.total_tokens
                            })

                        if chunk.choices and chunk.choices[0].delta.tool_calls:
                            for tool_call in chunk.choices[0].delta.tool_calls:
                                if tool_call.function and tool_call.function.arguments:
                                    has_yielded = True
                                    yield ("json", tool_call.function.arguments)

                    if not has_yielded:
                        raise ValueError("OpenAI structured stream call returned no content.")

                elif "gemini" in model_id:
                    self._configure_gemini_thinking(final_kwargs)
                    client = self._get_gemini_client()

                    converted_convo = convert_history_to_gemini(chat_history)

                    response_stream = await client.aio.models.generate_content_stream(
                        model=model_id,
                        contents=converted_convo,
                        config=types.GenerateContentConfig(
                            system_instruction=system_prompt,
                            response_mime_type="application/json",
                            response_schema=pydantic_obj.model_json_schema(),
                            **final_kwargs
                        ),
                    )

                    async for chunk in response_stream:
                        if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                            yield ("usage", {
                                "prompt_tokens": chunk.usage_metadata.prompt_token_count,
                                "completion_tokens": chunk.usage_metadata.candidates_token_count,
                                "total_tokens": chunk.usage_metadata.total_token_count
                            })

                        if hasattr(chunk, "candidates") and chunk.candidates:
                            for part in chunk.candidates[0].content.parts:
                                # Check for thought signature on the part
                                if hasattr(part, "thought_signature") and part.thought_signature:
                                    raw_sig = part.thought_signature

                                    # Encode signature to Base64 to ensure safe JSON serialization
                                    # The signature might contain non-UTF-8 bytes (represented as surrogates in Python strings)
                                    try:
                                        if isinstance(raw_sig, str):
                                            sig_bytes = raw_sig.encode("utf-8", "surrogateescape")
                                        elif isinstance(raw_sig, bytes):
                                            sig_bytes = raw_sig
                                        else:
                                            sig_bytes = str(raw_sig).encode("utf-8")

                                        safe_sig = base64.b64encode(sig_bytes).decode("utf-8")
                                        yield ("signature", safe_sig)
                                    except Exception as e:
                                        logger.warning(f"Failed to encode thought signature: {e}")

                                # Skip parts that are clearly not text to avoid SDK warnings
                                if hasattr(part, "function_call") and part.function_call:
                                    continue

                                if not part.text:
                                    continue

                                if getattr(part, "thought", False):
                                    has_yielded = True
                                    yield ("thought", part.text)
                                else:
                                    has_yielded = True
                                    yield ("json", part.text)

                    if not has_yielded:
                        raise ValueError("Gemini structured stream call returned no content.")

                else:
                    raise NotImplementedError(f"Structured streaming not supported for {self.model_id}.")

                # If we complete successfully, break the retry loop
                return

            except Exception as e:
                # If we have already yielded data, we cannot retry safely as it would corrupt the JSON stream.
                # We also don't retry for NotImplementedError.
                if has_yielded or isinstance(e, NotImplementedError):
                    logger.error(f"Error in structured_call_stream (cannot retry): {e}")
                    raise e

                if attempt == max_retries - 1:
                    logger.error(f"Max retries reached for structured_call_stream: {e}")
                    raise e

                logger.warning(f"Retrying structured_call_stream (attempt {attempt + 1}/{max_retries}) due to: {e}")
                await asyncio.sleep(retry_delay * (2 ** attempt))

    async def count_tokens(
        self,
        chat_history: List[Dict[str, str]],
        system_prompt: str,
        model_id: str,
    ) -> int:
        if "gpt" in model_id:
            # Placeholder: Implement OpenAI token counting (e.g. tiktoken) if needed.
            return 0
        elif "gemini" in model_id:
            try:
                client = self._get_gemini_client()

                converted_history = convert_history_to_gemini(chat_history)

                response = await client.aio.models.count_tokens(
                    model=model_id,
                    contents=converted_history,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                    ),
                )
                return response.total_tokens
            except Exception as e:
                logger.error(f"Error counting tokens: {e}")
                return 0
        else:
            return 0

    async def batch_call(
        self,
        chat_history: List[Dict[str, str]],
        system_prompt: str,
        model_id: str,
        model_kwargs: Optional[dict[str, Any]] = None,
        prefill: str = "",
    ) -> str:
        text = ""
        async for token in self.stream_call(
            chat_history=chat_history,
            system_prompt=system_prompt,
            model_id=model_id,
            model_kwargs=model_kwargs,
            prefill=prefill,
        ):
            text += token
        return text

    async def get_embedding(
        self,
        text: str,
        model_id: str = "gemini-embedding-001",
    ) -> List[float]:
        if not text:
            return []

        try:
            client = self._get_gemini_client()
            # Vertex AI syntax slightly different from AI Studio for some versions,
            # but client.aio.models.embed_content should work for both if unified.
            response = await client.aio.models.embed_content(
                model=model_id,
                contents=text,
                config=types.EmbedContentConfig(task_type="CLUSTERING")
            )
            return response.embeddings[0].values
        except Exception as e:
            logger.error(f"Failed to get embedding: {e}")
            return []