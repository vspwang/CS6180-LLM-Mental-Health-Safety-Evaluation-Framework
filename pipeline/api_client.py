import logging
import os
import time

from openai import OpenAI, AuthenticationError, RateLimitError, APITimeoutError, APIError

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
INITIAL_BACKOFF = 1  # seconds


def _get_client(base_url: str) -> OpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENROUTER_API_KEY environment variable is not set. "
            "Please set it before running: export OPENROUTER_API_KEY=your_key"
        )
    # No timeout on the client level; we handle timeouts via stream inactivity below.
    return OpenAI(api_key=api_key, base_url=base_url)


STREAM_CHUNK_TIMEOUT = 30  # seconds to wait for the next chunk before giving up


def call_model(
    model_id: str,
    messages: list,
    temperature: float,
    max_tokens: int,
    base_url: str = "https://openrouter.ai/api/v1",
) -> dict:
    """Call a model via OpenRouter using streaming. Returns dict with keys:
    response (str), status (str), response_time_ms (int), usage (dict).
    """
    client = _get_client(base_url)

    for attempt in range(MAX_RETRIES):
        start = time.time()
        try:
            stream = client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                stream_options={"include_usage": True},
            )

            chunks = []
            finish_reason = None
            usage = {"input_tokens": 0, "output_tokens": 0}
            last_chunk_time = time.time()

            for chunk in stream:
                # Guard against a stalled stream that stops sending chunks
                if time.time() - last_chunk_time > STREAM_CHUNK_TIMEOUT:
                    stream.close()
                    raise APITimeoutError(request=None)
                last_chunk_time = time.time()

                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    chunks.append(delta.content)
                if chunk.choices and chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason
                if chunk.usage:
                    usage["input_tokens"] = chunk.usage.prompt_tokens or 0
                    usage["output_tokens"] = chunk.usage.completion_tokens or 0

            elapsed_ms = int((time.time() - start) * 1000)
            content = "".join(chunks)

            if not content.strip() or finish_reason == "content_filter":
                return {
                    "response": content,
                    "status": "refused",
                    "response_time_ms": elapsed_ms,
                    "usage": usage,
                }

            return {
                "response": content,
                "status": "success",
                "response_time_ms": elapsed_ms,
                "usage": usage,
            }

        except AuthenticationError as e:
            raise EnvironmentError(
                f"Authentication failed. Check that OPENROUTER_API_KEY is valid. Details: {e}"
            ) from e

        except RateLimitError as e:
            elapsed_ms = int((time.time() - start) * 1000)
            if attempt < MAX_RETRIES - 1:
                backoff = INITIAL_BACKOFF * (2 ** attempt)
                logger.warning(
                    "Rate limit hit (attempt %d/%d). Retrying in %ds...",
                    attempt + 1, MAX_RETRIES, backoff,
                )
                time.sleep(backoff)
            else:
                logger.error("Rate limit hit after %d attempts.", MAX_RETRIES)
                return {
                    "response": str(e),
                    "status": "error",
                    "response_time_ms": elapsed_ms,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                }

        except APITimeoutError:
            elapsed_ms = int((time.time() - start) * 1000)
            if attempt < MAX_RETRIES - 1:
                backoff = INITIAL_BACKOFF * (2 ** attempt)
                logger.warning(
                    "Timeout (attempt %d/%d). Retrying in %ds...",
                    attempt + 1, MAX_RETRIES, backoff,
                )
                time.sleep(backoff)
            else:
                logger.error("Request timed out after %d attempts.", MAX_RETRIES)
                return {
                    "response": "Request timed out",
                    "status": "timeout",
                    "response_time_ms": elapsed_ms,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                }

        except APIError as e:
            elapsed_ms = int((time.time() - start) * 1000)
            logger.error("API error: %s", e)
            return {
                "response": str(e),
                "status": "error",
                "response_time_ms": elapsed_ms,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            }

    # Should not reach here, but just in case
    return {
        "response": "Request timed out",
        "status": "timeout",
        "response_time_ms": 0,
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }
