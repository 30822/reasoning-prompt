"""LLM routing: openai/* via OpenAI API, all other models via OpenRouter."""
import asyncio
import re
from typing import Optional, List, Dict, Any, Callable, Awaitable
import httpx
from openai import AsyncOpenAI
from src.utils import get_openrouter_key, call_llm_openai

_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
_LINE_TERM_PATTERN = re.compile(r"[\u2028\u2029\u0085]")
_client: Optional[AsyncOpenAI] = None


def _sanitize_line_terminators(text: str) -> str:
    if not text:
        return text
    return _LINE_TERM_PATTERN.sub("\n", text)


def get_llm_caller(model_name: str) -> Callable[..., Awaitable[str]]:
    """Return the async chat-completion caller for this model name."""
    m = (model_name or "").strip()
    if m.startswith("openai/"):
        return call_llm_openai
    return call_llm


def get_client(timeout_s: float = 120.0) -> AsyncOpenAI:
    global _client
    if _client is not None:
        return _client
    _client = AsyncOpenAI(
        api_key=get_openrouter_key(),
        base_url=_DEFAULT_BASE_URL,
        http_client=httpx.AsyncClient(timeout=timeout_s),
    )
    return _client


async def call_llm(
    model_name: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    retries: int = 3,
    response_format: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> str:
    client = get_client()
    last_err: Optional[Exception] = None
    kwargs.pop("max_retries", None)
    kwargs.pop("retries", None)

    base_payload = dict(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    for i in range(retries):
        try:
            payload = dict(base_payload)
            if response_format is not None:
                payload["response_format"] = response_format
            payload.update(kwargs)
            resp = await client.chat.completions.create(**payload)
            raw = resp.choices[0].message.content or ""
            return _sanitize_line_terminators(raw)
        except TypeError as e:
            msg = str(e).lower()
            if response_format is not None and ("response_format" in msg or "unexpected keyword" in msg):
                response_format = None
                last_err = e
                await asyncio.sleep(2 ** i)
                continue
            last_err = e
            await asyncio.sleep(2 ** i)
        except Exception as e:
            last_err = e
            await asyncio.sleep(2 ** i)

    raise RuntimeError(f"OpenRouter call failed: {last_err}")
