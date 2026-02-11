# src/openrouter_client.py
import asyncio
from typing import Optional, List, Dict, Any
import httpx
from openai import AsyncOpenAI
from src.utils import get_openrouter_key

_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
_client: Optional[AsyncOpenAI] = None


def get_client(timeout_s: float = 120.0) -> AsyncOpenAI:
    global _client
    if _client is not None:
        return _client

    key = get_openrouter_key()
    _client = AsyncOpenAI(
        api_key=key,
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
    """
    Standard chat-completions wrapper with retry.
    - response_format이 지원되지 않는 모델/라우터면 자동으로 제거하고 재시도
    - max_retries 같은 값이 kwargs로 와도 create()에 전달하지 않음(내부 retries로만 제어)
    """
    client = get_client()
    last_err: Optional[Exception] = None

    # 절대 create에 넘기면 안 되는 것들 (네가 겪은 max_retries 이슈 방지)
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

            # 1) response_format 먼저 시도
            if response_format is not None:
                payload["response_format"] = response_format

            # 2) 기타 옵션
            payload.update(kwargs)

            resp = await client.chat.completions.create(**payload)
            return resp.choices[0].message.content or ""

        except TypeError as e:
            # response_format 미지원(또는 OpenAI SDK/Router에서 인자 거부) 시 fallback
            msg = str(e).lower()
            if response_format is not None and ("response_format" in msg or "unexpected keyword" in msg):
                # response_format 없이 재시도
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