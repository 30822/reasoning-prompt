"""API keys, JSON sanitization, and direct OpenAI calls."""
import asyncio
import json
import re
import yaml
from pathlib import Path
from typing import Optional, List, Dict, Any

import httpx

_LINE_TERM_PATTERN = re.compile(r"[\u2028\u2029\u0085]")


def _recursive_sanitize(obj):
    if isinstance(obj, str):
        return _LINE_TERM_PATTERN.sub("\n", obj)
    if isinstance(obj, dict):
        return {k: _recursive_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_recursive_sanitize(x) for x in obj]
    return obj


def load_json_sanitized(path, encoding: str = "utf-8"):
    """Load JSON and replace Unicode line separators with \\n."""
    p = Path(path)
    data = json.loads(p.read_text(encoding=encoding))
    return _recursive_sanitize(data)


def _sanitize_line_terminators(text: str) -> str:
    if not text:
        return text
    return _LINE_TERM_PATTERN.sub("\n", text)


def _get_project_root():
    return Path(__file__).resolve().parent.parent


def _get_config_path():
    return _get_project_root() / "conf.d" / "conf.yaml"


def get_openai_key():
    try:
        with open(_get_config_path()) as f:
            return yaml.safe_load(f)["openai"]["key"]
    except FileNotFoundError:
        raise FileNotFoundError("conf.d/conf.yaml file not found. Please create one based on conf.example")
    except KeyError:
        raise KeyError("OpenAI key not found in conf.yaml. Please check configuration format")


def get_openrouter_key():
    try:
        with open(_get_config_path()) as f:
            return yaml.safe_load(f)["openrouter"]["key"]
    except FileNotFoundError:
        raise FileNotFoundError("conf.d/conf.yaml file not found. Please create one based on conf.example")
    except KeyError:
        raise KeyError("OpenRouter key not found in conf.yaml. Please check configuration format")


def strip_openai_prefix(model_name: str) -> str:
    if model_name.startswith("openai/"):
        return model_name[len("openai/"):].strip()
    return model_name


OPENAI_REQUEST_TIMEOUT = 300.0
_openai_client = None


def _get_openai_client():
    global _openai_client
    if _openai_client is not None:
        return _openai_client
    try:
        from openai import AsyncOpenAI
    except ImportError:
        raise ImportError("openai package required. pip install openai")
    _openai_client = AsyncOpenAI(
        api_key=get_openai_key(),
        http_client=httpx.AsyncClient(timeout=OPENAI_REQUEST_TIMEOUT),
    )
    return _openai_client


async def call_llm_openai(
    model_name: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    retries: int = 3,
    response_format: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> str:
    kwargs.pop("max_retries", None)
    kwargs.pop("retries", None)
    api_model = strip_openai_prefix(model_name)
    client = _get_openai_client()
    last_err: Optional[Exception] = None
    include_temperature = True  # some models (e.g. o3) reject non-default temperature

    for attempt in range(retries):
        payload: Dict[str, Any] = {
            "model": api_model,
            "messages": messages,
        }
        if include_temperature:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if response_format is not None:
            payload["response_format"] = response_format

        try:
            resp = await client.chat.completions.create(**payload)
            raw = (resp.choices[0].message.content or "").strip()
            return _sanitize_line_terminators(raw)
        except Exception as e:
            last_err = e
            err_msg = str(e).lower()
            if include_temperature and "temperature" in err_msg and (
                "does not support" in err_msg or "unsupported_value" in err_msg or "only the default" in err_msg
            ):
                include_temperature = False
            elif attempt < retries - 1:
                await asyncio.sleep(2 ** attempt)
    raise RuntimeError(f"OpenAI direct call failed: {last_err}")
