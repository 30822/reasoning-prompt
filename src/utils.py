import asyncio
import json
import re
import sys
import os
import yaml
from pathlib import Path
from typing import Optional, List, Dict, Any

import httpx

# \u2028 LINE SEPARATOR, \u2029 PARAGRAPH SEPARATOR, \u0085 NEXT LINE
# -> \n to avoid "unusual line terminators" in JSON parsing
_LINE_TERM_PATTERN = re.compile(r"[\u2028\u2029\u0085]")


def _recursive_sanitize(obj):
    """재귀적으로 모든 문자열에서 \\u2028/\\u2029/\\u0085 제거."""
    if isinstance(obj, str):
        return _LINE_TERM_PATTERN.sub("\n", obj)
    if isinstance(obj, dict):
        return {k: _recursive_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_recursive_sanitize(x) for x in obj]
    return obj


def load_json_sanitized(path, encoding: str = "utf-8"):
    """JSON 파일 로드 후 모든 문자열에서 \\u2028/\\u2029/\\u0085 치환 (이전 evaluation 저장본 대응)."""
    p = Path(path)
    raw = p.read_text(encoding=encoding)
    data = json.loads(raw)
    return _recursive_sanitize(data)


def _sanitize_line_terminators(text: str) -> str:
    """Replace Unicode line separators with standard \\n."""
    if not text:
        return text
    return _LINE_TERM_PATTERN.sub("\n", text)


def _get_project_root():
    """return the project root"""
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    return project_root


def _get_config_path():
    """return the absolute path of the config file"""
    return _get_project_root() / "conf.d" / "conf.yaml"


def get_openai_key():
    try:
        config_path = _get_config_path()
        with open(config_path) as f:
            config = yaml.safe_load(f)
            return config["openai"]["key"]
    except FileNotFoundError:
        raise FileNotFoundError("conf.d/conf.yaml file not found. Please create one based on conf.example")
    except KeyError:
        raise KeyError("OpenAI key not found in conf.yaml. Please check configuration format")


def get_anthropic_key():
    try:
        config_path = _get_config_path()
        with open(config_path) as f:
            config = yaml.safe_load(f)
            return config["anthropic"]["key"]
    except FileNotFoundError:
        raise FileNotFoundError("conf.d/conf.yaml file not found. Please create one based on conf.example")
    except KeyError:
        raise KeyError("Anthropic key not found in conf.yaml. Please check configuration format")


def get_openrouter_key():
    try:
        config_path = _get_config_path()
        with open(config_path) as f:
            config = yaml.safe_load(f)
            return config["openrouter"]["key"]
    except FileNotFoundError:
        raise FileNotFoundError("conf.d/conf.yaml file not found. Please create one based on conf.example")
    except KeyError:
        raise KeyError("OpenRouter key not found in conf.yaml. Please check configuration format")


def strip_openai_prefix(model_name: str) -> str:
    """openai/o3 -> o3, openai/gpt-4o -> gpt-4o. Non-openai names returned as-is."""
    if model_name.startswith("openai/"):
        return model_name[len("openai/"):].strip()
    return model_name


# 요청 타임아웃(초). o3 등 느린 모델 대비 5분
OPENAI_REQUEST_TIMEOUT = 300.0

_openai_client = None


def _get_openai_client():
    """OpenAI 직접 호출용 클라이언트 (타임아웃 5분)."""
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
    """
    Call OpenAI API directly (no OpenRouter).
    Signature aligned with openrouter_client.call_llm.
    model_name: e.g. "openai/o3" or "o3" (openai/ prefix is stripped).
    """
    kwargs.pop("max_retries", None)
    kwargs.pop("retries", None)
    api_model = strip_openai_prefix(model_name)
    client = _get_openai_client()
    last_err: Optional[Exception] = None
    include_temperature = True  # some models (e.g. o3) only support default temperature

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
            # e.g. o3: "temperature does not support 0.0 ... Only the default (1) value is supported"
            if include_temperature and "temperature" in err_msg and ("does not support" in err_msg or "unsupported_value" in err_msg or "only the default" in err_msg):
                include_temperature = False
            elif attempt < retries - 1:
                await asyncio.sleep(2 ** attempt)
    raise RuntimeError(f"OpenAI direct call failed: {last_err}")


def load_dataset(data_name):
    dataset = {'train':[], 'test':[], 'valid':[]}
    project_root = _get_project_root()
    for key in dataset:
        try:
            data_path = project_root / "resources" / "data" / f"{data_name}-{key}.txt"
            with open(data_path, 'r') as infile:
                for episode_idx, line in enumerate(infile):
                    data_item = eval(line.strip('\n'))
                    data_item['episode_idx'] = episode_idx
                    dataset[key].append(data_item)
        except FileNotFoundError:
            continue
    return dataset