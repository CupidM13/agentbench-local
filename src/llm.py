"""LLM interaction layer supporting OpenAI-compatible endpoints with dual protocols:
native tool-calling and strict json_text formatting, with think-block stripping and auto-probing.
"""

from __future__ import annotations
import json
import re
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from openai import OpenAI


@dataclass
class ToolCallAction:
    tool: str
    args: Dict[str, Any]
    call_id: Optional[str] = None
    raw_text: Optional[str] = None


@dataclass
class LLMResponse:
    content: Optional[str] = None
    thought: Optional[str] = None
    tool_calls: List[ToolCallAction] = field(default_factory=list)
    final_answer: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_sec: float = 0.0
    raw_response: Optional[Dict[str, Any]] = None


THINK_REGEX = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
JSON_BLOCK_REGEX = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def strip_think_blocks(text: str) -> Tuple[str, Optional[str]]:
    """Strips <think>...</think> blocks from text.
    Returns (cleaned_text, extracted_thoughts).
    """
    if not text:
        return "", None
    thoughts: List[str] = []
    def _repl(match: re.Match) -> str:
        thoughts.append(match.group(1).strip())
        return ""
    cleaned = THINK_REGEX.sub(_repl, text).strip()
    joined_thoughts = "\n".join(thoughts) if thoughts else None
    return cleaned, joined_thoughts


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Extracts first valid JSON object from text or fenced codeblock."""
    if not text:
        return None
    # 1. Try direct json parse
    text_clean = text.strip()
    try:
        data = json.loads(text_clean)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    # 2. Try fenced block
    matches = JSON_BLOCK_REGEX.findall(text)
    for m in matches:
        try:
            data = json.loads(m.strip())
            if isinstance(data, dict):
                return data
        except Exception:
            continue

    # 3. Try finding outermost braces { ... }
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        snippet = text[first_brace : last_brace + 1]
        try:
            data = json.loads(snippet)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return None


class LLMClient:
    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
        temperature: float = 0.2,
        seed: int = 42,
        timeout: float = 90.0,
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.temperature = temperature
        self.seed = seed
        self.timeout = timeout
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self._protocol_cache: Dict[str, str] = {}

    def get_ollama_metadata(self, model: str) -> Dict[str, Any]:
        """Fetch model details from Ollama /api/show and version from /api/version."""
        host = self.base_url.replace("/v1", "")
        metadata: Dict[str, Any] = {
            "model": model,
            "ollama_version": "unknown",
            "parameter_size": "unknown",
            "quantization": "unknown",
            "family": "unknown",
        }
        try:
            with urllib.request.urlopen(f"{host}/api/version", timeout=3) as resp:
                data = json.loads(resp.read().decode())
                metadata["ollama_version"] = data.get("version", "unknown")
        except Exception:
            pass

        try:
            req = urllib.request.Request(
                f"{host}/api/show",
                data=json.dumps({"name": model}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
                details = data.get("details", {})
                metadata["parameter_size"] = details.get("parameter_size", "unknown")
                metadata["quantization"] = details.get("quantization_level", "unknown")
                metadata["family"] = details.get("family", "unknown")
        except Exception:
            pass
        return metadata

    def probe_protocol(self, model: str, force: Optional[str] = None) -> str:
        """Probe model capability: returns 'native' or 'json_text'."""
        if force in ("native", "json_text"):
            self._protocol_cache[model] = force
            return force
        if model in self._protocol_cache:
            return self._protocol_cache[model]

        # Probe native tool support
        probe_tool = {
            "type": "function",
            "function": {
                "name": "probe_tool",
                "description": "A probe tool",
                "parameters": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                },
            },
        }
        try:
            resp = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Ping"}],
                tools=[probe_tool],
                tool_choice="auto",
                max_tokens=5,
                temperature=self.temperature,
                seed=self.seed,
            )
            # If server accepted tools parameter without error, model supports native
            protocol = "native"
        except Exception as e:
            err_str = str(e).lower()
            if "does not support tools" in err_str or "400" in err_str:
                protocol = "json_text"
            else:
                # Default to json_text on ambiguity to guarantee execution
                protocol = "json_text"

        self._protocol_cache[model] = protocol
        return protocol

    def chat(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools_schema: Optional[List[Dict[str, Any]]] = None,
        protocol: str = "native",
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Executes a chat step under the specified protocol."""
        start_time = time.time()
        thought: Optional[str] = None
        tool_calls: List[ToolCallAction] = []
        final_answer: Optional[str] = None
        cleaned_content: Optional[str] = None

        if protocol == "native" and tools_schema:
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=tools_schema,
                    tool_choice="auto",
                    temperature=self.temperature,
                    seed=self.seed,
                    max_tokens=max_tokens,
                )
            except Exception as e:
                # If native fails with unsupported tools at runtime, bubble up or return error
                raise e

            latency = time.time() - start_time
            choice = response.choices[0]
            msg = choice.message
            raw_content = msg.content or ""
            cleaned_content, thought = strip_think_blocks(raw_content)

            if msg.tool_calls:
                for tc in msg.tool_calls:
                    fn_name = tc.function.name
                    try:
                        args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                    except Exception:
                        args = {"__raw__": tc.function.arguments}
                    tool_calls.append(
                        ToolCallAction(
                            tool=fn_name,
                            args=args,
                            call_id=tc.id,
                            raw_text=tc.function.arguments,
                        )
                    )
            else:
                final_answer = cleaned_content

            prompt_tokens = response.usage.prompt_tokens if response.usage else 0
            comp_tokens = response.usage.completion_tokens if response.usage else 0

            return LLMResponse(
                content=cleaned_content,
                thought=thought,
                tool_calls=tool_calls,
                final_answer=final_answer,
                prompt_tokens=prompt_tokens,
                completion_tokens=comp_tokens,
                latency_sec=latency,
                raw_response=response.model_dump(),
            )

        else:
            # json_text protocol
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=self.temperature,
                seed=self.seed,
                max_tokens=max_tokens,
            )
            latency = time.time() - start_time
            choice = response.choices[0]
            raw_content = choice.message.content or ""
            cleaned_content, thought = strip_think_blocks(raw_content)

            parsed_json = extract_json_object(cleaned_content)
            if parsed_json:
                if "tool" in parsed_json:
                    args = parsed_json.get("args", {})
                    if not isinstance(args, dict):
                        args = {"value": args}
                    tool_calls.append(
                        ToolCallAction(
                            tool=str(parsed_json["tool"]),
                            args=args,
                            raw_text=cleaned_content,
                        )
                    )
                elif "final_answer" in parsed_json:
                    ans = parsed_json["final_answer"]
                    final_answer = ans if isinstance(ans, str) else str(ans)
                else:
                    # Generic JSON output without tool or final_answer
                    final_answer = cleaned_content
            else:
                # No JSON found: treat whole text as final answer
                final_answer = cleaned_content

            prompt_tokens = response.usage.prompt_tokens if response.usage else 0
            comp_tokens = response.usage.completion_tokens if response.usage else 0

            return LLMResponse(
                content=cleaned_content,
                thought=thought,
                tool_calls=tool_calls,
                final_answer=final_answer,
                prompt_tokens=prompt_tokens,
                completion_tokens=comp_tokens,
                latency_sec=latency,
                raw_response=response.model_dump(),
            )
