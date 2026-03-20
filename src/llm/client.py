from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import TypeVar, cast

from openai import APIConnectionError, APITimeoutError, BadRequestError, InternalServerError, OpenAI
from openai import AuthenticationError, RateLimitError
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessageParam
from openai.types.create_embedding_response import CreateEmbeddingResponse

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

LOGGER = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.nitec-ai.kz/v1"
DEFAULT_MODEL = "openai/gpt-oss-120b"

T = TypeVar("T")


class LLMClientError(Exception):
    pass


class LLMConfigurationError(LLMClientError):
    pass


class LLMClient:
    api_key: str
    base_url: str
    model: str
    timeout: int
    max_retries: int
    _client: OpenAI

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: int = 30,
        max_retries: int = 3,
    ) -> None:
        resolved_api_key = (
            api_key.strip() if api_key is not None else os.getenv("LLM_API_KEY", "").strip()
        )
        if not resolved_api_key:
            raise LLMConfigurationError(
                "LLM API key is required. Set LLM_API_KEY environment variable."
            )

        resolved_base_url = (
            base_url.strip()
            if base_url is not None
            else os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL).strip()
        )
        resolved_model = (
            model.strip() if model is not None else os.getenv("LLM_MODEL", DEFAULT_MODEL).strip()
        )
        if not resolved_model:
            raise LLMConfigurationError(
                "LLM model is required. Set LLM_MODEL environment variable."
            )
        if timeout <= 0:
            raise ValueError("timeout must be > 0")
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")

        self.api_key = resolved_api_key
        self.base_url = resolved_base_url
        self.model = resolved_model
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)

        LOGGER.info(
            "LLM client initialized: base_url=%s model=%s api_key=%s timeout=%ss",
            self.base_url,
            self.model,
            self._mask_token(self.api_key),
            self.timeout,
        )

    @staticmethod
    def _mask_token(token: str) -> str:
        prefix = token[:8]
        return f"{prefix}***" if prefix else "***"

    @staticmethod
    def _estimate_tokens(messages: Sequence[dict[str, str]]) -> int:
        total_chars = 0
        for message in messages:
            total_chars += len(message.get("role", ""))
            total_chars += len(message.get("content", ""))
        return max(1, total_chars // 4)

    @staticmethod
    def _prepare_messages(messages: Sequence[dict[str, str]]) -> list[ChatCompletionMessageParam]:
        prepared: list[ChatCompletionMessageParam] = []
        allowed_roles = {"system", "user", "assistant", "developer", "tool", "function"}
        for raw_message in messages:
            role = raw_message.get("role", "").strip().lower()
            content = raw_message.get("content", "")
            if role not in allowed_roles:
                raise ValueError(f"Unsupported message role: {role}")
            message = cast(object, {"role": role, "content": content})
            prepared.append(cast(ChatCompletionMessageParam, message))
        return prepared

    def _log_request(self, messages: Sequence[dict[str, str]], *, stream: bool) -> None:
        LOGGER.info(
            "LLM request: model=%s messages=%s estimated_tokens=%s stream=%s",
            self.model,
            len(messages),
            self._estimate_tokens(messages),
            stream,
        )

    @staticmethod
    def _log_response(operation: str, usage: object) -> None:
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        total_tokens = getattr(usage, "total_tokens", None)
        LOGGER.info(
            "LLM response: operation=%s prompt_tokens=%s completion_tokens=%s total_tokens=%s",
            operation,
            prompt_tokens,
            completion_tokens,
            total_tokens,
        )

    @staticmethod
    def _is_non_retryable(error: Exception) -> bool:
        return isinstance(error, (AuthenticationError, BadRequestError)) or (
            type(error).__name__ == "InvalidRequestError"
        )

    def _run_with_retries(self, operation_name: str, operation: Callable[[], T]) -> T:
        retryable_errors = (
            RateLimitError,
            APIConnectionError,
            APITimeoutError,
            InternalServerError,
        )
        attempts_total = self.max_retries + 1

        for attempt in range(attempts_total):
            try:
                return operation()
            except Exception as error:  # noqa: BLE001
                if self._is_non_retryable(error):
                    LOGGER.error(
                        "LLM %s failed without retry: %s", operation_name, type(error).__name__
                    )
                    raise

                if not isinstance(error, retryable_errors):
                    LOGGER.error("LLM %s failed: %s", operation_name, type(error).__name__)
                    raise

                if attempt >= self.max_retries:
                    LOGGER.error(
                        "LLM %s failed after %s attempts: %s",
                        operation_name,
                        attempts_total,
                        type(error).__name__,
                    )
                    raise

                wait_seconds: int = 1 << attempt
                LOGGER.warning(
                    "LLM request failed (attempt %s/%s): %s",
                    attempt + 1,
                    self.max_retries,
                    type(error).__name__,
                )
                time.sleep(float(wait_seconds))

        raise LLMClientError(f"LLM {operation_name} failed unexpectedly")

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        response_format: dict | None = None,
    ) -> str:
        self._log_request(messages, stream=False)
        prepared_messages = self._prepare_messages(messages)

        def _operation() -> ChatCompletion:
            kwargs = dict(
                model=self.model,
                messages=prepared_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=self.timeout,
            )
            if response_format is not None:
                kwargs["response_format"] = response_format
            return self._client.chat.completions.create(**kwargs)

        response = self._run_with_retries("chat", _operation)
        self._log_response("chat", response.usage)
        if not response.choices:
            return ""
        message = response.choices[0].message
        content = message.content
        if not content:
            content = getattr(message, "reasoning_content", None)
        return content if isinstance(content, str) else ""

    def chat_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> Iterator[str]:
        self._log_request(messages, stream=True)
        prepared_messages = self._prepare_messages(messages)

        def _operation() -> Iterator[ChatCompletionChunk]:
            return self._client.chat.completions.create(
                model=self.model,
                messages=prepared_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                timeout=self.timeout,
            )

        stream = self._run_with_retries("chat_stream", _operation)
        chunk_count = 0
        char_count = 0

        for chunk in stream:
            chunk_count += 1
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            content = delta.content
            if not content:
                content = getattr(delta, "reasoning_content", None)
            if isinstance(content, str) and content:
                char_count += len(content)
                yield content

        LOGGER.info(
            "LLM response: operation=chat_stream chunks=%s chars=%s", chunk_count, char_count
        )

    def generate_embedding(self, text: str) -> list[float]:
        LOGGER.info(
            "LLM embedding request: model=%s estimated_tokens=%s",
            self.model,
            max(1, len(text) // 4),
        )

        def _operation() -> CreateEmbeddingResponse:
            return self._client.embeddings.create(
                model=self.model,
                input=text,
                timeout=self.timeout,
            )

        response = self._run_with_retries("generate_embedding", _operation)
        self._log_response("generate_embedding", response.usage)

        if not response.data:
            raise LLMClientError("Embedding response did not contain data")
        embedding = response.data[0].embedding
        LOGGER.info("LLM embedding response: dimensions=%s", len(embedding))
        return [float(value) for value in embedding]


def _read_prompt_file(path_str: str) -> str:
    path = Path(path_str)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Test nitec-ai.kz OpenAI-compatible LLM client")
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    _ = prompt_group.add_argument("--prompt", type=str, help="Prompt text")
    _ = prompt_group.add_argument("--file", type=str, help="Path to file containing prompt text")
    _ = parser.add_argument("--model", type=str, default=None, help="Override model name")
    _ = parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    _ = parser.add_argument(
        "--max-tokens", type=int, default=2000, help="Maximum completion tokens"
    )
    _ = parser.add_argument("--stream", action="store_true", help="Stream response output")
    _ = parser.add_argument("--timeout", type=int, default=30, help="Request timeout in seconds")
    _ = parser.add_argument("--base-url", type=str, default=None, help="Override API base URL")
    return parser


def main() -> int:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
    parser = _build_parser()
    args = parser.parse_args()

    prompt_arg = cast(str | None, getattr(args, "prompt", None))
    file_arg = cast(str | None, getattr(args, "file", None))
    model_arg = cast(str | None, getattr(args, "model", None))
    temperature_arg = float(cast(float, getattr(args, "temperature")))
    max_tokens_arg = int(cast(int, getattr(args, "max_tokens")))
    stream_arg = bool(cast(bool, getattr(args, "stream")))
    timeout_arg = int(cast(int, getattr(args, "timeout")))
    base_url_arg = cast(str | None, getattr(args, "base_url", None))

    prompt_text = prompt_arg if prompt_arg is not None else _read_prompt_file(file_arg or "")
    messages = [{"role": "user", "content": prompt_text}]

    try:
        client = LLMClient(model=model_arg, timeout=timeout_arg, base_url=base_url_arg)
        if stream_arg:
            for piece in client.chat_stream(
                messages=messages,
                temperature=temperature_arg,
                max_tokens=max_tokens_arg,
            ):
                print(piece, end="", flush=True)
            print()
        else:
            response = client.chat(
                messages=messages,
                temperature=temperature_arg,
                max_tokens=max_tokens_arg,
            )
            print(response)
        return 0
    except Exception as error:  # noqa: BLE001
        LOGGER.error("LLM CLI request failed: %s", type(error).__name__)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
