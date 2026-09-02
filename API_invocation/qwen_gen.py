import json
import multiprocessing
import os
import time
import threading
from typing import Any

from dashscope import Generation
from openai import OpenAI

#qwen and gpt-4o 
DEFAULT_MODEL = "qwen2.5-72b-instruct"
DEFAULT_MAX_TOKENS = 8192
DEFAULT_TEMPERATURE = 0.0
DEFAULT_REQUEST_TIMEOUT = int(os.getenv("DASHSCOPE_REQUEST_TIMEOUT", "120"))
DEFAULT_OPENAI_BASE_URL = os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL") or "https://api.openai-proxy.org/v1"


def _safe_get(obj: Any, key: str, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _extract_response_content(response: Any):
    status_code = _safe_get(response, "status_code")
    if status_code and int(status_code) >= 400:
        code = _safe_get(response, "code", "")
        message = _safe_get(response, "message", "")
        request_id = _safe_get(response, "request_id", "")
        raise ValueError(
            "DashScope request failed "
            f"(status_code={status_code}, code={code}, message={message}, request_id={request_id})"
        )

    output = _safe_get(response, "output", {})
    if not output:
        code = _safe_get(response, "code", "")
        message = _safe_get(response, "message", "")
        request_id = _safe_get(response, "request_id", "")
        raise ValueError(
            "DashScope response missing output "
            f"(status_code={status_code}, code={code}, message={message}, request_id={request_id})"
        )

    choices = _safe_get(output, "choices", [])
    if not choices:
        code = _safe_get(response, "code", "")
        message = _safe_get(response, "message", "")
        request_id = _safe_get(response, "request_id", "")
        raise ValueError(
            "DashScope response does not contain any choices "
            f"(status_code={status_code}, code={code}, message={message}, request_id={request_id})"
        )

    first_choice = choices[0]
    message = _safe_get(first_choice, "message", {})
    content = _safe_get(message, "content", "")

    if isinstance(content, str):
        return content
    if content is None:
        return ""
    return json.dumps(content, ensure_ascii=False)


def _generation_process_worker(result_conn, call_kwargs):
    try:
        response = Generation.call(**call_kwargs)
        result_conn.send(
            (
                "ok",
                {
                    "content": _extract_response_content(response),
                    "usage": _safe_get(response, "usage"),
                },
            )
        )
    except BaseException as exc:
        try:
            result_conn.send(("error", f"{type(exc).__name__}: {exc}"))
        except BaseException:
            pass
    finally:
        try:
            result_conn.close()
        except BaseException:
            pass


class DashscopeClient:
    def __init__(
            self,
            api_key=None,
            model=DEFAULT_MODEL,
            max_tokens=DEFAULT_MAX_TOKENS,
            temperature=DEFAULT_TEMPERATURE,
            base_url=None,
    ):
        """
        初始化 DashscopeClient 对象。

        :param api_key: DashScope API 密钥，未提供时优先读取环境变量，
                        再回退到当前文件中的默认值。
        :param model: 模型名称，默认使用 qwen2.5-72b-instruct。
        :param max_tokens: 模型单次调用的最大输出 tokens，默认 8192。
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY") 
        self.base_url = base_url or DEFAULT_OPENAI_BASE_URL
        self.model = model or DEFAULT_MODEL
        self.max_tokens = int(max_tokens or DEFAULT_MAX_TOKENS)
        self.temperature = float(DEFAULT_TEMPERATURE if temperature is None else temperature)
        self.openai_client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self._usage_lock = threading.Lock()
        self.reset_usage()

    def reset_usage(self):
        with self._usage_lock:
            self.last_usage = {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            }
            self.total_usage = {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            }
            self.api_call_count = 0

    def get_usage_stats(self):
        with self._usage_lock:
            return {
                "input_tokens": self.total_usage["input_tokens"],
                "output_tokens": self.total_usage["output_tokens"],
                "total_tokens": self.total_usage["total_tokens"],
                "api_call_count": self.api_call_count,
                "last_usage": dict(self.last_usage),
            }

    def _normalize_usage(self, usage: Any):
        if usage is None:
            print("[Warning] DashScope response missing usage; counted as 0 tokens for this call.")
            return {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            }

        if not isinstance(usage, dict):
            usage = {
                "input_tokens": _safe_get(usage, "input_tokens", _safe_get(usage, "prompt_tokens", 0)),
                "output_tokens": _safe_get(usage, "output_tokens", _safe_get(usage, "completion_tokens", 0)),
                "total_tokens": _safe_get(usage, "total_tokens", 0),
            }

        input_tokens = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
        output_tokens = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
        total_tokens = int(usage.get("total_tokens") or (input_tokens + output_tokens))

        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }

    def _update_usage(self, usage: Any):
        normalized = self._normalize_usage(usage)
        with self._usage_lock:
            self.last_usage = normalized
            self.total_usage["input_tokens"] += normalized["input_tokens"]
            self.total_usage["output_tokens"] += normalized["output_tokens"]
            self.total_usage["total_tokens"] += normalized["total_tokens"]
            self.api_call_count += 1

    def _extract_content(self, response: Any):
        return _extract_response_content(response)

    def _uses_openai_chat(self) -> bool:
        return str(self.model or "").lower().startswith(("gpt-", "o1", "o3", "o4"))

    def _openai_chat_call(self, timeout_seconds: int, **call_kwargs):
        response = self.openai_client.chat.completions.create(
            model=call_kwargs["model"],
            messages=call_kwargs["messages"],
            max_tokens=call_kwargs["max_tokens"],
            temperature=call_kwargs["temperature"],
            timeout=timeout_seconds,
        )
        return {
            "content": response.choices[0].message.content or "",
            "usage": response.usage,
        }

    def _generation_call_with_timeout(self, timeout_seconds: int, **call_kwargs):
        """
        DashScope's SDK-level timeout is not always prompt in long generations.
        Run the blocking SDK call in a short-lived child process so the caller
        can retry or shrink a batch instead of waiting forever.
        """
        use_process_timeout = os.getenv("DASHSCOPE_USE_PROCESS_TIMEOUT", "").strip().lower()
        if os.name == "nt" and use_process_timeout not in {"1", "true", "yes"}:
            if self._uses_openai_chat():
                return self._openai_chat_call(timeout_seconds, **call_kwargs)
            response = Generation.call(**call_kwargs)
            return {
                "content": _extract_response_content(response),
                "usage": _safe_get(response, "usage"),
            }

        parent_conn, child_conn = multiprocessing.Pipe(duplex=False)
        process = multiprocessing.Process(
            target=_generation_process_worker,
            args=(child_conn, call_kwargs),
            daemon=True,
        )
        process.start()
        child_conn.close()
        deadline = time.monotonic() + max(1, int(timeout_seconds))
        try:
            while time.monotonic() < deadline:
                if parent_conn.poll(0.2):
                    status, payload = parent_conn.recv()
                    process.join(timeout=5)
                    if process.is_alive():
                        process.terminate()
                        process.join(timeout=5)
                    if status == "error":
                        raise RuntimeError(payload)
                    return payload
                if not process.is_alive():
                    process.join(timeout=1)
                    if parent_conn.poll(1):
                        status, payload = parent_conn.recv()
                        if status == "error":
                            raise RuntimeError(payload)
                        return payload
                    raise RuntimeError(
                        f"DashScope worker exited with code {process.exitcode} without a response"
                    )

            process.terminate()
            process.join(timeout=5)
            if process.is_alive() and hasattr(process, "kill"):
                process.kill()
                process.join(timeout=1)
            raise TimeoutError(f"DashScope request exceeded {timeout_seconds}s")
        finally:
            try:
                parent_conn.close()
            except BaseException:
                pass

    def send_message(self, messages=None, max_tokens=None, result_format="message", delay=0.0, retries=3, **kwargs):
        """
        与旧客户端保持兼容的调用接口。
        """
        if not messages:
            raise ValueError("messages 参数不能为空")

        if not self.api_key:
            raise ValueError("API 密钥未提供且未从环境变量中找到")

        effective_max_tokens = self.max_tokens if max_tokens is None else int(max_tokens)
        effective_max_tokens = min(effective_max_tokens, DEFAULT_MAX_TOKENS)
        effective_temperature = kwargs.pop("temperature", self.temperature)
        if effective_temperature is None:
            effective_temperature = DEFAULT_TEMPERATURE
        effective_temperature = float(effective_temperature)

        time.sleep(delay)
        if "timeout" in kwargs and "request_timeout" not in kwargs:
            kwargs["request_timeout"] = kwargs.pop("timeout")
        if kwargs.get("request_timeout") is None:
            kwargs["request_timeout"] = DEFAULT_REQUEST_TIMEOUT
        try:
            kwargs["request_timeout"] = int(kwargs["request_timeout"])
        except (TypeError, ValueError):
            kwargs["request_timeout"] = DEFAULT_REQUEST_TIMEOUT

        last_error = None
        result = None
        for attempt in range(max(1, retries)):
            try:
                result = self._generation_call_with_timeout(
                    kwargs["request_timeout"],
                    api_key=self.api_key,
                    model=self.model,
                    messages=messages,
                    result_format=result_format,
                    max_tokens=effective_max_tokens,
                    temperature=effective_temperature,
                    **kwargs,
                )
                break
            except Exception as exc:
                last_error = exc
                if attempt >= retries - 1:
                    raise
                wait_seconds = min(20, 2 ** attempt)
                print(f"[Warning] DashScope request failed ({exc}); retrying in {wait_seconds}s...")
                time.sleep(wait_seconds)
        else:
            raise last_error

        self._update_usage(result.get("usage"))
        return result.get("content", "")

    def generate_response(self, messages=None, result_format="message", delay=0.0, max_tokens=None, **kwargs):
        return self.send_message(
            messages=messages,
            max_tokens=max_tokens,
            result_format=result_format,
            delay=delay,
            **kwargs,
        )


shared_qwen_client = DashscopeClient(model=DEFAULT_MODEL)


def reset_shared_usage():
    shared_qwen_client.reset_usage()


def get_shared_usage():
    return shared_qwen_client.get_usage_stats()


if __name__ == "__main__":
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "你是谁？"},
    ]

    client = DashscopeClient()
    response = client.send_message(messages=messages)
    print(response)
    print(client.get_usage_stats())
