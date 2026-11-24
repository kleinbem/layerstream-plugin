from collections.abc import Callable, Iterator, Mapping, Sequence
from os import PathLike
from types import TracebackType
from typing import Any, Literal, overload

__version__: str

class StreamTagParser:
    def __init__(self) -> None: ...
    @property
    def has_tool_calls(self) -> bool: ...
    def push(self, piece: str) -> list[dict[str, Any]]: ...
    def flush(self) -> list[dict[str, Any]]: ...

class ServerHandle:
    @property
    def host(self) -> str: ...
    @property
    def port(self) -> int: ...
    @property
    def base_url(self) -> str: ...
    def stop(self) -> None: ...
    def wait(self) -> None: ...
    def __enter__(self) -> ServerHandle: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc_val: BaseException | None = None,
        exc_tb: TracebackType | None = None,
    ) -> bool: ...

class TokenStream(Iterator[str]):
    def __iter__(self) -> TokenStream: ...
    def __next__(self) -> str: ...
    def __enter__(self) -> TokenStream: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc_val: BaseException | None = None,
        exc_tb: TracebackType | None = None,
    ) -> bool: ...
    def close(self) -> None: ...
    @property
    def summary(self) -> dict[str, Any] | None: ...

class NativeWorkerStream(Iterator[str]):
    def __iter__(self) -> NativeWorkerStream: ...
    def __next__(self) -> str: ...
    def __enter__(self) -> NativeWorkerStream: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc_val: BaseException | None = None,
        exc_tb: TracebackType | None = None,
    ) -> bool: ...
    def close(self) -> None: ...
    @property
    def summary(self) -> dict[str, Any] | None: ...

class NativeWorkerClient:
    @staticmethod
    def spawn(
        program: str,
        args: list[str],
        open_json: str,
        on_progress: Callable[[dict[str, Any]], Any] | None = None,
    ) -> NativeWorkerClient: ...
    @property
    def is_loaded(self) -> bool: ...
    @property
    def worker_pid(self) -> int | None: ...
    @property
    def model_path(self) -> str | None: ...
    @property
    def last_summary(self) -> dict[str, Any] | None: ...
    def stats(self) -> dict[str, Any]: ...
    def tokenize(self, text: str, add_special: bool, parse_special: bool) -> list[int]: ...
    def detokenize(self, tokens: list[int]) -> str: ...
    def apply_chat_template(
        self,
        messages_json: str,
        add_generation_prompt: bool = True,
        tools_json: str | None = None,
    ) -> str: ...
    def kv_clear(self) -> None: ...
    def kv_trim(self, pos_min: int) -> bool: ...
    def save_state(self, path: str) -> int: ...
    def load_state(self, path: str) -> int: ...
    def generate(self, prompt: str, params_json: str) -> tuple[str, dict[str, Any] | None]: ...
    def chat(
        self, messages_json: str, params_json: str
    ) -> tuple[str, dict[str, Any] | None, dict[str, Any] | None]: ...
    def generate_stream(self, prompt: str, params_json: str) -> NativeWorkerStream: ...
    def chat_stream(self, messages_json: str, params_json: str) -> NativeWorkerStream: ...
    def unload(self) -> None: ...

def run_worker_server(port: int) -> None: ...

class NativeAutoModel:
    @classmethod
    def from_pretrained(
        cls,
        model: str,
        *,
        draft_model: str | None = None,
        quant: str | None = None,
        revision: str | None = None,
        cache_dir: str | PathLike[str] | None = None,
        vram_budget_mb: int | str | None = "auto",
        resident_layers: int | None = None,
        ctx_size: int = 4096,
        batch_size: int = 512,
        threads: int | None = None,
        kv_cache_type: str = "q8_0",
        use_gpu: bool = True,
        prefer_direct_io: bool = True,
        cpu_embeddings: bool = False,
        cpu_head: bool = False,
        stream_head: bool = False,
        moe_active_streaming: bool = True,
        max_draft_tokens: int = 5,
        verbose: bool = False,
        on_progress: Callable[[dict[str, Any]], Any] | None = None,
    ) -> NativeAutoModel: ...
    @overload
    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 0.95,
        min_p: float = 0.0,
        top_k: int = 40,
        repeat_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        penalty_last_n: int = 64,
        seed: int = 42,
        stop: str | Sequence[str] | None = None,
        stream: Literal[True],
        ngram_speculation: bool = True,
        max_draft_tokens: int | None = None,
        grammar: str | None = None,
        json_schema: Any = None,
        return_summary: Literal[False] = False,
    ) -> TokenStream: ...
    @overload
    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 0.95,
        min_p: float = 0.0,
        top_k: int = 40,
        repeat_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        penalty_last_n: int = 64,
        seed: int = 42,
        stop: str | Sequence[str] | None = None,
        stream: Literal[False] = False,
        ngram_speculation: bool = True,
        max_draft_tokens: int | None = None,
        grammar: str | None = None,
        json_schema: Any = None,
        return_summary: Literal[True],
    ) -> tuple[str, dict[str, Any]]: ...
    @overload
    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 0.95,
        min_p: float = 0.0,
        top_k: int = 40,
        repeat_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        penalty_last_n: int = 64,
        seed: int = 42,
        stop: str | Sequence[str] | None = None,
        stream: bool = False,
        ngram_speculation: bool = True,
        max_draft_tokens: int | None = None,
        grammar: str | None = None,
        json_schema: Any = None,
        return_summary: Literal[False] = False,
    ) -> str | TokenStream: ...
    @overload
    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 0.95,
        min_p: float = 0.0,
        top_k: int = 40,
        repeat_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        penalty_last_n: int = 64,
        seed: int = 42,
        stop: str | Sequence[str] | None = None,
        stream: Literal[True],
        add_generation_prompt: bool = True,
        ngram_speculation: bool = True,
        max_draft_tokens: int | None = None,
        tools: Sequence[Mapping[str, Any]] | None = None,
        grammar: str | None = None,
        json_schema: Any = None,
        parse_output: bool = False,
        return_summary: Literal[False] = False,
    ) -> TokenStream: ...
    @overload
    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 0.95,
        min_p: float = 0.0,
        top_k: int = 40,
        repeat_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        penalty_last_n: int = 64,
        seed: int = 42,
        stop: str | Sequence[str] | None = None,
        stream: Literal[False] = False,
        add_generation_prompt: bool = True,
        ngram_speculation: bool = True,
        max_draft_tokens: int | None = None,
        tools: Sequence[Mapping[str, Any]] | None = None,
        grammar: str | None = None,
        json_schema: Any = None,
        parse_output: bool = False,
        return_summary: Literal[True],
    ) -> tuple[str | dict[str, Any], dict[str, Any]]: ...
    @overload
    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 0.95,
        min_p: float = 0.0,
        top_k: int = 40,
        repeat_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        penalty_last_n: int = 64,
        seed: int = 42,
        stop: str | Sequence[str] | None = None,
        stream: bool = False,
        add_generation_prompt: bool = True,
        ngram_speculation: bool = True,
        max_draft_tokens: int | None = None,
        tools: Sequence[Mapping[str, Any]] | None = None,
        grammar: str | None = None,
        json_schema: Any = None,
        parse_output: bool = False,
        return_summary: Literal[False] = False,
    ) -> str | dict[str, Any] | TokenStream: ...
    def tokenize(
        self,
        text: str,
        *,
        add_special: bool = False,
        parse_special: bool = False,
    ) -> list[int]: ...
    def detokenize(self, tokens: Sequence[int]) -> str: ...
    def apply_chat_template(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        add_generation_prompt: bool = True,
        tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> str: ...
    def kv_clear(self) -> None: ...
    def kv_trim(self, pos_min: int) -> bool: ...
    def save_state(self, path: str) -> int: ...
    def load_state(self, path: str) -> int: ...
    def stats(self) -> dict[str, Any]: ...
    @property
    def last_summary(self) -> dict[str, Any] | None: ...
    @property
    def is_loaded(self) -> bool: ...
    @property
    def model_path(self) -> str | None: ...
    def unload(self) -> None: ...

def probe_gpu_memory(use_gpu: bool = True) -> tuple[int, int] | None: ...
def probe_gpu_device(use_gpu: bool = True) -> tuple[int, int, str] | None: ...
def inspect_gguf(path: str | PathLike[str]) -> dict[str, Any]: ...
def pull(
    spec: str,
    *,
    cache_dir: str | PathLike[str] | None = None,
    quant: str | None = None,
    revision: str | None = None,
    on_progress: Callable[[dict[str, Any]], Any] | None = None,
) -> str: ...
def recommend(
    repo: str,
    *,
    revision: str | None = None,
    free_ram_gb: float | None = None,
    nvme_mib_s: float | None = None,
    has_draft: bool = True,
    cache_dir: str | PathLike[str] | None = None,
    use_gpu: bool = True,
) -> list[dict[str, Any]]: ...
def check(
    model: str,
    *,
    draft_model: str | None = None,
    quant: str | None = None,
    revision: str | None = None,
    cache_dir: str | PathLike[str] | None = None,
    use_gpu: bool = True,
    allow_slow: bool = False,
) -> dict[str, Any]: ...
def start_server(
    model: str,
    *,
    host: str = "127.0.0.1",
    port: int = 8080,
    api_key: str | None = None,
    draft_model: str | None = None,
    quant: str | None = None,
    revision: str | None = None,
    cache_dir: str | PathLike[str] | None = None,
    vram_budget_mb: int | str | None = "auto",
    resident_layers: int | None = None,
    ctx_size: int = 4096,
    batch_size: int = 512,
    threads: int | None = None,
    kv_cache_type: str = "q8_0",
    use_gpu: bool = True,
    prefer_direct_io: bool = True,
    cpu_embeddings: bool = False,
    cpu_head: bool = False,
    stream_head: bool = False,
    moe_active_streaming: bool = True,
    idle_timeout_secs: int = 60,
    max_draft_tokens: int = 5,
    verbose: bool = False,
    on_progress: Callable[[dict[str, Any]], Any] | None = None,
) -> ServerHandle: ...
def json_schema_to_gbnf(schema: Any) -> str: ...
def json_object_gbnf() -> str: ...
def compile_response_format(
    response_format: Any, allow_raw_schema: bool = False
) -> tuple[str | None, bool]: ...
def validate_sample_params(
    temperature: float,
    top_p: float,
    min_p: float,
    repeat_penalty: float,
    frequency_penalty: float,
    presence_penalty: float,
    penalty_last_n: int = 64,
) -> None: ...
def normalize_open_budget_and_kv(
    vram_budget_mb: Any, kv_cache_type: str
) -> tuple[bool, int, str]: ...
def parse_assistant_output(raw: str) -> dict[str, Any]: ...
def format_tools_system_prompt(tools: Sequence[Mapping[str, Any]]) -> str: ...
