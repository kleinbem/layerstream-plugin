from collections.abc import Callable, Iterator, Mapping, Sequence
from os import PathLike
from types import TracebackType
from typing import Any, Literal, Unpack, overload

from ._types import SamplingOptions

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
    def tokenize(
        self, text: str, *, add_special: bool = False, parse_special: bool = False
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
    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        stream: bool = False,
        grammar: str | None = None,
        json_schema: Any = None,
        return_summary: bool = False,
        **sampling: Unpack[SamplingOptions],
    ) -> Any: ...
    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        max_tokens: int = 256,
        stream: bool = False,
        add_generation_prompt: bool = True,
        tools: Sequence[Mapping[str, Any]] | None = None,
        grammar: str | None = None,
        json_schema: Any = None,
        parse_output: bool = False,
        return_summary: bool = False,
        **sampling: Unpack[SamplingOptions],
    ) -> Any: ...
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
        stream: Literal[True],
        grammar: str | None = None,
        json_schema: Any = None,
        return_summary: Literal[False] = False,
        **sampling: Unpack[SamplingOptions],
    ) -> TokenStream: ...
    @overload
    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        stream: Literal[False] = False,
        grammar: str | None = None,
        json_schema: Any = None,
        return_summary: Literal[True],
        **sampling: Unpack[SamplingOptions],
    ) -> tuple[str, dict[str, Any]]: ...
    @overload
    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        stream: bool = False,
        grammar: str | None = None,
        json_schema: Any = None,
        return_summary: bool = False,
        **sampling: Unpack[SamplingOptions],
    ) -> str | TokenStream | tuple[str, dict[str, Any]]: ...
    @overload
    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        max_tokens: int = 256,
        stream: Literal[True],
        add_generation_prompt: bool = True,
        tools: Sequence[Mapping[str, Any]] | None = None,
        grammar: str | None = None,
        json_schema: Any = None,
        parse_output: bool = False,
        return_summary: Literal[False] = False,
        **sampling: Unpack[SamplingOptions],
    ) -> TokenStream: ...
    @overload
    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        max_tokens: int = 256,
        stream: Literal[False] = False,
        add_generation_prompt: bool = True,
        tools: Sequence[Mapping[str, Any]] | None = None,
        grammar: str | None = None,
        json_schema: Any = None,
        parse_output: bool = False,
        return_summary: Literal[True],
        **sampling: Unpack[SamplingOptions],
    ) -> tuple[str | dict[str, Any], dict[str, Any]]: ...
    @overload
    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        max_tokens: int = 256,
        stream: bool = False,
        add_generation_prompt: bool = True,
        tools: Sequence[Mapping[str, Any]] | None = None,
        grammar: str | None = None,
        json_schema: Any = None,
        parse_output: bool = False,
        return_summary: bool = False,
        **sampling: Unpack[SamplingOptions],
    ) -> str | dict[str, Any] | TokenStream | tuple[str | dict[str, Any], dict[str, Any]]: ...
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
def parse_assistant_output(raw: str) -> dict[str, Any]: ...
def format_tools_system_prompt(tools: Sequence[Mapping[str, Any]]) -> str: ...
