from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from os import PathLike
from types import TracebackType
from typing import Any, Literal, TypedDict, overload

from .openai import AsyncOpenAI as AsyncOpenAI, OpenAI as OpenAI

__version__: str

class ProgressEventDict(TypedDict):
    repo_id: str
    filename: str
    shard_index: int
    shard_count: int
    downloaded_bytes: int
    total_bytes: int | None
    mib_per_s: float
    cached: bool
    done: bool

class IoStatsDict(TypedDict):
    bytes_read: int
    span_reads: int
    pread_us: int
    cache_hits: int
    cache_misses: int
    cache_bytes_served: int
    direct_io_active: bool

class EngineStatsDict(TypedDict):
    loaded: bool
    isolated_worker: bool
    worker_pid: int | None
    backend_name: str
    cached_kv_tokens: int
    model_path: str
    draft_model_path: str | None
    source_uri: str | None
    resolved_repo: str | None
    selected_quant: str | None
    n_ctx: int
    vocab_size: int
    n_layers: int
    resident_layers: int
    streamed_layers: int
    resident_bytes: int
    cpu_resident_bytes: int
    slot_bytes: int
    expert_slot_bytes: int
    total_layer_loads: int
    moe_layers: int
    moe_experts_loaded: int
    moe_experts_total: int
    head_streamed: bool
    moe_active_streaming: bool
    is_uma_zero_copy: bool
    is_unified_memory: bool
    using_gpu: bool
    can_speculate: bool
    io: IoStatsDict

class SpeculationStatsDict(TypedDict):
    sweeps: int
    drafted_tokens: int
    accepted_tokens: int

class GenerateSummaryDict(TypedDict):
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    speculation: SpeculationStatsDict

class ToolCallFunctionDict(TypedDict):
    name: str
    arguments: str

class ToolCallDict(TypedDict):
    id: str
    type: str
    function: ToolCallFunctionDict

class ParsedAssistantOutputDict(TypedDict):
    content: str
    reasoning_content: str | None
    tool_calls: list[ToolCallDict]

class QuantRecommendationDict(TypedDict):
    resolved_repo: str
    quant: str
    primary_file: str
    total_bytes: int
    shard_count: int
    hf_uri: str
    verdict: str
    pinned_pct: int
    est_load_s: float
    est_tps_low: float
    est_tps_high: float
    recommended: bool
    score: int

class CheckReportDict(TypedDict):
    ok: bool
    verdict: str
    summary: str
    model_path: str
    draft_model_path: str | None
    architecture: str
    n_layers: int
    context_length: int | None
    total_file_bytes: int
    storage_mib_s: float
    direct_io: bool
    gpu_backend: str | None
    gpu_free_bytes: int | None
    gpu_total_bytes: int | None
    fits_in_vram: bool
    est_tps_low: float
    est_tps_high: float

class GgufInfoDict(TypedDict):
    architecture: str
    name: str | None
    n_layers: int
    context_length: int | None
    total_file_bytes: int
    shard_count: int
    tensor_count: int

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
    ) -> None: ...

class TokenStream(Iterator[str], AsyncIterator[str]):
    def __iter__(self) -> TokenStream: ...
    def __next__(self) -> str: ...
    def __enter__(self) -> TokenStream: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc_val: BaseException | None = None,
        exc_tb: TracebackType | None = None,
    ) -> None: ...
    def close(self) -> None: ...
    @property
    def summary(self) -> GenerateSummaryDict | None: ...
    def __aiter__(self) -> TokenStream: ...
    async def __anext__(self) -> str: ...
    async def __aenter__(self) -> TokenStream: ...
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc_val: BaseException | None = None,
        exc_tb: TracebackType | None = None,
    ) -> None: ...
    async def aclose(self) -> None: ...

class AutoModel:
    @classmethod
    def from_pretrained(
        cls,
        model: str,
        *,
        draft_model: str | Literal["auto"] | None = None,
        quant: str | None = None,
        revision: str | None = None,
        cache_dir: str | PathLike[str] | None = None,
        vram_budget_mb: int | Literal["auto"] | None = "auto",
        resident_layers: int | None = None,
        ctx_size: int = 4096,
        batch_size: int = 512,
        threads: int | None = None,
        kv_cache_type: Literal["f16", "q8_0", "q4_0"] = "q8_0",
        use_gpu: bool = True,
        prefer_direct_io: bool = True,
        cpu_embeddings: bool = False,
        cpu_head: bool = False,
        stream_head: bool = False,
        moe_active_streaming: bool = True,
        max_draft_tokens: int = 5,
        verbose: bool = False,
        isolated_worker: bool = False,
        on_progress: Callable[[dict[str, Any]], Any] | None = None,
    ) -> AutoModel: ...
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
    ) -> str: ...
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
    ) -> TokenStream: ...
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
        stream: Literal[False] = False,
        add_generation_prompt: bool = True,
        ngram_speculation: bool = True,
        max_draft_tokens: int | None = None,
        tools: Sequence[Mapping[str, Any]] | None = None,
        grammar: str | None = None,
        json_schema: Any = None,
        parse_output: Literal[False] = False,
    ) -> str: ...
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
        parse_output: Literal[True],
    ) -> ParsedAssistantOutputDict: ...
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
        parse_output: Literal[False] = False,
    ) -> TokenStream: ...
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
    ) -> str | ParsedAssistantOutputDict | TokenStream: ...
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
    def save_state(self, path: str | PathLike[str]) -> int: ...
    def load_state(self, path: str | PathLike[str]) -> int: ...
    def stats(self) -> EngineStatsDict: ...
    @property
    def last_summary(self) -> GenerateSummaryDict | None: ...
    @property
    def is_loaded(self) -> bool: ...
    @property
    def isolated_worker(self) -> bool: ...
    @property
    def worker_pid(self) -> int | None: ...
    @property
    def model_path(self) -> str | None: ...
    def unload(self) -> None: ...
    def __enter__(self) -> AutoModel: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc_val: BaseException | None = None,
        exc_tb: TracebackType | None = None,
    ) -> None: ...
    async def __aenter__(self) -> AutoModel: ...
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc_val: BaseException | None = None,
        exc_tb: TracebackType | None = None,
    ) -> None: ...

def configured_backend_dirs() -> list[str]: ...
def probe_gpu_memory(use_gpu: bool = True) -> tuple[int, int] | None: ...
def probe_gpu_device(use_gpu: bool = True) -> tuple[int, int, str] | None: ...
def inspect_gguf(path: str | PathLike[str]) -> GgufInfoDict: ...
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
) -> list[QuantRecommendationDict]: ...
def check(
    model: str,
    *,
    draft_model: str | Literal["auto"] | None = None,
    quant: str | None = None,
    revision: str | None = None,
    cache_dir: str | PathLike[str] | None = None,
    use_gpu: bool = True,
    allow_slow: bool = False,
) -> CheckReportDict: ...
def start_server(
    model: str,
    *,
    host: str = "127.0.0.1",
    port: int = 8080,
    api_key: str | None = None,
    draft_model: str | Literal["auto"] | None = None,
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
def serve(
    model: str,
    *,
    host: str = "127.0.0.1",
    port: int = 8080,
    api_key: str | None = None,
    draft_model: str | Literal["auto"] | None = None,
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
) -> None: ...
def json_schema_to_gbnf(schema: Any) -> str: ...
def json_object_gbnf() -> str: ...
def parse_assistant_output(raw: str) -> ParsedAssistantOutputDict: ...
def format_tools_system_prompt(tools: Sequence[Mapping[str, Any]]) -> str: ...
