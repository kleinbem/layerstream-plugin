from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from os import PathLike
from types import TracebackType
from typing import Literal, TypedDict, overload

__version__: str

class IoStatsDict(TypedDict):
    bytes_read: int
    span_reads: int
    pread_us: int
    cache_hits: int
    cache_misses: int
    cache_bytes_served: int
    direct_io_active: bool

class EngineStatsDict(TypedDict, total=False):
    loaded: bool
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

class AutoModel:
    @classmethod
    def from_pretrained(
        cls,
        model: str,
        *,
        draft_model: str | None = None,
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
    ) -> AutoModel: ...

    @overload
    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 40,
        seed: int = 42,
        stop: str | Sequence[str] | None = None,
        stream: Literal[False] = False,
        ngram_speculation: bool = True,
        max_draft_tokens: int | None = None,
    ) -> str: ...

    @overload
    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 40,
        seed: int = 42,
        stop: str | Sequence[str] | None = None,
        stream: Literal[True],
        ngram_speculation: bool = True,
        max_draft_tokens: int | None = None,
    ) -> TokenStream: ...

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 40,
        seed: int = 42,
        stop: str | Sequence[str] | None = None,
        stream: bool = False,
        ngram_speculation: bool = True,
        max_draft_tokens: int | None = None,
    ) -> str | TokenStream: ...

    @overload
    def chat(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 40,
        seed: int = 42,
        stop: str | Sequence[str] | None = None,
        stream: Literal[False] = False,
        add_generation_prompt: bool = True,
        ngram_speculation: bool = True,
        max_draft_tokens: int | None = None,
    ) -> str: ...

    @overload
    def chat(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 40,
        seed: int = 42,
        stop: str | Sequence[str] | None = None,
        stream: Literal[True],
        add_generation_prompt: bool = True,
        ngram_speculation: bool = True,
        max_draft_tokens: int | None = None,
    ) -> TokenStream: ...

    def chat(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 40,
        seed: int = 42,
        stop: str | Sequence[str] | None = None,
        stream: bool = False,
        add_generation_prompt: bool = True,
        ngram_speculation: bool = True,
        max_draft_tokens: int | None = None,
    ) -> str | TokenStream: ...

    def stats(self) -> EngineStatsDict: ...
    @property
    def last_summary(self) -> GenerateSummaryDict | None: ...
    @property
    def is_loaded(self) -> bool: ...
    @property
    def model_path(self) -> str | None: ...
    def unload(self) -> None: ...
    def __enter__(self) -> AutoModel: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc_val: BaseException | None = None,
        exc_tb: TracebackType | None = None,
    ) -> bool: ...

def probe_gpu_memory(use_gpu: bool = True) -> tuple[int, int] | None: ...
