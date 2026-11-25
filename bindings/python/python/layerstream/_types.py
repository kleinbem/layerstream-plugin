"""Public TypedDict schemas, PEP 692 kwargs contracts, and exception base for `layerstream`."""

from collections.abc import Callable, Sequence
from os import PathLike
from typing import Any, Literal, TypedDict

type JsonDict = dict[str, Any]


class LayerStreamError(RuntimeError):
    """Base exception for `layerstream` runtime errors (backwards-compatible with `RuntimeError`)."""


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


class SamplingOptions(TypedDict, total=False):
    """PEP 692 shared sampling & speculation keyword arguments."""

    temperature: float
    top_p: float
    min_p: float
    top_k: int
    repeat_penalty: float
    frequency_penalty: float
    presence_penalty: float
    penalty_last_n: int
    seed: int
    stop: str | Sequence[str] | None
    ngram_speculation: bool
    max_draft_tokens: int | None


class ModelLoadOptions(TypedDict, total=False):
    """PEP 692 shared model-loading keyword arguments."""

    draft_model: str | Literal["auto"] | None
    quant: str | None
    revision: str | None
    cache_dir: str | PathLike[str] | None
    vram_budget_mb: int | Literal["auto"] | None
    resident_layers: int | None
    ctx_size: int
    batch_size: int
    threads: int | None
    kv_cache_type: Literal["f16", "q8_0", "q4_0"]
    use_gpu: bool
    prefer_direct_io: bool
    cpu_embeddings: bool
    cpu_head: bool
    stream_head: bool
    moe_active_streaming: bool
    max_draft_tokens: int
    verbose: bool
    on_progress: Callable[[dict[str, Any]], Any] | None
