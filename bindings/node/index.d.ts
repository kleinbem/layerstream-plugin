export declare const version: string;

/** Structural AbortSignal interface compatible with Node.js, DOM, Bun, Deno, and WinterCG runtimes without requiring `@types/node` or `lib: ["DOM"]`. */
export interface AbortSignalLike {
  readonly aborted: boolean;
  readonly reason?: unknown;
  addEventListener(type: "abort", listener: () => void, options?: { once?: boolean }): void;
  removeEventListener(type: "abort", listener: () => void): void;
}

export interface IoStats {
  bytesRead: number;
  spanReads: number;
  preadUs: number;
  cacheHits: number;
  cacheMisses: number;
  cacheBytesServed: number;
  directIoActive: boolean;
}

export interface EngineStats {
  loaded: boolean;
  isolatedWorker: boolean;
  workerPid?: number | null;
  backendName?: string | null;
  cachedKvTokens?: number | null;
  modelPath?: string | null;
  draftModelPath?: string | null;
  sourceUri?: string | null;
  resolvedRepo?: string | null;
  selectedQuant?: string | null;
  nCtx?: number | null;
  vocabSize?: number | null;
  nLayers?: number | null;
  residentLayers?: number | null;
  streamedLayers?: number | null;
  residentBytes?: number | null;
  cpuResidentBytes?: number | null;
  slotBytes?: number | null;
  expertSlotBytes?: number | null;
  totalLayerLoads?: number | null;
  moeLayers?: number | null;
  moeExpertsLoaded?: number | null;
  moeExpertsTotal?: number | null;
  headStreamed?: boolean | null;
  moeActiveStreaming?: boolean | null;
  isUmaZeroCopy?: boolean | null;
  isUnifiedMemory?: boolean | null;
  usingGpu?: boolean | null;
  canSpeculate?: boolean | null;
  io?: IoStats | null;
}

export interface SpeculationStats {
  sweeps: number;
  draftedTokens: number;
  acceptedTokens: number;
}

export interface GenerateSummary {
  promptTokens: number;
  completionTokens: number;
  finishReason: string;
  speculation: SpeculationStats;
}

export interface GpuMemoryInfo {
  freeBytes: number;
  totalBytes: number;
}

export interface GgufInfo {
  architecture: string;
  name?: string | null;
  nLayers: number;
  contextLength?: number | null;
  totalFileBytes: number;
  shardCount: number;
  tensorCount: number;
}

export interface QuantRecommendation {
  resolvedRepo: string;
  quant: string;
  primaryFile: string;
  totalBytes: number;
  shardCount: number;
  hfUri: string;
  verdict: string;
  pinnedPct: number;
  estLoadS: number;
  estTpsLow: number;
  estTpsHigh: number;
  recommended: boolean;
  score: number;
}

export interface CheckReport {
  ok: boolean;
  verdict: string;
  summary: string;
  modelPath: string;
  draftModelPath?: string | null;
  architecture: string;
  nLayers: number;
  contextLength?: number | null;
  totalFileBytes: number;
  storageMibS: number;
  directIo: boolean;
  gpuFreeBytes?: number | null;
  gpuTotalBytes?: number | null;
  fitsInVram: boolean;
  estTpsLow: number;
  estTpsHigh: number;
}

export interface ToolCallFunction {
  name: string;
  arguments: string;
}

export interface ToolCall {
  id: string;
  type: string;
  function: ToolCallFunction;
}

export interface ParsedAssistantOutput {
  content: string;
  reasoningContent?: string | null;
  toolCalls: ToolCall[];
}

export interface StreamDelta {
  type: "content" | "reasoning" | "tool_call" | string;
  text?: string | null;
  index?: number | null;
  id?: string | null;
  name?: string | null;
  arguments?: string | null;
}

export declare class StreamTagParser {
  constructor();
  readonly hasToolCalls: boolean;
  push(piece: string): StreamDelta[];
  flush(): StreamDelta[];
}

export interface FromPretrainedOptions {
  draftModel?: string | null;
  quant?: string | null;
  revision?: string | null;
  cacheDir?: string | null;
  vramBudgetMb?: number | "auto" | null;
  residentLayers?: number | null;
  ctxSize?: number;
  batchSize?: number;
  threads?: number | null;
  kvCacheType?: "f16" | "q8_0" | "q4_0";
  useGpu?: boolean;
  preferDirectIo?: boolean;
  cpuEmbeddings?: boolean;
  cpuHead?: boolean;
  streamHead?: boolean;
  moeActiveStreaming?: boolean;
  maxDraftTokens?: number;
  verbose?: boolean;
  isolatedWorker?: boolean;
}

export interface SchemaWithToJsonSchema<T = unknown> {
  toJSONSchema(): Record<string, unknown>;
  parse?(data: unknown): T;
}

export interface GenerateOptions {
  maxTokens?: number;
  temperature?: number;
  topP?: number;
  minP?: number;
  topK?: number;
  repeatPenalty?: number;
  frequencyPenalty?: number;
  presencePenalty?: number;
  penaltyLastN?: number;
  seed?: number;
  stop?: string | string[] | null;
  stream?: boolean;
  ngramSpeculation?: boolean;
  maxDraftTokens?: number | null;
  grammar?: string | null;
  jsonSchema?: Record<string, unknown> | string | boolean | SchemaWithToJsonSchema | null;
  signal?: AbortSignalLike;
}

export type ChatMessageContent = string | Array<string | { type?: string; text: string }> | null;

export interface ChatMessageToolCallInput {
  id?: string;
  type?: string;
  function: { name: string; arguments: string | Record<string, unknown> };
}

interface ChatMessageBase {
  role: "system" | "user" | "assistant" | "tool" | string;
  name?: string | null;
  tool_call_id?: string | null;
}

export type ChatMessageInput =
  | (ChatMessageBase & {
      content: ChatMessageContent;
      tool_calls?: ChatMessageToolCallInput[] | null;
    })
  | (ChatMessageBase & {
      content?: ChatMessageContent;
      tool_calls: ChatMessageToolCallInput[] | null;
    });

export interface ChatOptions extends GenerateOptions {
  addGenerationPrompt?: boolean;
  tools?: Array<Record<string, unknown>> | null;
  parseOutput?: boolean;
}

export interface TokenizeOptions {
  addSpecial?: boolean;
  parseSpecial?: boolean;
}

export interface ApplyChatTemplateOptions {
  addGenerationPrompt?: boolean;
  tools?: Array<Record<string, unknown>> | null;
}

export declare class TokenStream implements AsyncIterable<string>, AsyncDisposable {
  readonly summary: GenerateSummary | null;
  cancel(): void;
  close(): Promise<void>;
  [Symbol.asyncDispose](): Promise<void>;
  [Symbol.asyncIterator](): AsyncIterator<string>;
}

export declare class AutoModel implements AsyncDisposable {
  static fromPretrained(model: string, opts?: FromPretrainedOptions): Promise<AutoModel>;

  generate(prompt: string, opts: GenerateOptions & { stream: true }): Promise<TokenStream>;
  generate(prompt: string, opts?: GenerateOptions & { stream?: false }): Promise<string>;

  chat(
    messages: ChatMessageInput[],
    opts: ChatOptions & { stream: true; parseOutput?: false },
  ): Promise<TokenStream>;
  chat(
    messages: ChatMessageInput[],
    opts: ChatOptions & { stream?: false; parseOutput: true },
  ): Promise<ParsedAssistantOutput>;
  chat(
    messages: ChatMessageInput[],
    opts?: ChatOptions & { stream?: false; parseOutput?: false },
  ): Promise<string>;

  tokenize(text: string, opts?: TokenizeOptions): Promise<number[]>;
  detokenize(tokens: number[]): Promise<string>;
  applyChatTemplate(messages: ChatMessageInput[], opts?: ApplyChatTemplateOptions): Promise<string>;
  kvClear(): Promise<void>;
  kvTrim(posMin: number): Promise<boolean>;
  saveState(path: string): Promise<number>;
  loadState(path: string): Promise<number>;
  stats(): Promise<EngineStats>;

  readonly isLoaded: boolean;
  readonly isolatedWorker: boolean;
  readonly workerPid: number | null;
  readonly modelPath: string | null;
  readonly lastSummary: GenerateSummary | null;

  unload(): Promise<void>;
  [Symbol.asyncDispose](): Promise<void>;
}

export interface RecommendOptions {
  revision?: string;
  freeRamGb?: number;
  nvmeMibS?: number;
  hasDraft?: boolean;
  cacheDir?: string;
  useGpu?: boolean;
}

export interface CheckOptions {
  draftModel?: string;
  quant?: string;
  revision?: string;
  cacheDir?: string;
  useGpu?: boolean;
  allowSlow?: boolean;
}

export declare function probeGpuMemory(useGpu?: boolean): Promise<GpuMemoryInfo | null>;
export declare function inspectGguf(path: string): Promise<GgufInfo>;
export declare function pull(spec: string, opts?: { cacheDir?: string }): Promise<string>;
export declare function recommend(
  repo: string,
  opts?: RecommendOptions,
): Promise<QuantRecommendation[]>;
export declare function check(model: string, opts?: CheckOptions): Promise<CheckReport>;
export declare function jsonObjectGbnf(): string;
export declare function jsonSchemaToGbnf(
  schema: Record<string, unknown> | string | boolean | SchemaWithToJsonSchema,
): string;
export declare function parseAssistantOutput(raw: string): ParsedAssistantOutput;
export declare function formatToolsSystemPrompt(tools: Array<Record<string, unknown>>): string;

// --- In-process OpenAI-compatible SDK types ---

export interface CompletionUsage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

export interface OpenAIChatCompletionMessage<TParsed = unknown> {
  role: "assistant";
  content: string | null;
  reasoning_content: string | null;
  tool_calls: ToolCall[] | null;
  parsed: TParsed | null;
}

export interface OpenAIChatCompletion<TParsed = unknown> {
  id: string;
  object: "chat.completion";
  created: number;
  model: string;
  choices: Array<{
    index: number;
    message: OpenAIChatCompletionMessage<TParsed>;
    finish_reason: "stop" | "length" | "tool_calls" | string;
    logprobs: null;
  }>;
  usage: CompletionUsage | null;
}

export interface OpenAIChatCompletionChunk {
  id: string;
  object: "chat.completion.chunk";
  created: number;
  model: string;
  choices: Array<{
    index: number;
    delta: {
      role?: "assistant";
      content?: string;
      reasoning_content?: string;
      tool_calls?: Array<{
        index: number;
        id: string;
        type: "function";
        function: { name: string; arguments: string };
      }>;
    };
    finish_reason: string | null;
    logprobs: null;
  }>;
  usage: CompletionUsage | null;
}

export interface OpenAICompletion {
  id: string;
  object: "text_completion";
  created: number;
  model: string;
  choices: Array<{
    text: string;
    index: number;
    finish_reason: string | null;
    logprobs: null;
  }>;
  usage: CompletionUsage | null;
}

export interface ChatCompletionStream
  extends AsyncIterable<OpenAIChatCompletionChunk>,
    AsyncDisposable {
  close(): Promise<void>;
  [Symbol.asyncDispose](): Promise<void>;
  [Symbol.asyncIterator](): AsyncIterator<OpenAIChatCompletionChunk>;
}

export interface CompletionStream extends AsyncIterable<OpenAICompletion>, AsyncDisposable {
  close(): Promise<void>;
  [Symbol.asyncDispose](): Promise<void>;
  [Symbol.asyncIterator](): AsyncIterator<OpenAICompletion>;
}

export type ResponseFormatSpec<T = unknown> =
  | { type: "text" }
  | { type: "json_object" }
  | { type: "gbnf"; grammar: string }
  | {
      type: "json_schema";
      json_schema?: {
        name?: string;
        schema: Record<string, unknown> | boolean;
        parse?: (v: unknown) => T;
      };
      schema?: Record<string, unknown> | boolean;
      parse?: (v: unknown) => T;
    }
  | SchemaWithToJsonSchema<T>
  | Record<string, unknown>
  | boolean;

export interface OpenAIChatCreateParams {
  model?: string;
  messages: ChatMessageInput[];
  max_tokens?: number;
  max_completion_tokens?: number;
  temperature?: number;
  top_p?: number;
  min_p?: number;
  top_k?: number;
  repeat_penalty?: number;
  frequency_penalty?: number;
  presence_penalty?: number;
  penalty_last_n?: number;
  seed?: number;
  stop?: string | string[] | null;
  stream?: boolean;
  tools?: Array<Record<string, unknown>> | null;
  response_format?: ResponseFormatSpec | null;
  ngram_speculation?: boolean;
  max_draft_tokens?: number;
  signal?: AbortSignalLike;
}

export interface OpenAICompletionsCreateParams {
  model?: string;
  prompt: string;
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
  min_p?: number;
  top_k?: number;
  repeat_penalty?: number;
  frequency_penalty?: number;
  presence_penalty?: number;
  penalty_last_n?: number;
  seed?: number;
  stop?: string | string[] | null;
  stream?: boolean;
  grammar?: string;
  json_schema?: Record<string, unknown> | string;
  ngram_speculation?: boolean;
  max_draft_tokens?: number;
  signal?: AbortSignalLike;
}

export interface OpenAIClientOptions extends FromPretrainedOptions {
  model?: string | AutoModel | null;
  apiKey?: string;
  baseURL?: string;
}

export declare class OpenAI implements AsyncDisposable {
  constructor(
    modelOrOptions?: string | AutoModel | OpenAIClientOptions,
    extraOptions?: FromPretrainedOptions,
  );

  readonly chat: {
    completions: {
      create(
        params: OpenAIChatCreateParams & { stream: true },
        callOpts?: { signal?: AbortSignalLike },
      ): Promise<ChatCompletionStream>;
      create(
        params: OpenAIChatCreateParams & { stream?: false },
        callOpts?: { signal?: AbortSignalLike },
      ): Promise<OpenAIChatCompletion>;
      parse<T = Record<string, unknown>>(
        params: Omit<OpenAIChatCreateParams, "stream" | "response_format"> & {
          response_format: ResponseFormatSpec<T>;
        },
        callOpts?: { signal?: AbortSignalLike },
      ): Promise<OpenAIChatCompletion<T>>;
    };
  };

  readonly completions: {
    create(
      params: OpenAICompletionsCreateParams & { stream: true },
      callOpts?: { signal?: AbortSignalLike },
    ): Promise<CompletionStream>;
    create(
      params: OpenAICompletionsCreateParams & { stream?: false },
      callOpts?: { signal?: AbortSignalLike },
    ): Promise<OpenAICompletion>;
  };

  readonly models: {
    list(): Promise<{
      object: "list";
      data: Array<{ id: string; object: "model"; created: number; owned_by: string }>;
    }>;
  };

  unload(): Promise<void>;
  close(): Promise<void>;
  [Symbol.asyncDispose](): Promise<void>;
}
