import type {
  AbortSignalLike,
  AutoModel,
  ChatMessageContent,
  ChatMessageInput,
  ChatMessageToolCallInput,
  CompletionUsage,
  FromPretrainedOptions,
  OpenAIChatCompletion,
  OpenAIChatCompletionChunk,
  OpenAIChatCompletionMessage,
  OpenAIChatCreateParams,
  OpenAIClientOptions,
  OpenAICompletion,
  OpenAICompletionsCreateParams,
  ResponseFormatSpec,
  SchemaWithToJsonSchema,
  ToolCall,
  ToolCallFunction,
} from "./index.js";

export type {
  AbortSignalLike,
  ChatMessageContent,
  ChatMessageInput,
  ChatMessageToolCallInput,
  CompletionUsage,
  FromPretrainedOptions,
  OpenAIChatCompletion,
  OpenAIChatCompletionChunk,
  OpenAIChatCompletionMessage,
  OpenAIChatCreateParams,
  OpenAIClientOptions,
  OpenAICompletion,
  OpenAICompletionsCreateParams,
  ResponseFormatSpec,
  SchemaWithToJsonSchema,
  ToolCall,
  ToolCallFunction,
};

export declare class ChatCompletionStream
  implements AsyncIterable<OpenAIChatCompletionChunk>, AsyncDisposable
{
  close(): Promise<void>;
  [Symbol.asyncDispose](): Promise<void>;
  [Symbol.asyncIterator](): AsyncIterator<OpenAIChatCompletionChunk>;
}

export declare class CompletionStream
  implements AsyncIterable<OpenAICompletion>, AsyncDisposable
{
  close(): Promise<void>;
  [Symbol.asyncDispose](): Promise<void>;
  [Symbol.asyncIterator](): AsyncIterator<OpenAICompletion>;
}

export declare class OpenAI implements AsyncDisposable {
  constructor(
    modelOrOptions?: string | AutoModel | OpenAIClientOptions,
    extraOptions?: FromPretrainedOptions
  );

  readonly chat: {
    completions: {
      create(
        params: OpenAIChatCreateParams & { stream: true },
        callOpts?: { signal?: AbortSignalLike }
      ): Promise<ChatCompletionStream>;
      create(
        params: OpenAIChatCreateParams & { stream?: false },
        callOpts?: { signal?: AbortSignalLike }
      ): Promise<OpenAIChatCompletion>;
      parse<T = Record<string, unknown>>(
        params: Omit<OpenAIChatCreateParams, "stream" | "response_format"> & {
          response_format: ResponseFormatSpec<T>;
        },
        callOpts?: { signal?: AbortSignalLike }
      ): Promise<OpenAIChatCompletion<T>>;
    };
  };

  readonly completions: {
    create(
      params: OpenAICompletionsCreateParams & { stream: true },
      callOpts?: { signal?: AbortSignalLike }
    ): Promise<CompletionStream>;
    create(
      params: OpenAICompletionsCreateParams & { stream?: false },
      callOpts?: { signal?: AbortSignalLike }
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

export default OpenAI;
