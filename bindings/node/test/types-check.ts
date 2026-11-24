import {
  type AbortSignalLike,
  AutoModel,
  type ChatCompletionStream,
  type ChatMessageInput,
  type CheckReport,
  check,
  type EngineStats,
  type GgufInfo,
  inspectGguf,
  jsonObjectGbnf,
  jsonSchemaToGbnf,
  OpenAI,
  parseAssistantOutput,
  probeGpuMemory,
  pull,
  type QuantRecommendation,
  recommend,
  StreamTagParser,
  version,
} from "../index.js";
import OpenAISubpathDefault, {
  ChatCompletionStream as SubpathChatCompletionStream,
  CompletionStream as SubpathCompletionStream,
  OpenAI as SubpathOpenAI,
} from "../openai.js";

export async function verifyTypesOnly(signal: AbortSignalLike): Promise<void> {
  const v: string = version;
  void v;
  const mem = await probeGpuMemory(false);
  void mem;

  const g1: string = jsonObjectGbnf();
  const g2: string = jsonSchemaToGbnf({ type: "object" });
  void g1;
  void g2;

  const info: GgufInfo = await inspectGguf("/tmp/model.gguf");
  const report: CheckReport = await check("/tmp/model.gguf", { useGpu: false });
  const recs: QuantRecommendation[] = await recommend("Qwen/Qwen2.5-14B-Instruct");
  const local: string = await pull("hf://Org/Repo:file.gguf", { cacheDir: "/tmp" });
  void info;
  void report;
  void recs;
  void local;

  const parser = new StreamTagParser();
  const deltas = parser.push("<think>hi</think>");
  const parsed = parseAssistantOutput("hello");
  void deltas;
  void parsed;

  const validMessages: ChatMessageInput[] = [
    { role: "user", content: "" },
    {
      role: "assistant",
      tool_calls: [{ function: { name: "ping", arguments: "{}" } }],
    },
    {
      role: "assistant",
      content: "calling",
      tool_calls: [{ function: { name: "ping", arguments: { x: 1 } } }],
    },
  ];

  // @ts-expect-error ChatMessageInput requires at least one of content or tool_calls
  const invalidMsg: ChatMessageInput = { role: "user" };
  void invalidMsg;

  await using model = await AutoModel.fromPretrained("/tmp/model.gguf", {
    isolatedWorker: true,
    vramBudgetMb: "auto",
  });
  const text: string = await model.generate("hello", { maxTokens: 16, signal });
  void text;
  const stream = await model.generate("hello", { stream: true, signal });
  for await (const chunk of stream) {
    const s: string = chunk;
    void s;
  }
  const st: EngineStats = await model.stats();
  void st;

  await using client = new OpenAI(model);
  await using subClient = new OpenAISubpathDefault(model);
  const subNamed = SubpathOpenAI;
  void subClient;
  void subNamed;
  void SubpathChatCompletionStream;
  void SubpathCompletionStream;

  const chatStream: ChatCompletionStream = await client.chat.completions.create({
    messages: validMessages,
    stream: true,
  });
  await chatStream.close();

  const chatResp = await client.chat.completions.parse<{ ok: boolean }>({
    messages: [{ role: "user", content: "hi" }],
    response_format: {
      toJSONSchema: () => ({ type: "object", properties: { ok: { type: "boolean" } } }),
      parse: (val) => val as { ok: boolean },
    },
  });
  const okFlag: boolean | undefined = chatResp.choices[0]?.message.parsed?.ok;
  void okFlag;
}
