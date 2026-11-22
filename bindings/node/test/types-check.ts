import {
  AutoModel,
  OpenAI,
  StreamTagParser,
  check,
  inspectGguf,
  jsonObjectGbnf,
  jsonSchemaToGbnf,
  parseAssistantOutput,
  probeGpuMemory,
  pull,
  recommend,
  version,
  type AbortSignalLike,
  type CheckReport,
  type EngineStats,
  type GgufInfo,
  type QuantRecommendation,
} from "../index.js";

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
