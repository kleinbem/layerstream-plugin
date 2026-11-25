//! OpenAI response/chunk assembly and option/message normalization in Rust for the Node.js SDK.

use layerstream_core::{
    normalize_kv_cache_type as core_normalize_kv_cache_type,
    normalize_vram_budget_mb as core_normalize_vram_budget_mb,
    parse_assistant_output as core_parse_assistant_output,
};
use napi::bindgen_prelude::*;
use napi_derive::napi;
use serde_json::{Map, Value as JsonValue, json};

fn usage_value(summary: Option<&JsonValue>) -> JsonValue {
    let Some(obj) = summary.and_then(JsonValue::as_object) else {
        return JsonValue::Null;
    };
    let p = obj
        .get("promptTokens")
        .or_else(|| obj.get("prompt_tokens"))
        .and_then(JsonValue::as_u64)
        .unwrap_or(0);
    let c = obj
        .get("completionTokens")
        .or_else(|| obj.get("completion_tokens"))
        .and_then(JsonValue::as_u64)
        .unwrap_or(0);
    json!({
        "prompt_tokens": p,
        "completion_tokens": c,
        "total_tokens": p + c,
    })
}

fn summary_finish_reason(summary: Option<&JsonValue>) -> &str {
    summary
        .and_then(JsonValue::as_object)
        .and_then(|o| o.get("finishReason").or_else(|| o.get("finish_reason")))
        .and_then(JsonValue::as_str)
        .unwrap_or("stop")
}

#[napi]
pub fn build_chat_completion_json(
    req_id: String,
    created: i64,
    model_id: String,
    raw: String,
    structured_json: bool,
    summary_json: Option<String>,
) -> String {
    let summary_val: Option<JsonValue> = summary_json
        .as_deref()
        .and_then(|s| serde_json::from_str(s).ok());
    let raw_reason = summary_finish_reason(summary_val.as_ref());

    let (message, finish_reason) = if structured_json {
        let msg = json!({
            "role": "assistant",
            "content": raw,
            "reasoning_content": JsonValue::Null,
            "tool_calls": JsonValue::Null,
            "parsed": JsonValue::Null,
        });
        let reason = if raw_reason == "length" {
            "length"
        } else {
            "stop"
        };
        (msg, reason)
    } else {
        let parsed = core_parse_assistant_output(&raw);
        let tcs: Vec<JsonValue> = parsed
            .tool_calls
            .into_iter()
            .map(|tc| {
                json!({
                    "id": tc.id,
                    "type": tc.kind,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                })
            })
            .collect();
        let has_tcs = !tcs.is_empty();
        let content_val = if !parsed.content.is_empty() || !has_tcs {
            JsonValue::String(parsed.content)
        } else {
            JsonValue::Null
        };
        let reasoning_val = parsed
            .reasoning_content
            .map_or(JsonValue::Null, JsonValue::String);
        let tool_calls_val = if has_tcs {
            JsonValue::Array(tcs)
        } else {
            JsonValue::Null
        };
        let msg = json!({
            "role": "assistant",
            "content": content_val,
            "reasoning_content": reasoning_val,
            "tool_calls": tool_calls_val,
            "parsed": JsonValue::Null,
        });
        let reason = if has_tcs {
            "tool_calls"
        } else if raw_reason == "length" {
            "length"
        } else {
            "stop"
        };
        (msg, reason)
    };

    json!({
        "id": req_id,
        "object": "chat.completion",
        "created": created,
        "model": model_id,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason,
            "logprobs": JsonValue::Null,
        }],
        "usage": usage_value(summary_val.as_ref()),
    })
    .to_string()
}

#[napi]
pub fn build_text_completion_json(
    req_id: String,
    created: i64,
    model_id: String,
    text: String,
    summary_json: Option<String>,
) -> String {
    let summary_val: Option<JsonValue> = summary_json
        .as_deref()
        .and_then(|s| serde_json::from_str(s).ok());
    let reason = summary_finish_reason(summary_val.as_ref());
    json!({
        "id": req_id,
        "object": "text_completion",
        "created": created,
        "model": model_id,
        "choices": [{
            "text": text,
            "index": 0,
            "finish_reason": reason,
            "logprobs": JsonValue::Null,
        }],
        "usage": usage_value(summary_val.as_ref()),
    })
    .to_string()
}

#[napi]
pub fn build_chat_delta_chunks_json(
    req_id: String,
    created: i64,
    model_id: String,
    deltas_json: String,
) -> String {
    let Ok(JsonValue::Array(deltas)) = serde_json::from_str::<JsonValue>(&deltas_json) else {
        return "[]".to_string();
    };
    let mut chunks = Vec::with_capacity(deltas.len());
    for ev in deltas {
        let Some(obj) = ev.as_object() else {
            continue;
        };
        let delta = match obj.get("type").and_then(JsonValue::as_str) {
            Some("content") => {
                let text = obj.get("text").and_then(JsonValue::as_str).unwrap_or("");
                json!({ "content": text })
            }
            Some("reasoning") => {
                let text = obj.get("text").and_then(JsonValue::as_str).unwrap_or("");
                json!({ "reasoning_content": text })
            }
            Some("tool_call") => {
                let idx = obj.get("index").and_then(JsonValue::as_u64).unwrap_or(0);
                let id = obj
                    .get("id")
                    .and_then(JsonValue::as_str)
                    .unwrap_or("call_ls_1");
                let name = obj.get("name").and_then(JsonValue::as_str).unwrap_or("");
                let arguments = obj
                    .get("arguments")
                    .and_then(JsonValue::as_str)
                    .unwrap_or("{}");
                json!({
                    "tool_calls": [{
                        "index": idx,
                        "id": id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": arguments,
                        },
                    }]
                })
            }
            _ => continue,
        };
        chunks.push(json!({
            "id": req_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_id,
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": JsonValue::Null,
                "logprobs": JsonValue::Null,
            }],
            "usage": JsonValue::Null,
        }));
    }
    JsonValue::Array(chunks).to_string()
}

#[napi]
pub fn normalize_chat_messages_json(messages_json: String) -> Result<String> {
    let val: JsonValue = serde_json::from_str(&messages_json).map_err(|_| {
        Error::new(
            Status::InvalidArg,
            "TYPE_ERR:messages must be an array of { role, content } objects",
        )
    })?;
    let JsonValue::Array(items) = val else {
        return Err(Error::new(
            Status::InvalidArg,
            "TYPE_ERR:messages must be an array of { role, content } objects",
        ));
    };
    if items.is_empty() {
        return Err(Error::new(
            Status::InvalidArg,
            "RANGE_ERR:messages must not be empty",
        ));
    }

    let mut out = Vec::with_capacity(items.len());
    for item in items {
        let JsonValue::Object(obj) = item else {
            return Err(Error::new(
                Status::InvalidArg,
                "TYPE_ERR:each message must be an object",
            ));
        };
        let Some(role) = obj.get("role").and_then(JsonValue::as_str) else {
            return Err(Error::new(
                Status::InvalidArg,
                "TYPE_ERR:each message must contain a non-empty 'role' string",
            ));
        };
        if role.is_empty() {
            return Err(Error::new(
                Status::InvalidArg,
                "TYPE_ERR:each message must contain a non-empty 'role' string",
            ));
        }
        if !obj.contains_key("content") && !obj.contains_key("tool_calls") {
            return Err(Error::new(
                Status::InvalidArg,
                "TYPE_ERR:each message must contain 'content' or 'tool_calls'",
            ));
        }

        let mut clean = Map::new();
        clean.insert("role".to_string(), JsonValue::String(role.to_string()));
        clean.insert(
            "content".to_string(),
            obj.get("content").cloned().unwrap_or(JsonValue::Null),
        );
        if let Some(tcs) = obj.get("tool_calls")
            && !tcs.is_null()
        {
            if !tcs.is_array() {
                return Err(Error::new(
                    Status::InvalidArg,
                    "TYPE_ERR:message 'tool_calls' must be an array",
                ));
            }
            clean.insert("tool_calls".to_string(), tcs.clone());
        }
        if let Some(name) = obj.get("name")
            && !name.is_null()
        {
            let name_str = match name {
                JsonValue::String(s) => s.clone(),
                other => other.to_string(),
            };
            clean.insert("name".to_string(), JsonValue::String(name_str));
        }
        if let Some(tc_id) = obj.get("tool_call_id")
            && !tc_id.is_null()
        {
            let id_str = match tc_id {
                JsonValue::String(s) => s.clone(),
                other => other.to_string(),
            };
            clean.insert("tool_call_id".to_string(), JsonValue::String(id_str));
        }
        out.push(JsonValue::Object(clean));
    }
    Ok(JsonValue::Array(out).to_string())
}

fn type_err(msg: &str) -> Error {
    Error::new(Status::InvalidArg, format!("TYPE_ERR:{msg}"))
}

fn range_err(msg: impl std::fmt::Display) -> Error {
    Error::new(Status::InvalidArg, format!("RANGE_ERR:{msg}"))
}

fn pick<'a>(obj: &'a Map<String, JsonValue>, camel: &str, snake: &str) -> Option<&'a JsonValue> {
    obj.get(camel)
        .filter(|v| !v.is_null())
        .or_else(|| obj.get(snake).filter(|v| !v.is_null()))
}

fn pick_raw<'a>(
    obj: &'a Map<String, JsonValue>,
    camel: &str,
    snake: &str,
) -> Option<&'a JsonValue> {
    obj.get(camel).or_else(|| obj.get(snake))
}

#[napi]
pub fn normalize_open_options_json(opts_json: String) -> Result<String> {
    let val: JsonValue =
        serde_json::from_str(&opts_json).map_err(|_| type_err("options must be an object"))?;
    let JsonValue::Object(obj) = val else {
        return Err(type_err("options must be an object"));
    };

    let vram_raw = pick_raw(&obj, "vramBudgetMb", "vram_budget_mb");
    let vram_for_core = match vram_raw {
        None => JsonValue::String("auto".to_string()),
        Some(JsonValue::Bool(_)) => {
            return Err(type_err(
                "vramBudgetMb must be 'auto', null, or a non-negative integer MiB (not boolean)",
            ));
        }
        Some(JsonValue::String(s)) if s == "__NON_FINITE__" => {
            return Err(range_err("vramBudgetMb must be non-negative"));
        }
        Some(JsonValue::Number(n)) => {
            let Some(f) = n.as_f64() else {
                return Err(range_err("vramBudgetMb must be non-negative"));
            };
            if !f.is_finite() || f < 0.0 {
                return Err(range_err("vramBudgetMb must be non-negative"));
            }
            JsonValue::Number(n.clone())
        }
        Some(v @ (JsonValue::Null | JsonValue::String(_))) => v.clone(),
        Some(_) => {
            return Err(type_err(
                "vramBudgetMb must be 'auto', null, or a non-negative number MiB",
            ));
        }
    };

    let kv_raw = match pick(&obj, "kvCacheType", "kv_cache_type") {
        None => "q8_0".to_string(),
        Some(JsonValue::String(s)) => s.clone(),
        Some(other) => {
            return Err(range_err(format!(
                "Invalid kvCacheType '{other}' (expected 'f16', 'q8_0', or 'q4_0')"
            )));
        }
    };

    let (vram_budget_auto, vram_budget_bytes) =
        core_normalize_vram_budget_mb(&vram_for_core).map_err(|e| range_err(format!("{e:#}")))?;
    let kv_cache_type =
        core_normalize_kv_cache_type(&kv_raw).map_err(|e| range_err(format!("{e:#}")))?;

    let isolated_worker = pick(&obj, "isolatedWorker", "isolated_worker")
        .and_then(JsonValue::as_bool)
        .unwrap_or(false);

    let mut wire = Map::new();
    for (camel, snake) in [
        ("draftModel", "draft_model"),
        ("quant", "quant"),
        ("revision", "revision"),
        ("cacheDir", "cache_dir"),
        ("residentLayers", "resident_layers"),
        ("threads", "threads"),
    ] {
        if let Some(v) = pick(&obj, camel, snake) {
            wire.insert(camel.to_string(), v.clone());
        }
    }
    wire.insert(
        "vramBudgetAuto".to_string(),
        JsonValue::Bool(vram_budget_auto),
    );
    wire.insert("vramBudgetBytes".to_string(), json!(vram_budget_bytes));
    wire.insert(
        "ctxSize".to_string(),
        pick(&obj, "ctxSize", "ctx_size")
            .cloned()
            .unwrap_or_else(|| json!(4096)),
    );
    wire.insert(
        "batchSize".to_string(),
        pick(&obj, "batchSize", "batch_size")
            .cloned()
            .unwrap_or_else(|| json!(512)),
    );
    wire.insert(
        "kvCacheType".to_string(),
        JsonValue::String(kv_cache_type.to_string()),
    );
    for (camel, snake, default) in [
        ("useGpu", "use_gpu", true),
        ("preferDirectIo", "prefer_direct_io", true),
        ("cpuEmbeddings", "cpu_embeddings", false),
        ("cpuHead", "cpu_head", false),
        ("streamHead", "stream_head", false),
        ("moeActiveStreaming", "moe_active_streaming", true),
        ("verbose", "verbose", false),
    ] {
        let val = pick(&obj, camel, snake)
            .cloned()
            .unwrap_or(JsonValue::Bool(default));
        wire.insert(camel.to_string(), val);
    }
    wire.insert(
        "maxDraftTokens".to_string(),
        pick(&obj, "maxDraftTokens", "max_draft_tokens")
            .cloned()
            .unwrap_or_else(|| json!(5)),
    );

    Ok(json!({
        "isolatedWorker": isolated_worker,
        "wire": JsonValue::Object(wire),
    })
    .to_string())
}

#[napi]
pub fn normalize_gen_options_json(
    opts_json: String,
    resolved_grammar: Option<String>,
    allow_chat_extras: bool,
) -> Result<String> {
    let val: JsonValue = serde_json::from_str(&opts_json)
        .map_err(|_| type_err("generation options must be an object"))?;
    let JsonValue::Object(obj) = val else {
        return Err(type_err("generation options must be an object"));
    };

    let parse_f64 = |camel: &str, snake: &str, default: f64| -> std::result::Result<f64, ()> {
        match pick(&obj, camel, snake) {
            None => Ok(default),
            Some(JsonValue::Number(n)) => n.as_f64().filter(|v| v.is_finite()).ok_or(()),
            Some(_) => Err(()),
        }
    };

    let temperature = parse_f64("temperature", "temperature", 0.0)
        .ok()
        .filter(|&v| v >= 0.0)
        .ok_or_else(|| range_err("temperature must be a finite non-negative number"))?;
    let top_p = parse_f64("topP", "top_p", 0.95)
        .ok()
        .filter(|&v| v > 0.0 && v <= 1.0)
        .ok_or_else(|| range_err("top_p must be in (0.0, 1.0]"))?;
    let min_p = parse_f64("minP", "min_p", 0.0)
        .ok()
        .filter(|&v| (0.0..=1.0).contains(&v))
        .ok_or_else(|| range_err("min_p must be in [0.0, 1.0]"))?;
    let repeat_penalty = parse_f64("repeatPenalty", "repeat_penalty", 1.0)
        .ok()
        .filter(|&v| v > 0.0)
        .ok_or_else(|| range_err("repeat_penalty must be a finite positive number"))?;
    let frequency_penalty = parse_f64("frequencyPenalty", "frequency_penalty", 0.0)
        .map_err(|()| range_err("frequency_penalty must be finite"))?;
    let presence_penalty = parse_f64("presencePenalty", "presence_penalty", 0.0)
        .map_err(|()| range_err("presence_penalty must be finite"))?;

    let penalty_last_n = match pick(&obj, "penaltyLastN", "penalty_last_n") {
        None => 64_i64,
        Some(JsonValue::Number(n)) => n
            .as_i64()
            .filter(|&v| v >= -1)
            .ok_or_else(|| range_err("penalty_last_n must be >= -1"))?,
        Some(_) => return Err(range_err("penalty_last_n must be >= -1")),
    };

    let stop = match obj.get("stop") {
        None | Some(JsonValue::Null) => None,
        Some(JsonValue::String(s)) => Some(JsonValue::Array(vec![JsonValue::String(s.clone())])),
        Some(JsonValue::Array(items)) if items.iter().all(JsonValue::is_string) => {
            Some(JsonValue::Array(items.clone()))
        }
        Some(_) => {
            return Err(type_err(
                "stop must be undefined, null, a string, or an array of strings",
            ));
        }
    };

    let mut wire = Map::new();
    wire.insert(
        "maxTokens".to_string(),
        pick(&obj, "maxTokens", "max_tokens")
            .cloned()
            .unwrap_or_else(|| json!(256)),
    );
    wire.insert("temperature".to_string(), json!(temperature));
    wire.insert("topP".to_string(), json!(top_p));
    wire.insert("minP".to_string(), json!(min_p));
    wire.insert(
        "topK".to_string(),
        pick(&obj, "topK", "top_k")
            .cloned()
            .unwrap_or_else(|| json!(40)),
    );
    wire.insert("repeatPenalty".to_string(), json!(repeat_penalty));
    wire.insert("frequencyPenalty".to_string(), json!(frequency_penalty));
    wire.insert("presencePenalty".to_string(), json!(presence_penalty));
    wire.insert("penaltyLastN".to_string(), json!(penalty_last_n));
    wire.insert(
        "seed".to_string(),
        obj.get("seed")
            .filter(|v| !v.is_null())
            .cloned()
            .unwrap_or_else(|| json!(42)),
    );
    if let Some(s) = stop {
        wire.insert("stop".to_string(), s);
    }
    wire.insert(
        "ngramSpeculation".to_string(),
        pick(&obj, "ngramSpeculation", "ngram_speculation")
            .cloned()
            .unwrap_or(JsonValue::Bool(true)),
    );
    if let Some(mdt) = pick(&obj, "maxDraftTokens", "max_draft_tokens") {
        wire.insert("maxDraftTokens".to_string(), mdt.clone());
    }
    if let Some(g) = resolved_grammar {
        wire.insert("grammar".to_string(), JsonValue::String(g));
    }

    if allow_chat_extras {
        wire.insert(
            "addGenerationPrompt".to_string(),
            pick(&obj, "addGenerationPrompt", "add_generation_prompt")
                .cloned()
                .unwrap_or(JsonValue::Bool(true)),
        );
        if let Some(tools) = obj.get("tools")
            && !tools.is_null()
        {
            let JsonValue::Array(items) = tools else {
                return Err(type_err(
                    "tools must be an array of tool definition objects",
                ));
            };
            if !items.iter().all(JsonValue::is_object) {
                return Err(type_err("each entry in 'tools' must be a plain object"));
            }
            wire.insert("tools".to_string(), tools.clone());
        }
        let parse_output = pick(&obj, "parseOutput", "parse_output")
            .and_then(JsonValue::as_bool)
            .unwrap_or(false);
        wire.insert("parseOutput".to_string(), JsonValue::Bool(parse_output));
    }

    Ok(JsonValue::Object(wire).to_string())
}

#[napi]
pub fn build_chat_final_chunk_json(
    req_id: String,
    created: i64,
    model_id: String,
    has_tool_calls: bool,
    summary_json: Option<String>,
) -> String {
    let summary_val: Option<JsonValue> = summary_json
        .as_deref()
        .and_then(|s| serde_json::from_str(s).ok());
    let raw_reason = summary_finish_reason(summary_val.as_ref());
    let finish_reason = if has_tool_calls {
        "tool_calls"
    } else if raw_reason == "length" {
        "length"
    } else {
        "stop"
    };
    json!({
        "id": req_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_id,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": finish_reason,
            "logprobs": JsonValue::Null,
        }],
        "usage": usage_value(summary_val.as_ref()),
    })
    .to_string()
}

#[napi]
pub fn build_text_stream_chunk_json(
    req_id: String,
    created: i64,
    model_id: String,
    text: String,
    final_chunk: bool,
    summary_json: Option<String>,
) -> String {
    if final_chunk {
        let summary_val: Option<JsonValue> = summary_json
            .as_deref()
            .and_then(|s| serde_json::from_str(s).ok());
        let reason = summary_finish_reason(summary_val.as_ref());
        return json!({
            "id": req_id,
            "object": "text_completion",
            "created": created,
            "model": model_id,
            "choices": [{
                "text": "",
                "index": 0,
                "finish_reason": reason,
                "logprobs": JsonValue::Null,
            }],
            "usage": usage_value(summary_val.as_ref()),
        })
        .to_string();
    }
    json!({
        "id": req_id,
        "object": "text_completion",
        "created": created,
        "model": model_id,
        "choices": [{
            "text": text,
            "index": 0,
            "finish_reason": JsonValue::Null,
            "logprobs": JsonValue::Null,
        }],
        "usage": JsonValue::Null,
    })
    .to_string()
}
