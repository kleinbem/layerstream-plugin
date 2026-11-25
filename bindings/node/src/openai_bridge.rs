//! OpenAI response/chunk assembly and message normalization in Rust for the Node.js SDK.

use layerstream_core::parse_assistant_output as core_parse_assistant_output;
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
