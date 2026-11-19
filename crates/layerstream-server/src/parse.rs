//! Helpers for extracting `<think>...</think>` reasoning blocks and
//! `<tool_call>...</tool_call>` OpenAI tool-call payloads from assistant output.

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub function: OpenAiFunctionCall,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiFunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Default)]
pub struct ParsedAssistantOutput {
    pub content: String,
    pub reasoning_content: Option<String>,
    pub tool_calls: Vec<OpenAiToolCall>,
}

/// Splits `<think>...</think>` and `<tool_call>...</tool_call>` blocks out of raw assistant text.
pub fn parse_assistant_output(raw: &str) -> ParsedAssistantOutput {
    let mut reasoning_parts = Vec::new();
    let mut after_think = String::with_capacity(raw.len());
    let mut cursor = raw;

    while let Some(start) = cursor.find("<think>") {
        after_think.push_str(&cursor[..start]);
        let rest = &cursor[start + "<think>".len()..];
        if let Some(end) = rest.find("</think>") {
            let block = rest[..end].trim();
            if !block.is_empty() {
                reasoning_parts.push(block.to_string());
            }
            cursor = &rest[end + "</think>".len()..];
        } else {
            let block = rest.trim();
            if !block.is_empty() {
                reasoning_parts.push(block.to_string());
            }
            cursor = "";
            break;
        }
    }
    after_think.push_str(cursor);

    let mut tool_calls = Vec::new();
    let mut visible = String::with_capacity(after_think.len());
    let mut tcur = after_think.as_str();
    let mut call_idx = 0usize;

    while let Some(start) = tcur.find("<tool_call>") {
        visible.push_str(&tcur[..start]);
        let rest = &tcur[start + "<tool_call>".len()..];
        if let Some(end) = rest.find("</tool_call>") {
            let body = rest[..end].trim();
            if let Ok(val) = serde_json::from_str::<Value>(body)
                && let Some(name) = val.get("name").and_then(Value::as_str)
            {
                let args_val = val
                    .get("arguments")
                    .cloned()
                    .unwrap_or_else(|| Value::Object(Default::default()));
                let arguments = match args_val {
                    Value::String(s) => s,
                    other => other.to_string(),
                };
                call_idx += 1;
                tool_calls.push(OpenAiToolCall {
                    id: format!("call_ls_{call_idx}"),
                    kind: "function".to_string(),
                    function: OpenAiFunctionCall {
                        name: name.to_string(),
                        arguments,
                    },
                });
            }
            tcur = &rest[end + "</tool_call>".len()..];
        } else {
            visible.push_str(&tcur[start..]);
            tcur = "";
            break;
        }
    }
    visible.push_str(tcur);

    let reasoning_content = if reasoning_parts.is_empty() {
        None
    } else {
        Some(reasoning_parts.join("\n"))
    };

    ParsedAssistantOutput {
        content: visible.trim().to_string(),
        reasoning_content,
        tool_calls,
    }
}

/// Formats OpenAI `tools` definitions into a concise system instruction block
/// so models emit structured `<tool_call>{"name": ..., "arguments": ...}</tool_call>` payloads.
pub fn format_tools_system_prompt(tools: &[Value]) -> String {
    let mut out = String::from(
        "You have access to the following tools. To call a tool, respond with a \
         <tool_call>{\"name\": \"<tool_name>\", \"arguments\": {...}}</tool_call> block.\n\nTools:\n",
    );
    for tool in tools {
        if let Ok(json) = serde_json::to_string(tool) {
            out.push_str(&json);
            out.push('\n');
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_reasoning_and_tool_call() {
        let raw = "<think>Need weather.</think>\n<tool_call>{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Vienna\"}}</tool_call>";
        let parsed = parse_assistant_output(raw);
        assert_eq!(parsed.reasoning_content.as_deref(), Some("Need weather."));
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].function.name, "get_weather");
        assert!(parsed.content.is_empty());
    }
}
