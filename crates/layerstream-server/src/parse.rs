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

/// Incremental streaming parser for `<think>...</think>` and `<tool_call>...</tool_call>`
/// blocks that handles tags split arbitrarily across token boundaries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamDelta {
    Content(String),
    Reasoning(String),
    ToolCall {
        index: usize,
        id: String,
        name: String,
        arguments: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TagState {
    Normal,
    InThink,
    InToolCall,
}

pub struct StreamTagParser {
    state: TagState,
    pending: String,
    tool_buf: String,
    tool_call_count: usize,
}

impl Default for StreamTagParser {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamTagParser {
    pub fn new() -> Self {
        Self {
            state: TagState::Normal,
            pending: String::new(),
            tool_buf: String::new(),
            tool_call_count: 0,
        }
    }

    pub fn has_tool_calls(&self) -> bool {
        self.tool_call_count > 0
    }

    pub fn push(&mut self, piece: &str) -> Vec<StreamDelta> {
        let mut out = Vec::new();
        self.pending.push_str(piece);

        loop {
            match self.state {
                TagState::Normal => {
                    let think_pos = self.pending.find("<think>");
                    let tool_pos = self.pending.find("<tool_call>");
                    match (think_pos, tool_pos) {
                        (Some(t), Some(c)) if t <= c => {
                            if t > 0 {
                                out.push(StreamDelta::Content(self.pending[..t].to_string()));
                            }
                            self.pending.drain(..t + "<think>".len());
                            self.state = TagState::InThink;
                        }
                        (Some(t), None) => {
                            if t > 0 {
                                out.push(StreamDelta::Content(self.pending[..t].to_string()));
                            }
                            self.pending.drain(..t + "<think>".len());
                            self.state = TagState::InThink;
                        }
                        (_, Some(c)) => {
                            if c > 0 {
                                out.push(StreamDelta::Content(self.pending[..c].to_string()));
                            }
                            self.pending.drain(..c + "<tool_call>".len());
                            self.tool_buf.clear();
                            self.state = TagState::InToolCall;
                        }
                        (None, None) => {
                            let hold =
                                partial_tag_suffix(&self.pending, &["<think>", "<tool_call>"]);
                            let emit_len = self.pending.len() - hold;
                            if emit_len > 0 {
                                out.push(StreamDelta::Content(
                                    self.pending[..emit_len].to_string(),
                                ));
                                self.pending.drain(..emit_len);
                            }
                            break;
                        }
                    }
                }
                TagState::InThink => {
                    if let Some(end) = self.pending.find("</think>") {
                        if end > 0 {
                            out.push(StreamDelta::Reasoning(self.pending[..end].to_string()));
                        }
                        self.pending.drain(..end + "</think>".len());
                        self.state = TagState::Normal;
                    } else {
                        let hold = partial_tag_suffix(&self.pending, &["</think>"]);
                        let emit_len = self.pending.len() - hold;
                        if emit_len > 0 {
                            out.push(StreamDelta::Reasoning(self.pending[..emit_len].to_string()));
                            self.pending.drain(..emit_len);
                        }
                        break;
                    }
                }
                TagState::InToolCall => {
                    if let Some(end) = self.pending.find("</tool_call>") {
                        self.tool_buf.push_str(&self.pending[..end]);
                        self.pending.drain(..end + "</tool_call>".len());
                        let body = self.tool_buf.trim().to_string();
                        self.tool_buf.clear();
                        self.state = TagState::Normal;
                        if let Ok(val) = serde_json::from_str::<Value>(&body)
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
                            let idx = self.tool_call_count;
                            self.tool_call_count += 1;
                            out.push(StreamDelta::ToolCall {
                                index: idx,
                                id: format!("call_ls_{}", idx + 1),
                                name: name.to_string(),
                                arguments,
                            });
                        } else if !body.is_empty() {
                            out.push(StreamDelta::Content(format!(
                                "<tool_call>{body}</tool_call>"
                            )));
                        }
                    } else {
                        let hold = partial_tag_suffix(&self.pending, &["</tool_call>"]);
                        let move_len = self.pending.len() - hold;
                        if move_len > 0 {
                            self.tool_buf.push_str(&self.pending[..move_len]);
                            self.pending.drain(..move_len);
                        }
                        break;
                    }
                }
            }
        }

        out
    }

    pub fn flush(&mut self) -> Vec<StreamDelta> {
        let mut out = Vec::new();
        let rem = std::mem::take(&mut self.pending);
        match self.state {
            TagState::Normal => {
                if !rem.is_empty() {
                    out.push(StreamDelta::Content(rem));
                }
            }
            TagState::InThink => {
                if !rem.is_empty() {
                    out.push(StreamDelta::Reasoning(rem));
                }
            }
            TagState::InToolCall => {
                let mut unclosed = std::mem::take(&mut self.tool_buf);
                unclosed.push_str(&rem);
                if !unclosed.is_empty() {
                    out.push(StreamDelta::Content(format!("<tool_call>{unclosed}")));
                }
            }
        }
        self.state = TagState::Normal;
        out
    }
}

/// Returns the byte length of the longest suffix of `s` that is a proper prefix of any tag.
fn partial_tag_suffix(s: &str, tags: &[&str]) -> usize {
    let mut best = 0usize;
    for tag in tags {
        let max_len = (tag.len() - 1).min(s.len());
        for len in (1..=max_len).rev() {
            if len <= best {
                break;
            }
            let start = s.len() - len;
            if s.is_char_boundary(start) && tag.starts_with(&s[start..]) {
                best = len;
                break;
            }
        }
    }
    best
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

    #[test]
    fn stream_tag_parser_handles_split_tags() {
        let mut parser = StreamTagParser::new();
        let pieces = [
            "Hi ",
            "<th",
            "ink>",
            "reason",
            "ing</th",
            "ink>",
            " ok ",
            "<tool_",
            "call>{\"name\":\"ls\",\"arguments\":{\"path\":\"/tmp\"}}</tool",
            "_call>!",
        ];
        let mut content = String::new();
        let mut reasoning = String::new();
        let mut calls = Vec::new();
        for p in pieces {
            for ev in parser.push(p) {
                match ev {
                    StreamDelta::Content(s) => content.push_str(&s),
                    StreamDelta::Reasoning(s) => reasoning.push_str(&s),
                    StreamDelta::ToolCall {
                        name, arguments, ..
                    } => calls.push((name, arguments)),
                }
            }
        }
        for ev in parser.flush() {
            if let StreamDelta::Content(s) = ev {
                content.push_str(&s);
            }
        }
        assert_eq!(reasoning, "reasoning");
        assert_eq!(content, "Hi  ok !");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, "ls");
        assert!(parser.has_tool_calls());
    }
}
