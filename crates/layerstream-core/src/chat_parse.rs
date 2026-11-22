//! Helpers for extracting `<think>...</think>` reasoning blocks and
//! `<tool_call>...</tool_call>` OpenAI tool-call payloads from assistant output.

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct OpenAiToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub function: OpenAiFunctionCall,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct OpenAiFunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ParsedAssistantOutput {
    pub content: String,
    pub reasoning_content: Option<String>,
    pub tool_calls: Vec<OpenAiToolCall>,
}

/// Splits `<think>...</think>` and `<tool_call>...</tool_call>` blocks out of raw assistant text
/// using the exact same single-pass state machine as streaming mode so tags inside JSON tool
/// arguments are never stripped or prematurely terminated.
pub fn parse_assistant_output(raw: &str) -> ParsedAssistantOutput {
    let mut parser = StreamTagParser::new();
    let mut events = parser.push(raw);
    events.extend(parser.flush());

    let mut visible = String::with_capacity(raw.len());
    let mut reasoning_parts = Vec::new();
    let mut tool_calls = Vec::new();

    for ev in events {
        match ev {
            StreamDelta::Content(s) => visible.push_str(&s),
            StreamDelta::Reasoning(s) => {
                let trimmed = s.trim();
                if !trimmed.is_empty() {
                    reasoning_parts.push(trimmed.to_string());
                }
            }
            StreamDelta::ToolCall {
                id,
                name,
                arguments,
                ..
            } => tool_calls.push(OpenAiToolCall {
                id,
                kind: "function".to_string(),
                function: OpenAiFunctionCall { name, arguments },
            }),
        }
    }

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
                    if !self.pending.is_empty() {
                        self.tool_buf.push_str(&self.pending);
                        self.pending.clear();
                    }
                    if let Some(end) = find_tool_call_close(&self.tool_buf) {
                        let body = self.tool_buf[..end].trim().to_string();
                        let rest = self.tool_buf[end + "</tool_call>".len()..].to_string();
                        self.tool_buf.clear();
                        self.pending = rest;
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

/// Finds the closing `</tool_call>` tag outside of JSON string literals so that a
/// `"</tool_call>"` substring inside a tool argument value does not truncate the call.
fn find_tool_call_close(buf: &str) -> Option<usize> {
    const CLOSE: &[u8] = b"</tool_call>";
    let bytes = buf.as_bytes();
    let trimmed_start = bytes
        .iter()
        .position(|b| !b.is_ascii_whitespace())
        .unwrap_or(bytes.len());
    if trimmed_start >= bytes.len() {
        return None;
    }
    if bytes[trimmed_start] != b'{' {
        return buf.find("</tool_call>");
    }

    let mut in_string = false;
    let mut escape = false;
    let mut i = trimmed_start;
    while i < bytes.len() {
        let b = bytes[i];
        if in_string {
            if escape {
                escape = false;
            } else if b == b'\\' {
                escape = true;
            } else if b == b'"' {
                in_string = false;
            }
            i += 1;
            continue;
        }
        if b == b'"' {
            in_string = true;
            i += 1;
            continue;
        }
        if b == b'<' && bytes[i..].starts_with(CLOSE) {
            return Some(i);
        }
        i += 1;
    }
    None
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

    #[test]
    fn preserves_think_and_close_tags_inside_tool_arguments_in_both_modes() {
        let raw = r#"<tool_call>{"name":"write_note","arguments":{"text":"<think>note</think> and </tool_call> literal"}}</tool_call>done"#;

        let parsed = parse_assistant_output(raw);
        assert!(parsed.reasoning_content.is_none());
        assert_eq!(parsed.content, "done");
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].function.name, "write_note");
        assert_eq!(
            parsed.tool_calls[0].function.arguments,
            r#"{"text":"<think>note</think> and </tool_call> literal"}"#
        );

        let mut parser = StreamTagParser::new();
        let pieces = [
            "<tool_call>{\"name\":\"write_note\",\"arguments\":{\"text\":\"<think>note</think> and </tool_",
            "call> literal\"}}</tool",
            "_call>done",
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
        assert!(reasoning.is_empty());
        assert_eq!(content, "done");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, "write_note");
        assert_eq!(
            calls[0].1,
            r#"{"text":"<think>note</think> and </tool_call> literal"}"#
        );
    }

    #[test]
    fn malformed_tool_call_is_preserved_in_content() {
        let raw = "before <tool_call>{not valid json</tool_call> after";
        let parsed = parse_assistant_output(raw);
        assert!(parsed.tool_calls.is_empty());
        assert_eq!(
            parsed.content,
            "before <tool_call>{not valid json</tool_call> after"
        );
    }
}
