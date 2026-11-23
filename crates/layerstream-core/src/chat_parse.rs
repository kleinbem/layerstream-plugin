//! Helpers for extracting `<think>...</think>` reasoning blocks and
//! `<tool_call>...</tool_call>` OpenAI tool-call payloads from assistant output.

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::engine::ChatMessage;

/// Canonical OpenAI-compatible incoming chat message normalizer shared across `layerstream-server`,
/// Python bindings, and Node.js bindings.
#[derive(Debug, Clone, Deserialize)]
pub struct IncomingChatMessage {
    pub role: String,
    #[serde(default)]
    pub content: Value,
    #[serde(default)]
    pub tool_calls: Option<Vec<Value>>,
    #[serde(default)]
    pub tool_call_id: Option<String>,
    #[serde(default)]
    pub name: Option<String>,
}

impl IncomingChatMessage {
    pub fn to_chat_message(&self) -> Result<ChatMessage, String> {
        let mut text = match &self.content {
            Value::String(s) => s.clone(),
            Value::Array(parts) => {
                let mut texts = Vec::with_capacity(parts.len());
                for part in parts {
                    match part {
                        Value::String(s) => texts.push(s.as_str()),
                        Value::Object(map) => {
                            let kind = map.get("type").and_then(Value::as_str);
                            match kind {
                                Some("text") | None => {
                                    if let Some(t) = map.get("text").and_then(Value::as_str) {
                                        texts.push(t);
                                    } else {
                                        return Err(format!(
                                            "unsupported chat message content part: {part}"
                                        ));
                                    }
                                }
                                Some(other) => {
                                    return Err(format!(
                                        "unsupported multimodal content part type '{other}'; only text content is supported"
                                    ));
                                }
                            }
                        }
                        _ => {
                            return Err(format!(
                                "unsupported chat message content part (expected string or object): {part}"
                            ));
                        }
                    }
                }
                texts.join("\n")
            }
            Value::Null => String::new(),
            other => {
                return Err(format!(
                    "unsupported chat message content value (expected string, array, or null): {other}"
                ));
            }
        };

        // Preserve assistant `tool_calls` across multi-turn agent loops.
        if let Some(calls) = &self.tool_calls
            && !calls.is_empty()
        {
            for call in calls {
                let func = call.get("function").unwrap_or(call);
                let name = func.get("name").and_then(Value::as_str).unwrap_or("");
                let raw_args = func.get("arguments").cloned().unwrap_or(Value::Null);
                let args_val = match raw_args {
                    Value::String(ref s) => {
                        serde_json::from_str::<Value>(s).unwrap_or(Value::String(s.clone()))
                    }
                    other => other,
                };
                if !name.is_empty() {
                    if !text.is_empty() && !text.ends_with('\n') {
                        text.push('\n');
                    }
                    let mut payload = json!({"name": name, "arguments": args_val});
                    if let Some(call_id) = call.get("id").and_then(Value::as_str)
                        && !call_id.is_empty()
                    {
                        payload["id"] = json!(call_id);
                    }
                    text.push_str(&format!("<tool_call>{payload}</tool_call>"));
                }
            }
        }

        // Preserve tool response identity (`tool_call_id` / `name`) for `role: "tool"`.
        if self.role == "tool" {
            let mut prefix = String::new();
            if let Some(name) = &self.name
                && !name.is_empty()
            {
                prefix.push_str(&format!("tool={name} "));
            }
            if let Some(id) = &self.tool_call_id
                && !id.is_empty()
            {
                prefix.push_str(&format!("call_id={id}: "));
            }
            if !prefix.is_empty() {
                text = format!("{prefix}{text}");
            }
        }

        Ok(ChatMessage {
            role: self.role.clone(),
            content: text,
        })
    }
}

/// Injects formatted OpenAI `tools` definitions into the leading system message.
pub fn inject_tools_into_messages(messages: &mut Vec<ChatMessage>, tools: &[Value]) {
    if tools.is_empty() {
        return;
    }
    let tool_prompt = format_tools_system_prompt(tools);
    if let Some(first) = messages.first_mut()
        && first.role == "system"
    {
        if !first.content.is_empty() {
            first.content.push_str("\n\n");
        }
        first.content.push_str(&tool_prompt);
    } else {
        messages.insert(
            0,
            ChatMessage {
                role: "system".to_string(),
                content: tool_prompt,
            },
        );
    }
}

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

#[derive(Debug, Default, Clone)]
struct ToolScanState {
    pos: usize,
    started_json: Option<bool>,
    in_string: bool,
    escape: bool,
    json_invalid: bool,
}

impl ToolScanState {
    fn reset(&mut self) {
        *self = Self::default();
    }

    /// Incrementally scans `buf` from `self.pos` for the closing `</tool_call>` tag outside
    /// valid JSON string literals. Recovers immediately at `</tool_call>` if a raw control
    /// character (such as `\n` or `\r`) or invalid escape breaks a quoted JSON string.
    fn find_close(&mut self, buf: &str) -> Option<usize> {
        const CLOSE: &[u8] = b"</tool_call>";
        let bytes = buf.as_bytes();

        while self.pos < bytes.len() {
            let b = bytes[self.pos];

            if self.started_json.is_none() {
                if b.is_ascii_whitespace() {
                    self.pos += 1;
                    continue;
                }
                self.started_json = Some(b == b'{');
            }

            let plain_scan = self.started_json == Some(false) || self.json_invalid;
            if !plain_scan && self.in_string {
                if self.escape {
                    self.escape = false;
                    if !matches!(
                        b,
                        b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' | b'u'
                    ) {
                        self.json_invalid = true;
                        self.in_string = false;
                        continue;
                    }
                    self.pos += 1;
                    continue;
                }
                if b == b'\\' {
                    self.escape = true;
                    self.pos += 1;
                    continue;
                }
                if b == b'"' {
                    self.in_string = false;
                    self.pos += 1;
                    continue;
                }
                // RFC 8259: raw control characters (< 0x20, including \n and \r) are illegal in
                // JSON strings; mark JSON prefix invalid so `</tool_call>` can recover cleanly.
                if b < 0x20 {
                    self.json_invalid = true;
                    self.in_string = false;
                    continue;
                }
                self.pos += 1;
                continue;
            }

            if !plain_scan && b == b'"' {
                self.in_string = true;
                self.pos += 1;
                continue;
            }

            if b == b'<' {
                let rem = &bytes[self.pos..];
                if rem.starts_with(CLOSE) {
                    return Some(self.pos);
                }
                if CLOSE.starts_with(rem) {
                    // Partial `</tool_call>` suffix at end of buffer; resume from `self.pos` on next push.
                    return None;
                }
            }
            self.pos += 1;
        }
        None
    }
}

pub struct StreamTagParser {
    state: TagState,
    pending: String,
    tool_buf: String,
    tool_scan: ToolScanState,
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
            tool_scan: ToolScanState::default(),
            tool_call_count: 0,
        }
    }

    pub fn has_tool_calls(&self) -> bool {
        self.tool_call_count > 0
    }

    pub fn push(&mut self, piece: &str) -> Vec<StreamDelta> {
        let mut out = Vec::new();
        if !piece.is_empty() {
            self.pending.push_str(piece);
        }

        // Advance `cursor` forward through `self.pending` and compact once at the end of `push`,
        // caching the next `<think>` and `<tool_call>` offsets so multi-tag inputs never rescan.
        let mut cursor = 0usize;
        let mut cached_think: Option<Option<usize>> = None;
        let mut cached_tool: Option<Option<usize>> = None;

        loop {
            match self.state {
                TagState::Normal => {
                    let slice = &self.pending[cursor..];
                    let think_pos = match cached_think {
                        Some(Some(abs)) if abs >= cursor => Some(abs - cursor),
                        Some(None) => None,
                        _ => {
                            let found = slice.find("<think>").map(|rel| cursor + rel);
                            cached_think = Some(found);
                            found.map(|abs| abs - cursor)
                        }
                    };
                    let tool_pos = match cached_tool {
                        Some(Some(abs)) if abs >= cursor => Some(abs - cursor),
                        Some(None) => None,
                        _ => {
                            let found = slice.find("<tool_call>").map(|rel| cursor + rel);
                            cached_tool = Some(found);
                            found.map(|abs| abs - cursor)
                        }
                    };

                    match (think_pos, tool_pos) {
                        (Some(t), Some(c)) if t <= c => {
                            if t > 0 {
                                out.push(StreamDelta::Content(slice[..t].to_string()));
                            }
                            cursor += t + "<think>".len();
                            self.state = TagState::InThink;
                        }
                        (Some(t), None) => {
                            if t > 0 {
                                out.push(StreamDelta::Content(slice[..t].to_string()));
                            }
                            cursor += t + "<think>".len();
                            self.state = TagState::InThink;
                        }
                        (_, Some(c)) => {
                            if c > 0 {
                                out.push(StreamDelta::Content(slice[..c].to_string()));
                            }
                            cursor += c + "<tool_call>".len();
                            self.tool_buf.clear();
                            self.tool_scan.reset();
                            self.state = TagState::InToolCall;
                        }
                        (None, None) => {
                            let hold = partial_tag_suffix(slice, &["<think>", "<tool_call>"]);
                            let emit_len = slice.len() - hold;
                            if emit_len > 0 {
                                out.push(StreamDelta::Content(slice[..emit_len].to_string()));
                                cursor += emit_len;
                            }
                            break;
                        }
                    }
                }
                TagState::InThink => {
                    let slice = &self.pending[cursor..];
                    if let Some(end) = slice.find("</think>") {
                        if end > 0 {
                            out.push(StreamDelta::Reasoning(slice[..end].to_string()));
                        }
                        cursor += end + "</think>".len();
                        self.state = TagState::Normal;
                    } else {
                        let hold = partial_tag_suffix(slice, &["</think>"]);
                        let emit_len = slice.len() - hold;
                        if emit_len > 0 {
                            out.push(StreamDelta::Reasoning(slice[..emit_len].to_string()));
                            cursor += emit_len;
                        }
                        break;
                    }
                }
                TagState::InToolCall => {
                    if cursor < self.pending.len() {
                        self.tool_buf.push_str(&self.pending[cursor..]);
                        cursor = self.pending.len();
                    }
                    if let Some(end) = self.tool_scan.find_close(&self.tool_buf) {
                        let raw_body = &self.tool_buf[..end];
                        let trimmed_body = raw_body.trim();
                        let decoded =
                            serde_json::from_str::<Value>(trimmed_body)
                                .ok()
                                .and_then(|val| {
                                    let name = val.get("name").and_then(Value::as_str)?.to_string();
                                    let args_val = val
                                        .get("arguments")
                                        .cloned()
                                        .unwrap_or_else(|| Value::Object(Default::default()));
                                    let arguments = match args_val {
                                        Value::String(s) => s,
                                        other => other.to_string(),
                                    };
                                    Some((name, arguments))
                                });
                        if let Some((name, arguments)) = decoded {
                            let idx = self.tool_call_count;
                            self.tool_call_count += 1;
                            out.push(StreamDelta::ToolCall {
                                index: idx,
                                id: format!("call_ls_{}", idx + 1),
                                name,
                                arguments,
                            });
                        } else {
                            out.push(StreamDelta::Content(format!(
                                "<tool_call>{raw_body}</tool_call>"
                            )));
                        }
                        let rest = self.tool_buf[end + "</tool_call>".len()..].to_string();
                        self.tool_buf.clear();
                        self.tool_scan.reset();
                        self.pending = rest;
                        cursor = 0;
                        cached_think = None;
                        cached_tool = None;
                        self.state = TagState::Normal;
                    } else {
                        break;
                    }
                }
            }
        }

        if cursor > 0 {
            if cursor >= self.pending.len() {
                self.pending.clear();
            } else {
                self.pending.drain(..cursor);
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
                self.tool_scan.reset();
                unclosed.push_str(&rem);
                if !unclosed.is_empty() {
                    out.push(StreamDelta::Content(format!("<tool_call>{unclosed}")));
                } else {
                    out.push(StreamDelta::Content("<tool_call>".to_string()));
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
    fn malformed_tool_call_preserves_exact_whitespace_and_recovers_after_newline_in_string() {
        let raw = "before <tool_call> {not valid json} </tool_call> mid <tool_call>   </tool_call>";
        let parsed = parse_assistant_output(raw);
        assert!(parsed.tool_calls.is_empty());
        assert_eq!(
            parsed.content,
            "before <tool_call> {not valid json} </tool_call> mid <tool_call>   </tool_call>"
        );

        // Raw newline inside an unclosed JSON string invalidates the JSON prefix; scanner recovers
        // at `</tool_call>` so the trailing `<think>` block is extracted as reasoning.
        let broken_then_think = "<tool_call>{\"name\":\"x\",\"arguments\":\"unclosed\n</tool_call><think>plan</think>answer";
        let p2 = parse_assistant_output(broken_then_think);
        assert_eq!(p2.reasoning_content.as_deref(), Some("plan"));
        assert_eq!(
            p2.content,
            "<tool_call>{\"name\":\"x\",\"arguments\":\"unclosed\n</tool_call>answer"
        );
    }
}
