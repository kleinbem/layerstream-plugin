//! Shared OpenAI response-format compilation, finish-reason normalization, and
//! usage helpers used across `layerstream-server`, Python bindings, and Node.js bindings.

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::engine::GenerateSummary;
use crate::grammar::{json_object_gbnf, json_schema_to_gbnf};

/// Result of compiling an OpenAI `response_format` specification into a GBNF grammar.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompiledResponseFormat {
    /// Compiled GBNF grammar string, or `None` for unconstrained `"text"` mode.
    pub grammar: Option<String>,
    /// Whether the output is expected to parse as JSON (`json_object`, `json_schema`, or raw schema).
    pub expects_json: bool,
}

/// Canonical OpenAI token usage object.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenAiUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

impl OpenAiUsage {
    pub fn from_summary(summary: &GenerateSummary) -> Self {
        Self {
            prompt_tokens: summary.prompt_tokens,
            completion_tokens: summary.completion_tokens,
            total_tokens: summary.prompt_tokens + summary.completion_tokens,
        }
    }
}

/// Normalize an engine finish reason into the canonical OpenAI `finish_reason` string.
pub fn normalize_finish_reason(raw: &str, has_tool_calls: bool) -> &'static str {
    if has_tool_calls {
        "tool_calls"
    } else if raw == "length" {
        "length"
    } else {
        "stop"
    }
}

/// Compile an OpenAI `response_format` JSON value into a GBNF grammar.
///
/// Supports:
/// - `null` -> unconstrained text (`grammar: None, expects_json: false`)
/// - `bool` -> JSON Schema boolean schema (`true` / `false`)
/// - `{"type": "text"}` -> unconstrained text
/// - `{"type": "json_object"}` -> arbitrary JSON object GBNF
/// - `{"type": "json_schema", "json_schema": {"schema": ...}}` or `{"type": "json_schema", "schema": ...}`
/// - `{"type": "gbnf", "grammar": "..."}` -> raw GBNF grammar
/// - Any other JSON object (when `allow_raw_schema` is `true`, e.g. raw JSON Schema `{ "properties": ... }`)
pub fn compile_response_format(
    rf: &Value,
    allow_raw_schema: bool,
) -> Result<CompiledResponseFormat> {
    if rf.is_null() {
        return Ok(CompiledResponseFormat {
            grammar: None,
            expects_json: false,
        });
    }
    if rf.is_boolean() {
        return Ok(CompiledResponseFormat {
            grammar: Some(json_schema_to_gbnf(rf)?),
            expects_json: true,
        });
    }
    let Some(obj) = rf.as_object() else {
        bail!("response_format must be a JSON object");
    };

    let rf_type = obj.get("type").and_then(Value::as_str);
    match rf_type {
        Some("text") => Ok(CompiledResponseFormat {
            grammar: None,
            expects_json: false,
        }),
        Some("json_object") => Ok(CompiledResponseFormat {
            grammar: Some(json_object_gbnf()),
            expects_json: true,
        }),
        Some("json_schema") => {
            let schema = if let Some(inner) = obj.get("json_schema") {
                let Some(inner_obj) = inner.as_object() else {
                    bail!("response_format.json_schema must be an object");
                };
                inner_obj.get("schema").ok_or_else(|| {
                    anyhow::anyhow!("response_format 'json_schema' requires an explicit 'schema'")
                })?
            } else if let Some(s) = obj.get("schema") {
                s
            } else {
                bail!("response_format 'json_schema' requires an explicit 'schema'");
            };
            if schema.is_null() {
                bail!("response_format 'json_schema' requires a non-null 'schema'");
            }
            Ok(CompiledResponseFormat {
                grammar: Some(json_schema_to_gbnf(schema)?),
                expects_json: true,
            })
        }
        Some("gbnf") => {
            let g = obj
                .get("grammar")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .ok_or_else(|| {
                    anyhow::anyhow!("response_format 'gbnf' requires a non-empty 'grammar' string")
                })?;
            Ok(CompiledResponseFormat {
                grammar: Some(g.to_string()),
                expects_json: false,
            })
        }
        None if !allow_raw_schema => Ok(CompiledResponseFormat {
            grammar: None,
            expects_json: false,
        }),
        Some(other) if !allow_raw_schema => {
            bail!(
                "unsupported response_format type '{other}' \
                 (expected 'text', 'json_object', 'json_schema', or 'gbnf')"
            );
        }
        _ => Ok(CompiledResponseFormat {
            grammar: Some(json_schema_to_gbnf(rf)?),
            expects_json: true,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn compile_response_format_handles_all_standard_variants() {
        let text = compile_response_format(&json!({"type": "text"}), false).unwrap();
        assert_eq!(text.grammar, None);
        assert!(!text.expects_json);

        let obj = compile_response_format(&json!({"type": "json_object"}), false).unwrap();
        assert!(obj.grammar.as_deref().unwrap().contains("root ::="));
        assert!(obj.expects_json);

        let schema = compile_response_format(
            &json!({
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "schema": {
                        "type": "object",
                        "properties": {"ok": {"type": "boolean"}},
                        "required": ["ok"]
                    }
                }
            }),
            false,
        )
        .unwrap();
        assert!(schema.grammar.as_deref().unwrap().contains("root ::="));
        assert!(schema.expects_json);

        let gbnf = compile_response_format(
            &json!({"type": "gbnf", "grammar": "root ::= \"yes\""}),
            false,
        )
        .unwrap();
        assert_eq!(gbnf.grammar.as_deref(), Some("root ::= \"yes\""));
        assert!(!gbnf.expects_json);

        assert!(compile_response_format(&json!({"type": "bogus"}), false).is_err());
        assert!(compile_response_format(&json!({"type": "gbnf", "grammar": "  "}), false).is_err());
        assert!(compile_response_format(&json!({"type": "json_schema"}), false).is_err());
        assert!(
            compile_response_format(
                &json!({"type": "json_schema", "json_schema": {"name": "missing"}}),
                true,
            )
            .is_err()
        );
    }

    #[test]
    fn normalize_finish_reason_prioritizes_tool_calls() {
        assert_eq!(normalize_finish_reason("stop", true), "tool_calls");
        assert_eq!(normalize_finish_reason("length", true), "tool_calls");
        assert_eq!(normalize_finish_reason("length", false), "length");
        assert_eq!(normalize_finish_reason("stop", false), "stop");
        assert_eq!(normalize_finish_reason("eos", false), "stop");
    }
}
