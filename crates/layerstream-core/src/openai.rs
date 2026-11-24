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

/// Validates sampling parameters with `f32` precision (matching native FFI),
/// enforcing canonical bounds across Python, Node.js, and Rust callers.
pub fn validate_sample_params(
    temperature: f64,
    top_p: f64,
    min_p: f64,
    repeat_penalty: f64,
    frequency_penalty: f64,
    presence_penalty: f64,
    penalty_last_n: i32,
) -> Result<()> {
    let as_f32 = |v: f64| if v.is_finite() { v as f32 } else { f32::NAN };
    let temperature = as_f32(temperature);
    let top_p = as_f32(top_p);
    let min_p = as_f32(min_p);
    let repeat_penalty = as_f32(repeat_penalty);
    let frequency_penalty = as_f32(frequency_penalty);
    let presence_penalty = as_f32(presence_penalty);

    if !temperature.is_finite() || temperature < 0.0 {
        bail!("temperature must be a finite non-negative number");
    }
    if !top_p.is_finite() || top_p <= 0.0 || top_p > 1.0 {
        bail!("top_p must be in (0.0, 1.0]");
    }
    if !min_p.is_finite() || !(0.0..=1.0).contains(&min_p) {
        bail!("min_p must be in [0.0, 1.0]");
    }
    if !repeat_penalty.is_finite() || repeat_penalty <= 0.0 {
        bail!("repeat_penalty must be a finite positive number");
    }
    if !frequency_penalty.is_finite() || !(-2.0..=2.0).contains(&frequency_penalty) {
        bail!("frequency_penalty must be in [-2.0, 2.0]");
    }
    if !presence_penalty.is_finite() || !(-2.0..=2.0).contains(&presence_penalty) {
        bail!("presence_penalty must be in [-2.0, 2.0]");
    }
    if penalty_last_n < -1 {
        bail!("penalty_last_n must be >= -1");
    }
    Ok(())
}

/// Normalizes a `vram_budget_mb` JSON value (`"auto"`, `null`, non-negative number, or integer
/// string) into `(vram_budget_auto, vram_budget_bytes)`.
pub fn normalize_vram_budget_mb(raw: &Value) -> Result<(bool, u64)> {
    match raw {
        Value::Null => Ok((false, 0)),
        Value::Bool(_) => {
            bail!("vram_budget_mb must be 'auto', None, or a non-negative integer MiB (not bool)")
        }
        Value::String(s) => {
            let trimmed = s.trim();
            if trimmed.eq_ignore_ascii_case("auto") {
                Ok((true, 0))
            } else {
                let parsed: i64 = trimmed.parse().map_err(|_| {
                    anyhow::anyhow!(
                        "invalid vram_budget_mb '{trimmed}' (expected 'auto' or non-negative integer MiB)"
                    )
                })?;
                if parsed < 0 {
                    bail!("vram_budget_mb must be non-negative");
                }
                Ok((false, (parsed as u64).saturating_mul(1024 * 1024)))
            }
        }
        Value::Number(n) => {
            if let Some(u) = n.as_u64() {
                Ok((false, u.saturating_mul(1024 * 1024)))
            } else if let Some(i) = n.as_i64() {
                if i < 0 {
                    bail!("vram_budget_mb must be non-negative");
                }
                Ok((false, (i as u64).saturating_mul(1024 * 1024)))
            } else if let Some(f) = n.as_f64() {
                if !f.is_finite() || f < 0.0 {
                    bail!("vram_budget_mb must be non-negative");
                }
                Ok((false, (f * 1024.0 * 1024.0).floor() as u64))
            } else {
                bail!("vram_budget_mb must be non-negative");
            }
        }
        _ => bail!("vram_budget_mb must be 'auto', None, or a non-negative integer MiB"),
    }
}

/// Normalizes and validates `kv_cache_type` (`"f16"`, `"q8_0"`, or `"q4_0"`).
pub fn normalize_kv_cache_type(raw: &str) -> Result<&'static str> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "f16" => Ok("f16"),
        "q8_0" => Ok("q8_0"),
        "q4_0" => Ok("q4_0"),
        _ => bail!("invalid kv_cache_type '{raw}' (expected f16, q8_0, or q4_0)"),
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

    #[test]
    fn validate_sample_params_enforces_f32_bounds() {
        assert!(validate_sample_params(0.0, 0.95, 0.0, 1.0, 0.0, 0.0, 64).is_ok());
        assert!(validate_sample_params(-0.1, 0.95, 0.0, 1.0, 0.0, 0.0, 64).is_err());
        assert!(validate_sample_params(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 64).is_err());
        assert!(validate_sample_params(0.0, 1e-50, 0.0, 1.0, 0.0, 0.0, 64).is_err());
        assert!(validate_sample_params(0.0, 0.95, 1.1, 1.0, 0.0, 0.0, 64).is_err());
        assert!(validate_sample_params(0.0, 0.95, 0.0, 0.0, 0.0, 0.0, 64).is_err());
        assert!(validate_sample_params(0.0, 0.95, 0.0, 1.0, 2.5, 0.0, 64).is_err());
        assert!(validate_sample_params(0.0, 0.95, 0.0, 1.0, 0.0, -2.5, 64).is_err());
        assert!(validate_sample_params(0.0, 0.95, 0.0, 1.0, 0.0, 0.0, -2).is_err());
    }

    #[test]
    fn normalize_vram_budget_and_kv_handles_edge_cases() {
        assert_eq!(normalize_vram_budget_mb(&json!("auto")).unwrap(), (true, 0));
        assert_eq!(
            normalize_vram_budget_mb(&json!(" AUTO ")).unwrap(),
            (true, 0)
        );
        assert_eq!(normalize_vram_budget_mb(&json!(null)).unwrap(), (false, 0));
        assert_eq!(
            normalize_vram_budget_mb(&json!(1024)).unwrap(),
            (false, 1024 * 1024 * 1024)
        );
        assert_eq!(
            normalize_vram_budget_mb(&json!("512")).unwrap(),
            (false, 512 * 1024 * 1024)
        );
        assert!(normalize_vram_budget_mb(&json!(true)).is_err());
        assert!(normalize_vram_budget_mb(&json!(-1)).is_err());
        assert!(normalize_vram_budget_mb(&json!("bogus")).is_err());

        assert_eq!(normalize_kv_cache_type("F16").unwrap(), "f16");
        assert_eq!(normalize_kv_cache_type(" q8_0 ").unwrap(), "q8_0");
        assert_eq!(normalize_kv_cache_type("q4_0").unwrap(), "q4_0");
        assert!(normalize_kv_cache_type("q2_k").is_err());
    }
}
