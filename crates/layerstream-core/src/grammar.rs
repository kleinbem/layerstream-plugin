//! Pure-Rust JSON Schema & RFC 8259 JSON-object to `llama.cpp` GBNF grammar compiler.
//!
//! Avoids pulling `LLAMA_BUILD_COMMON` into C++ builds while supporting OpenAI
//! `response_format` (`json_object` and `json_schema`) and Pydantic v1/v2 schemas
//! (`$defs`, `definitions`, `$ref`, `anyOf`, `oneOf`, `enum`, `const`, `required`,
//! optional object fields, `additionalProperties`, and bounded arrays).

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

use anyhow::{Result, bail};
use serde_json::{Map, Value};

const BASE_PRIMITIVE_RULES: &str = r#"ws ::= ([ \t\n\r])*
json-char ::= [^"\\\x7F\x00-\x1F] | "\\" (["\\bfnrt/] | "u" [0-9a-fA-F]{4})
json-string ::= "\"" json-char* "\"" ws
json-int ::= ("-"? ([0-9] | [1-9] [0-9]{0,18})) ws
json-number ::= ("-"? ([0-9] | [1-9] [0-9]{0,18})) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws
json-bool ::= ("true" | "false") ws
json-null ::= "null" ws
json-value ::= json-object | json-array | json-string | json-number | json-bool | json-null
json-array ::= "[" ws (json-value ("," ws json-value)*)? "]" ws
json-object ::= "{" ws (json-string ":" ws json-value ("," ws json-string ":" ws json-value)*)? "}" ws
"#;

/// Returns a self-contained GBNF grammar constraining output to a valid JSON object (`{...}`).
pub fn json_object_gbnf() -> String {
    format!("root ::= ws json-object\n{BASE_PRIMITIVE_RULES}")
}

/// Compiles a JSON Schema (`serde_json::Value`) into a `llama.cpp` GBNF grammar with `root` start rule.
pub fn json_schema_to_gbnf(schema: &Value) -> Result<String> {
    let mut compiler = SchemaCompiler::new(schema)?;
    let root_expr = compiler.compile_node(schema, "root")?;
    Ok(compiler.finish(&root_expr))
}

struct SchemaCompiler<'a> {
    root_doc: &'a Value,
    rules: BTreeMap<String, String>,
    ref_rules: BTreeMap<String, String>,
    active_refs: BTreeSet<String>,
    counter: usize,
}

impl<'a> SchemaCompiler<'a> {
    fn new(root_doc: &'a Value) -> Result<Self> {
        if !root_doc.is_object() && !root_doc.is_boolean() {
            bail!("JSON schema must be a JSON object or boolean");
        }
        Ok(Self {
            root_doc,
            rules: BTreeMap::new(),
            ref_rules: BTreeMap::new(),
            active_refs: BTreeSet::new(),
            counter: 0,
        })
    }

    fn fresh_rule_name(&mut self, hint: &str) -> String {
        self.counter += 1;
        let clean: String = hint
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() {
                    c.to_ascii_lowercase()
                } else {
                    '-'
                }
            })
            .collect();
        let trimmed = clean.trim_matches('-');
        let stem = if trimmed.is_empty() { "node" } else { trimmed };
        format!("s-{}-{stem}", self.counter)
    }

    fn add_rule(&mut self, hint: &str, expr: String) -> String {
        let name = self.fresh_rule_name(hint);
        self.rules.insert(name.clone(), expr);
        name
    }

    fn finish(self, root_expr: &str) -> String {
        let mut out = String::new();
        let _ = writeln!(&mut out, "root ::= ws {root_expr}");
        for (name, expr) in &self.rules {
            let _ = writeln!(&mut out, "{name} ::= {expr}");
        }
        out.push_str(BASE_PRIMITIVE_RULES);
        out
    }

    fn resolve_ref(&self, reference: &str) -> Result<&'a Value> {
        if reference == "#" {
            return Ok(self.root_doc);
        }
        let Some(pointer) = reference.strip_prefix("#/") else {
            bail!(
                "unsupported external JSON Schema $ref '{reference}' (only local '#/' refs are supported)"
            );
        };
        let mut cur = self.root_doc;
        for raw_seg in pointer.split('/') {
            let seg = raw_seg.replace("~1", "/").replace("~0", "~");
            cur = cur
                .get(&seg)
                .ok_or_else(|| anyhow::anyhow!("unresolved JSON Schema $ref '{reference}'"))?;
        }
        Ok(cur)
    }

    fn compile_node(&mut self, node: &'a Value, hint: &str) -> Result<String> {
        match node {
            Value::Bool(true) => return Ok("json-value".to_string()),
            Value::Bool(false) => bail!("JSON Schema 'false' rejects all values"),
            Value::Object(_) => {}
            _ => bail!("invalid JSON Schema node at '{hint}': expected object or boolean"),
        }
        let obj = node.as_object().unwrap();

        if let Some(ref_str) = obj.get("$ref").and_then(Value::as_str) {
            if self.active_refs.contains(ref_str) {
                bail!("unguarded recursive JSON Schema $ref cycle at '{ref_str}'");
            }
            if let Some(existing) = self.ref_rules.get(ref_str) {
                return Ok(existing.clone());
            }
            let target = self.resolve_ref(ref_str)?;
            let rule_name = self.fresh_rule_name(ref_str.rsplit('/').next().unwrap_or("ref"));
            self.ref_rules
                .insert(ref_str.to_string(), rule_name.clone());
            self.active_refs.insert(ref_str.to_string());
            let expr = self.compile_node(target, &rule_name)?;
            self.active_refs.remove(ref_str);
            self.rules.insert(rule_name.clone(), expr);
            return Ok(rule_name);
        }

        if let Some(const_val) = obj.get("const") {
            let lit = gbnf_json_value_literal(const_val)?;
            return Ok(format!("{lit} ws"));
        }

        if let Some(enum_arr) = obj.get("enum").and_then(Value::as_array) {
            if enum_arr.is_empty() {
                bail!("JSON Schema 'enum' at '{hint}' must not be empty");
            }
            let mut alts = Vec::with_capacity(enum_arr.len());
            for item in enum_arr {
                alts.push(format!("{} ws", gbnf_json_value_literal(item)?));
            }
            let expr = alts.join(" | ");
            return Ok(self.add_rule(hint, expr));
        }

        for comb_key in ["anyOf", "oneOf"] {
            if let Some(branches) = obj.get(comb_key).and_then(Value::as_array) {
                if branches.is_empty() {
                    bail!("JSON Schema '{comb_key}' at '{hint}' must not be empty");
                }
                let mut alts = Vec::with_capacity(branches.len());
                for (idx, br) in branches.iter().enumerate() {
                    alts.push(self.compile_node(br, &format!("{hint}-{idx}"))?);
                }
                let expr = alts.join(" | ");
                return Ok(self.add_rule(hint, expr));
            }
        }

        if let Some(all_of) = obj.get("allOf").and_then(Value::as_array) {
            if all_of.is_empty() {
                bail!("JSON Schema 'allOf' at '{hint}' must not be empty");
            }
            if all_of.len() == 1 {
                return self.compile_node(&all_of[0], hint);
            }
            let merged = merge_all_of(all_of)?;
            return self.compile_owned_object(&merged, hint);
        }

        self.compile_object_map(obj, hint)
    }

    fn compile_owned_object(&mut self, obj: &Map<String, Value>, hint: &str) -> Result<String> {
        let owned = Value::Object(obj.clone());
        // Owned `allOf` merges cannot contain new root `$defs` targets; compile directly if no `$ref`.
        if let Some(ref_str) = obj.get("$ref").and_then(Value::as_str) {
            let target = self.resolve_ref(ref_str)?;
            return self.compile_node(target, hint);
        }
        let mut sub_compiler_expr = String::new();
        let _ = &owned;
        sub_compiler_expr.push_str(&self.compile_object_map_owned(obj, hint)?);
        Ok(sub_compiler_expr)
    }

    fn compile_object_map(&mut self, obj: &'a Map<String, Value>, hint: &str) -> Result<String> {
        if let Some(type_val) = obj.get("type") {
            match type_val {
                Value::String(t) => return self.compile_single_type(t, obj, hint),
                Value::Array(types) => {
                    if types.is_empty() {
                        bail!("JSON Schema 'type' array at '{hint}' must not be empty");
                    }
                    let mut alts = Vec::with_capacity(types.len());
                    for tv in types {
                        let Some(ts) = tv.as_str() else {
                            bail!("JSON Schema 'type' array elements must be strings");
                        };
                        alts.push(self.compile_single_type(ts, obj, hint)?);
                    }
                    let expr = alts.join(" | ");
                    return Ok(self.add_rule(hint, expr));
                }
                _ => bail!("JSON Schema 'type' at '{hint}' must be a string or array of strings"),
            }
        }

        if obj.contains_key("properties") || obj.contains_key("additionalProperties") {
            return self.compile_single_type("object", obj, hint);
        }
        if obj.contains_key("items") || obj.contains_key("prefixItems") {
            return self.compile_single_type("array", obj, hint);
        }

        Ok("json-value".to_string())
    }

    fn compile_object_map_owned(&mut self, obj: &Map<String, Value>, hint: &str) -> Result<String> {
        // Handle merged `allOf` schemas without lifetime ties to `self.root_doc` except via `$ref`.
        let type_str =
            obj.get("type")
                .and_then(Value::as_str)
                .unwrap_or(if obj.contains_key("properties") {
                    "object"
                } else if obj.contains_key("items") {
                    "array"
                } else {
                    "any"
                });
        match type_str {
            "string" => Ok("json-string".to_string()),
            "integer" => Ok("json-int".to_string()),
            "number" => Ok("json-number".to_string()),
            "boolean" => Ok("json-bool".to_string()),
            "null" => Ok("json-null".to_string()),
            "object" => {
                let Some(props) = obj.get("properties").and_then(Value::as_object) else {
                    return Ok("json-object".to_string());
                };
                let required: BTreeSet<&str> = obj
                    .get("required")
                    .and_then(Value::as_array)
                    .map(|arr| arr.iter().filter_map(Value::as_str).collect())
                    .unwrap_or_default();
                let mut prop_rules = Vec::with_capacity(props.len());
                for (k, v) in props {
                    let val_expr = self.compile_owned_value(v, &format!("{hint}-{k}"))?;
                    let kv_expr = format!("{} \":\" ws {}", gbnf_string_token(k), val_expr);
                    let kv_rule = self.add_rule(&format!("{hint}-kv-{k}"), kv_expr);
                    prop_rules.push((kv_rule, required.contains(k.as_str())));
                }
                let body = build_ordered_object_fields(&prop_rules, None);
                Ok(self.add_rule(hint, body))
            }
            _ => Ok("json-value".to_string()),
        }
    }

    fn compile_owned_value(&mut self, val: &Value, hint: &str) -> Result<String> {
        let Some(obj) = val.as_object() else {
            return Ok("json-value".to_string());
        };
        if let Some(r) = obj.get("$ref").and_then(Value::as_str) {
            let target = self.resolve_ref(r)?;
            return self.compile_node(target, hint);
        }
        self.compile_object_map_owned(obj, hint)
    }

    fn compile_single_type(
        &mut self,
        t: &str,
        obj: &'a Map<String, Value>,
        hint: &str,
    ) -> Result<String> {
        match t {
            "string" => self.compile_string(obj, hint),
            "integer" => Ok("json-int".to_string()),
            "number" => Ok("json-number".to_string()),
            "boolean" => Ok("json-bool".to_string()),
            "null" => Ok("json-null".to_string()),
            "array" => self.compile_array(obj, hint),
            "object" => self.compile_object(obj, hint),
            other => bail!("unsupported JSON Schema type '{other}' at '{hint}'"),
        }
    }

    fn compile_string(&mut self, obj: &'a Map<String, Value>, hint: &str) -> Result<String> {
        let min_len = obj
            .get("minLength")
            .and_then(Value::as_u64)
            .unwrap_or(0)
            .min(4096) as usize;
        let max_len = obj
            .get("maxLength")
            .and_then(Value::as_u64)
            .map(|m| m.min(4096) as usize);
        if min_len == 0 && max_len.is_none() {
            return Ok("json-string".to_string());
        }
        if let Some(max_n) = max_len
            && max_n < min_len
        {
            bail!("JSON Schema string at '{hint}' has maxLength ({max_n}) < minLength ({min_len})");
        }
        let expr = match (min_len, max_len) {
            (_, Some(0)) => "\"\\\"\" \"\\\"\" ws".to_string(),
            (m, None) => format!("\"\\\"\" json-char{{{m},}} \"\\\"\" ws"),
            (m, Some(n)) if m == n => format!("\"\\\"\" json-char{{{m}}} \"\\\"\" ws"),
            (m, Some(n)) => format!("\"\\\"\" json-char{{{m},{n}}} \"\\\"\" ws"),
        };
        Ok(self.add_rule(hint, expr))
    }

    fn compile_array(&mut self, obj: &'a Map<String, Value>, hint: &str) -> Result<String> {
        let saved_unguarded = std::mem::take(&mut self.active_refs);
        let res = self.compile_array_inner(obj, hint);
        self.active_refs = saved_unguarded;
        res
    }

    fn compile_array_inner(&mut self, obj: &'a Map<String, Value>, hint: &str) -> Result<String> {
        if let Some(prefix_items) = obj.get("prefixItems").and_then(Value::as_array) {
            let mut parts = Vec::with_capacity(prefix_items.len());
            for (idx, item) in prefix_items.iter().enumerate() {
                parts.push(self.compile_node(item, &format!("{hint}-tuple-{idx}"))?);
            }
            let inner = parts.join(" \",\" ws ");
            let expr = if inner.is_empty() {
                "\"[\" ws \"]\" ws".to_string()
            } else {
                format!("\"[\" ws {inner} \"]\" ws")
            };
            return Ok(self.add_rule(hint, expr));
        }

        let item_expr = match obj.get("items") {
            Some(item_schema) => self.compile_node(item_schema, &format!("{hint}-item"))?,
            None => "json-value".to_string(),
        };

        let min_items = obj
            .get("minItems")
            .and_then(Value::as_u64)
            .unwrap_or(0)
            .min(64) as usize;
        let max_items = obj
            .get("maxItems")
            .and_then(Value::as_u64)
            .map(|m| m.min(64) as usize);

        if let Some(max_n) = max_items
            && max_n < min_items
        {
            bail!("JSON Schema array at '{hint}' has maxItems ({max_n}) < minItems ({min_items})");
        }

        let expr = match (min_items, max_items) {
            (_, Some(0)) => "\"[\" ws \"]\" ws".to_string(),
            (0, None) => format!("\"[\" ws ({item_expr} (\",\" ws {item_expr})*)? \"]\" ws"),
            (1, None) => format!("\"[\" ws {item_expr} (\",\" ws {item_expr})* \"]\" ws"),
            (min_n, None) => {
                let tail_min = min_n - 1;
                format!("\"[\" ws {item_expr} (\",\" ws {item_expr}){{{tail_min},}} \"]\" ws")
            }
            (0, Some(1)) => format!("\"[\" ws ({item_expr})? \"]\" ws"),
            (0, Some(max_n)) => {
                let tail_max = max_n - 1;
                format!("\"[\" ws ({item_expr} (\",\" ws {item_expr}){{0,{tail_max}}})? \"]\" ws")
            }
            (min_n, Some(max_n)) if min_n == max_n => {
                if min_n == 1 {
                    format!("\"[\" ws {item_expr} \"]\" ws")
                } else {
                    let tail = min_n - 1;
                    format!("\"[\" ws {item_expr} (\",\" ws {item_expr}){{{tail}}} \"]\" ws")
                }
            }
            (min_n, Some(max_n)) => {
                let tail_min = min_n - 1;
                let tail_max = max_n - 1;
                format!(
                    "\"[\" ws {item_expr} (\",\" ws {item_expr}){{{tail_min},{tail_max}}} \"]\" ws"
                )
            }
        };
        Ok(self.add_rule(hint, expr))
    }

    fn compile_object(&mut self, obj: &'a Map<String, Value>, hint: &str) -> Result<String> {
        let saved_unguarded = std::mem::take(&mut self.active_refs);
        let res = self.compile_object_inner(obj, hint);
        self.active_refs = saved_unguarded;
        res
    }

    fn compile_object_inner(&mut self, obj: &'a Map<String, Value>, hint: &str) -> Result<String> {
        let props_opt = obj.get("properties").and_then(Value::as_object);
        let addl = obj.get("additionalProperties");

        let addl_kv_rule = match addl {
            None | Some(Value::Bool(false)) => {
                if props_opt.is_none() {
                    if matches!(addl, Some(Value::Bool(false))) {
                        return Ok(self.add_rule(hint, "\"{\" ws \"}\" ws".to_string()));
                    }
                    return Ok("json-object".to_string());
                }
                None
            }
            Some(Value::Bool(true)) => {
                if props_opt.is_none() {
                    return Ok("json-object".to_string());
                }
                Some(self.add_rule(
                    &format!("{hint}-addl"),
                    "json-string \":\" ws json-value".to_string(),
                ))
            }
            Some(schema_val @ Value::Object(_)) => {
                let val_expr = self.compile_node(schema_val, &format!("{hint}-addl-val"))?;
                let kv = format!("json-string \":\" ws {val_expr}");
                let kv_rule = self.add_rule(&format!("{hint}-addl"), kv);
                if props_opt.is_none() {
                    let map_expr =
                        format!("\"{{{{\" ws ({kv_rule} (\",\" ws {kv_rule})*)? \"}}}}\" ws")
                            .replace("{{", "{")
                            .replace("}}", "}");
                    return Ok(self.add_rule(hint, map_expr));
                }
                Some(kv_rule)
            }
            Some(_) => bail!("invalid 'additionalProperties' in JSON Schema at '{hint}'"),
        };

        let props = props_opt.unwrap();
        let required_list: Vec<&str> = obj
            .get("required")
            .and_then(Value::as_array)
            .map(|arr| arr.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();
        let required_set: BTreeSet<&str> = required_list.iter().copied().collect();

        // Preserve schema property order while ensuring any required key missing from `properties`
        // is still enforced as an unconstrained `json-value` entry.
        let mut prop_rules: Vec<(String, bool)> = Vec::with_capacity(props.len());
        for (prop_name, prop_schema) in props {
            let val_expr = self.compile_node(prop_schema, &format!("{hint}-{prop_name}"))?;
            let kv_expr = format!("{} \":\" ws {}", gbnf_string_token(prop_name), val_expr);
            let kv_rule = self.add_rule(&format!("{hint}-kv-{prop_name}"), kv_expr);
            prop_rules.push((kv_rule, required_set.contains(prop_name.as_str())));
        }
        for req_name in required_list {
            if !props.contains_key(req_name) {
                let kv_expr = format!("{} \":\" ws json-value", gbnf_string_token(req_name));
                let kv_rule = self.add_rule(&format!("{hint}-kv-{req_name}"), kv_expr);
                prop_rules.push((kv_rule, true));
            }
        }

        let obj_expr = build_ordered_object_fields(&prop_rules, addl_kv_rule.as_deref());
        Ok(self.add_rule(hint, obj_expr))
    }
}

/// Builds a compact right-factored GBNF expression for ordered object properties `(kv_rule, is_required)`
/// where optional properties may be omitted without trailing/leading comma errors.
fn build_ordered_object_fields(props: &[(String, bool)], addl_kv: Option<&str>) -> String {
    if props.is_empty() {
        return match addl_kv {
            Some(akv) => format!("\"{{\" ws ({akv} (\",\" ws {akv})*)? \"}}\" ws"),
            None => "\"{\" ws \"}\" ws".to_string(),
        };
    }

    let addl_tail = addl_kv
        .map(|akv| format!(" (\",\" ws {akv})*"))
        .unwrap_or_default();

    // First required index (if any).
    let first_req = props.iter().position(|(_, req)| *req);

    // Build per-start-index suffix cleanly so optional leading fields can start the object without a leading comma.
    let suffix_after = |start_idx: usize| -> String {
        let mut s = String::new();
        for (rule, is_req) in &props[start_idx + 1..] {
            if *is_req {
                let _ = write!(&mut s, " \",\" ws {rule}");
            } else {
                let _ = write!(&mut s, " (\",\" ws {rule})?");
            }
        }
        s.push_str(&addl_tail);
        s
    };

    let limit = first_req.unwrap_or(props.len() - 1);
    let mut start_alts = Vec::with_capacity(limit + 1);
    for (idx, (rule, _)) in props.iter().enumerate().take(limit + 1) {
        let tail = suffix_after(idx);
        start_alts.push(format!("{rule}{tail}"));
    }

    if first_req.is_none() {
        if let Some(akv) = addl_kv {
            start_alts.push(format!("{akv} (\",\" ws {akv})*"));
        }
        format!("\"{{\" ws ({})? \"}}\" ws", start_alts.join(" | "))
    } else if start_alts.len() == 1 {
        format!("\"{{\" ws {} \"}}\" ws", start_alts[0])
    } else {
        format!("\"{{\" ws ({}) \"}}\" ws", start_alts.join(" | "))
    }
}

fn merge_all_of(items: &[Value]) -> Result<Map<String, Value>> {
    let mut merged = Map::new();
    let mut merged_props = Map::new();
    let mut merged_req: Vec<Value> = Vec::new();

    for item in items {
        let Some(obj) = item.as_object() else {
            bail!("'allOf' entries must be JSON Schema objects");
        };
        for (k, v) in obj {
            match k.as_str() {
                "properties" => {
                    if let Some(p) = v.as_object() {
                        for (pk, pv) in p {
                            merged_props.insert(pk.clone(), pv.clone());
                        }
                    }
                }
                "required" => {
                    if let Some(arr) = v.as_array() {
                        for r in arr {
                            if !merged_req.contains(r) {
                                merged_req.push(r.clone());
                            }
                        }
                    }
                }
                _ => {
                    merged.insert(k.clone(), v.clone());
                }
            }
        }
    }
    if !merged_props.is_empty() {
        merged.insert("properties".to_string(), Value::Object(merged_props));
    }
    if !merged_req.is_empty() {
        merged.insert("required".to_string(), Value::Array(merged_req));
    }
    Ok(merged)
}

/// Formats a JSON string token (`"foo"`) as a GBNF terminal literal (`"\"foo\""` + optional ws when requested).
fn gbnf_string_token(s: &str) -> String {
    let json_encoded = serde_json::to_string(s).unwrap_or_else(|_| "\"\"".to_string());
    format!("{} ws", gbnf_literal(&json_encoded))
}

fn gbnf_json_value_literal(v: &Value) -> Result<String> {
    let serialized = serde_json::to_string(v)?;
    Ok(gbnf_literal(&serialized))
}

fn gbnf_literal(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len() + 4);
    out.push('"');
    for ch in raw.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(&mut out, "\\x{:02X}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn compiles_pydantic_style_schema_with_defs_optional_and_enum() {
        let schema = json!({
            "$defs": {
                "Step": {
                    "type": "object",
                    "properties": {
                        "explanation": {"type": "string"},
                        "output": {"type": "string"}
                    },
                    "required": ["explanation", "output"],
                    "additionalProperties": false
                }
            },
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "items": {"$ref": "#/$defs/Step"},
                    "minItems": 1
                },
                "final_answer": {"type": "number"},
                "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                "notes": {"anyOf": [{"type": "string"}, {"type": "null"}]}
            },
            "required": ["steps", "final_answer"]
        });

        let gbnf = json_schema_to_gbnf(&schema).unwrap();
        assert!(gbnf.starts_with("root ::= ws "));
        assert!(gbnf.contains("\\\"steps\\\""));
        assert!(gbnf.contains("\\\"final_answer\\\""));
        assert!(gbnf.contains("\\\"low\\\""));
        assert!(gbnf.contains("json-null"));
    }

    #[test]
    fn supports_guarded_recursive_schema_and_rejects_unguarded_ref_cycle() {
        let guarded_tree = json!({
            "$defs": {
                "Node": {
                    "type": "object",
                    "properties": {
                        "next": {"$ref": "#/$defs/Node"}
                    }
                }
            },
            "$ref": "#/$defs/Node"
        });
        let gbnf = json_schema_to_gbnf(&guarded_tree).unwrap();
        assert!(gbnf.contains("\\\"next\\\""));

        let unguarded_cycle = json!({
            "$defs": {
                "A": {"anyOf": [{"$ref": "#/$defs/B"}, {"type": "string"}]},
                "B": {"$ref": "#/$defs/A"}
            },
            "$ref": "#/$defs/A"
        });
        let err = json_schema_to_gbnf(&unguarded_cycle).unwrap_err();
        assert!(err.to_string().contains("cycle"));
    }
}
