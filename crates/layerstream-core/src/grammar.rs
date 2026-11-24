//! Pure-Rust JSON Schema & RFC 8259 JSON-object to `llama.cpp` GBNF grammar compiler.
//!
//! Avoids pulling `LLAMA_BUILD_COMMON` into C++ builds while supporting OpenAI
//! `response_format` (`json_object` and `json_schema`) and Pydantic v1/v2 schemas
//! (`$defs`, `definitions`, `$ref`, `anyOf`, `oneOf`, `allOf`, `enum`, `const`,
//! `required`, optional object fields, `additionalProperties`, bounded strings,
//! `prefixItems`, and bounded arrays).

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

use anyhow::{Result, bail};
use serde_json::{Map, Value};

const MAX_STRING_BOUND: u64 = 4096;
const MAX_ARRAY_BOUND: u64 = 64;

const BASE_PRIMITIVE_RULES: &str = r#"ws ::= ([ \t\n\r])*
json-hex4-bmp ::= [0-9a-ce-fA-CE-F] [0-9a-fA-F]{3} | [dD] [0-7] [0-9a-fA-F]{2}
json-surrogate-pair ::= "\\" "u" [dD] [89aAbB] [0-9a-fA-F]{2} "\\" "u" [dD] [c-fC-F] [0-9a-fA-F]{2}
json-char ::= [^"\\\x7F\x00-\x1F] | "\\" (["\\bfnrt/] | "u" json-hex4-bmp) | json-surrogate-pair
json-string ::= "\"" json-char* "\"" ws
json-int ::= ("-"? ("0" | [1-9] [0-9]*)) ws
json-number ::= ("-"? ("0" | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws
json-nonint-number ::= ("-"? ("0" | [1-9] [0-9]*)) "." [0-9]* [1-9] [0-9]* ws
json-bool ::= ("true" | "false") ws
json-null ::= "null" ws
json-value ::= json-object | json-array | json-string | json-number | json-bool | json-null
json-array ::= "[" ws (json-value ("," ws json-value)*)? "]" ws
json-object ::= "{" ws (json-string ":" ws json-value ("," ws json-string ":" ws json-value)*)? "}" ws
"#;

const ANNOTATION_KEYWORDS: &[&str] = &[
    "$schema",
    "$id",
    "$comment",
    "$defs",
    "definitions",
    "title",
    "description",
    "default",
    "examples",
    "readOnly",
    "writeOnly",
    "deprecated",
];

const UNSUPPORTED_VALIDATION_KEYWORDS: &[&str] = &[
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "pattern",
    "format",
    "contentEncoding",
    "contentMediaType",
    "contentSchema",
    "uniqueItems",
    "contains",
    "minContains",
    "maxContains",
    "additionalItems",
    "minProperties",
    "maxProperties",
    "patternProperties",
    "propertyNames",
    "dependencies",
    "dependentRequired",
    "dependentSchemas",
    "unevaluatedProperties",
    "unevaluatedItems",
    "not",
    "if",
    "then",
    "else",
];

/// Returns a self-contained GBNF grammar constraining output to a valid JSON object (`{...}`).
pub fn json_object_gbnf() -> String {
    format!("root ::= ws json-object\n{BASE_PRIMITIVE_RULES}")
}

/// Compiles a JSON Schema (`serde_json::Value`) into a `llama.cpp` GBNF grammar with `root` start rule.
pub fn json_schema_to_gbnf(schema: &Value) -> Result<String> {
    let mut compiler = SchemaCompiler::new(schema)?;
    let Some(root_expr) = compiler.compile_value(schema, "root")? else {
        bail!("JSON Schema rejects all values (unsatisfiable)");
    };
    compiler.check_no_zero_input_cycles(&root_expr)?;
    Ok(compiler.finish(&root_expr))
}

fn is_annotation_key(k: &str) -> bool {
    ANNOTATION_KEYWORDS.contains(&k) || k.starts_with("x-")
}

fn has_validation_keys(obj: &Map<String, Value>) -> bool {
    obj.keys().any(|k| !is_annotation_key(k))
}

fn validate_supported_keywords(obj: &Map<String, Value>, hint: &str) -> Result<()> {
    for k in obj.keys() {
        if UNSUPPORTED_VALIDATION_KEYWORDS.contains(&k.as_str()) {
            bail!("unsupported JSON Schema constraint '{k}' at '{hint}'");
        }
    }
    Ok(())
}

fn parse_nonneg_bound(
    obj: &Map<String, Value>,
    key: &str,
    max_allowed: u64,
    hint: &str,
) -> Result<Option<usize>> {
    let Some(val) = obj.get(key) else {
        return Ok(None);
    };
    let Some(n) = val.as_u64() else {
        bail!("JSON Schema '{key}' at '{hint}' must be a non-negative integer");
    };
    if n > max_allowed {
        bail!(
            "JSON Schema '{key}' ({n}) at '{hint}' exceeds maximum supported GBNF bound ({max_allowed})"
        );
    }
    Ok(Some(n as usize))
}

fn json_values_schema_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(na), Value::Number(nb)) => {
            if na == nb {
                return true;
            }
            match (na.as_f64(), nb.as_f64()) {
                (Some(fa), Some(fb)) => fa == fb,
                _ => false,
            }
        }
        (Value::Array(aa), Value::Array(ab)) => {
            aa.len() == ab.len()
                && aa
                    .iter()
                    .zip(ab.iter())
                    .all(|(x, y)| json_values_schema_equal(x, y))
        }
        (Value::Object(oa), Value::Object(ob)) => {
            oa.len() == ob.len()
                && oa
                    .iter()
                    .all(|(k, vx)| ob.get(k).is_some_and(|vy| json_values_schema_equal(vx, vy)))
        }
        _ => a == b,
    }
}

fn value_matches_primitive_type(val: &Value, t: &str) -> bool {
    match t {
        "null" => val.is_null(),
        "boolean" => val.is_boolean(),
        "string" => val.is_string(),
        "array" => val.is_array(),
        "object" => val.is_object(),
        "number" => val.is_number(),
        "integer" => match val {
            Value::Number(n) => {
                n.is_i64()
                    || n.is_u64()
                    || n.as_f64()
                        .is_some_and(|f| f.is_finite() && f.fract() == 0.0)
            }
            _ => false,
        },
        _ => false,
    }
}

fn value_matches_local_constraints(val: &Value, obj: &Map<String, Value>) -> Result<bool> {
    if let Some(tval) = obj.get("type") {
        let ok = match tval {
            Value::String(ts) => value_matches_primitive_type(val, ts),
            Value::Array(arr) => arr
                .iter()
                .filter_map(Value::as_str)
                .any(|ts| value_matches_primitive_type(val, ts)),
            _ => bail!("JSON Schema 'type' must be a string or array of strings"),
        };
        if !ok {
            return Ok(false);
        }
    }
    if let Value::String(s) = val {
        let char_len = s.chars().count();
        if let Some(min_l) = parse_nonneg_bound(obj, "minLength", MAX_STRING_BOUND, "literal")?
            && char_len < min_l
        {
            return Ok(false);
        }
        if let Some(max_l) = parse_nonneg_bound(obj, "maxLength", MAX_STRING_BOUND, "literal")?
            && char_len > max_l
        {
            return Ok(false);
        }
    }
    if let Value::Array(arr) = val {
        if let Some(min_i) = parse_nonneg_bound(obj, "minItems", MAX_ARRAY_BOUND, "literal")?
            && arr.len() < min_i
        {
            return Ok(false);
        }
        if let Some(max_i) = parse_nonneg_bound(obj, "maxItems", MAX_ARRAY_BOUND, "literal")?
            && arr.len() > max_i
        {
            return Ok(false);
        }
    }
    if let Value::Object(map) = val
        && let Some(req_arr) = obj.get("required").and_then(Value::as_array)
    {
        for r in req_arr.iter().filter_map(Value::as_str) {
            if !map.contains_key(r) {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

struct SchemaCompiler<'a> {
    root_doc: &'a Value,
    rules: BTreeMap<String, String>,
    ref_rules: BTreeMap<String, Option<String>>,
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

    fn resolve_ref(&self, reference: &str) -> Result<Value> {
        if reference == "#" {
            return Ok(self.root_doc.clone());
        }
        let Some(pointer) = reference.strip_prefix("#/") else {
            bail!(
                "unsupported external JSON Schema $ref '{reference}' (only local '#/' refs are supported)"
            );
        };
        let mut cur = self.root_doc;
        for raw_seg in pointer.split('/') {
            let seg = raw_seg.replace("~1", "/").replace("~0", "~");
            cur = match cur {
                Value::Object(map) => map.get(&seg),
                Value::Array(arr) => seg.parse::<usize>().ok().and_then(|idx| arr.get(idx)),
                _ => None,
            }
            .ok_or_else(|| anyhow::anyhow!("unresolved JSON Schema $ref '{reference}'"))?;
        }
        Ok(cur.clone())
    }

    fn deref_shallow(&self, val: &Value) -> Result<Value> {
        let mut seen = BTreeSet::new();
        self.deref_shallow_with_seen(val, &mut seen)
    }

    fn deref_shallow_with_seen(&self, val: &Value, seen: &mut BTreeSet<String>) -> Result<Value> {
        let mut cur = val.clone();
        let mut pushed = Vec::new();
        let out = (|| -> Result<Value> {
            while let Value::Object(obj) = &cur {
                let Some(ref_val) = obj.get("$ref") else {
                    break;
                };
                let Some(ref_str) = ref_val.as_str() else {
                    bail!("JSON Schema '$ref' must be a string");
                };
                if !seen.insert(ref_str.to_string()) {
                    bail!("unguarded recursive JSON Schema $ref cycle at '{ref_str}'");
                }
                pushed.push(ref_str.to_string());
                let target = self.resolve_ref(ref_str)?;
                let mut sib = obj.clone();
                sib.remove("$ref");
                cur = if has_validation_keys(&sib) {
                    match self.intersect_schemas_with_seen(
                        &target,
                        &Value::Object(sib),
                        "ref",
                        seen,
                    )? {
                        Some(m) => m,
                        None => return Ok(Value::Bool(false)),
                    }
                } else {
                    target
                };
            }
            Ok(cur)
        })();
        for r in pushed {
            seen.remove(&r);
        }
        out
    }

    /// Compiles `node`, returning `Ok(None)` when the schema branch is unsatisfiable (`false`).
    fn compile_value(&mut self, node: &Value, hint: &str) -> Result<Option<String>> {
        match node {
            Value::Bool(true) => return Ok(Some("json-value".to_string())),
            Value::Bool(false) => return Ok(None),
            Value::Object(_) => {}
            _ => bail!("invalid JSON Schema node at '{hint}': expected object or boolean"),
        }
        let obj = node.as_object().unwrap();
        validate_supported_keywords(obj, hint)?;

        if let Some(ref_val) = obj.get("$ref") {
            let Some(ref_str) = ref_val.as_str() else {
                bail!("JSON Schema '$ref' at '{hint}' must be a string");
            };
            let mut sib = obj.clone();
            sib.remove("$ref");
            if !has_validation_keys(&sib) {
                if self.active_refs.contains(ref_str) {
                    bail!("unguarded recursive JSON Schema $ref cycle at '{ref_str}'");
                }
                if let Some(cached) = self.ref_rules.get(ref_str) {
                    return Ok(cached.clone());
                }
                let target = self.resolve_ref(ref_str)?;
                let rule_name = self.fresh_rule_name(ref_str.rsplit('/').next().unwrap_or("ref"));
                self.ref_rules
                    .insert(ref_str.to_string(), Some(rule_name.clone()));
                self.active_refs.insert(ref_str.to_string());
                let compiled = self.compile_value(&target, &rule_name)?;
                self.active_refs.remove(ref_str);
                return match compiled {
                    Some(expr) => {
                        self.rules.insert(rule_name.clone(), expr);
                        Ok(Some(rule_name))
                    }
                    None => {
                        self.ref_rules.insert(ref_str.to_string(), None);
                        Ok(None)
                    }
                };
            }

            if self.active_refs.contains(ref_str) {
                bail!("unguarded recursive JSON Schema $ref cycle at '{ref_str}'");
            }
            let target = self.resolve_ref(ref_str)?;
            self.active_refs.insert(ref_str.to_string());
            let merged_opt = self.intersect_schemas(&target, &Value::Object(sib), hint)?;
            let res = match merged_opt {
                Some(merged) => self.compile_value(&merged, hint),
                None => Ok(None),
            };
            self.active_refs.remove(ref_str);
            return res;
        }

        if let Some(all_of_val) = obj.get("allOf") {
            let Some(all_of) = all_of_val.as_array() else {
                bail!("JSON Schema 'allOf' at '{hint}' must be an array");
            };
            if all_of.is_empty() {
                bail!("JSON Schema 'allOf' at '{hint}' must not be empty");
            }
            let mut sib = obj.clone();
            sib.remove("allOf");
            let mut acc = if has_validation_keys(&sib) {
                Value::Object(sib)
            } else {
                Value::Bool(true)
            };
            for item in all_of {
                match self.intersect_schemas(&acc, item, hint)? {
                    Some(next) => acc = next,
                    None => return Ok(None),
                }
            }
            return self.compile_value(&acc, hint);
        }

        if let Some(any_of_val) = obj.get("anyOf") {
            let Some(branches) = any_of_val.as_array() else {
                bail!("JSON Schema 'anyOf' at '{hint}' must be an array");
            };
            if branches.is_empty() {
                bail!("JSON Schema 'anyOf' at '{hint}' must not be empty");
            }
            let mut sib = obj.clone();
            sib.remove("anyOf");
            let has_sib = has_validation_keys(&sib);
            let sib_val = Value::Object(sib);

            let mut alts = Vec::with_capacity(branches.len());
            for (idx, br) in branches.iter().enumerate() {
                let eff_br = if has_sib {
                    match self.intersect_schemas(&sib_val, br, &format!("{hint}-{idx}"))? {
                        Some(m) => m,
                        None => continue,
                    }
                } else {
                    br.clone()
                };
                if let Some(compiled) = self.compile_value(&eff_br, &format!("{hint}-{idx}"))? {
                    alts.push(compiled);
                }
            }
            if alts.is_empty() {
                return Ok(None);
            }
            if alts.len() == 1 {
                return Ok(alts.into_iter().next());
            }
            return Ok(Some(self.add_rule(hint, alts.join(" | "))));
        }

        if let Some(one_of_val) = obj.get("oneOf") {
            let Some(branches) = one_of_val.as_array() else {
                bail!("JSON Schema 'oneOf' at '{hint}' must be an array");
            };
            if branches.is_empty() {
                bail!("JSON Schema 'oneOf' at '{hint}' must not be empty");
            }
            let mut sib = obj.clone();
            sib.remove("oneOf");
            let has_sib = has_validation_keys(&sib);
            let sib_val = Value::Object(sib);

            let mut eff_branches = Vec::with_capacity(branches.len());
            for (idx, br) in branches.iter().enumerate() {
                let eff_br = if has_sib {
                    match self.intersect_schemas(&sib_val, br, &format!("{hint}-{idx}"))? {
                        Some(m) => m,
                        None => continue,
                    }
                } else {
                    br.clone()
                };
                if !matches!(eff_br, Value::Bool(false)) {
                    eff_branches.push(eff_br);
                }
            }
            return self.compile_one_of(&eff_branches, hint);
        }

        if let Some(const_val) = obj.get("const") {
            if let Some(enum_arr) = obj.get("enum").and_then(Value::as_array)
                && !enum_arr
                    .iter()
                    .any(|e| json_values_schema_equal(e, const_val))
            {
                return Ok(None);
            }
            if !value_matches_local_constraints(const_val, obj)? {
                return Ok(None);
            }
            let lit = gbnf_json_value_literal(const_val)?;
            return Ok(Some(format!("{lit} ws")));
        }

        if let Some(enum_val) = obj.get("enum") {
            let Some(enum_arr) = enum_val.as_array() else {
                bail!("JSON Schema 'enum' at '{hint}' must be an array");
            };
            if enum_arr.is_empty() {
                bail!("JSON Schema 'enum' at '{hint}' must not be empty");
            }
            let mut alts = Vec::with_capacity(enum_arr.len());
            for item in enum_arr {
                if value_matches_local_constraints(item, obj)? {
                    alts.push(format!("{} ws", gbnf_json_value_literal(item)?));
                }
            }
            if alts.is_empty() {
                return Ok(None);
            }
            return Ok(Some(self.add_rule(hint, alts.join(" | "))));
        }

        self.compile_object_map(obj, hint)
    }

    fn compile_one_of(&mut self, branches: &[Value], hint: &str) -> Result<Option<String>> {
        if branches.is_empty() {
            return Ok(None);
        }
        if branches.len() == 1 {
            return self.compile_value(&branches[0], hint);
        }

        let mut resolved = Vec::with_capacity(branches.len());
        for br in branches {
            let d = self.deref_shallow(br)?;
            if let Some(obj) = d.as_object() {
                validate_supported_keywords(obj, hint)?;
            }
            if !matches!(d, Value::Bool(false)) {
                resolved.push((br.clone(), d));
            }
        }
        if resolved.is_empty() {
            return Ok(None);
        }
        if resolved.len() == 1 {
            return self.compile_value(&resolved[0].0, hint);
        }

        // Case 1: All branches are pure literal (`const` / `enum`) sets -> keep values matching exactly 1 branch.
        if let Some(literal_sets) = resolved
            .iter()
            .map(|(_, d)| extract_pure_literals(d))
            .collect::<Option<Result<Vec<Vec<Value>>>>>()
        {
            let sets = literal_sets?;
            let mut exclusive = Vec::new();
            for (i, set_i) in sets.iter().enumerate() {
                for cand in set_i {
                    let match_count = sets
                        .iter()
                        .filter(|s| s.iter().any(|x| json_values_schema_equal(x, cand)))
                        .count();
                    if match_count == 1
                        && !exclusive.iter().any(|x| json_values_schema_equal(x, cand))
                        && resolved[i].1.as_object().is_none_or(|o| {
                            value_matches_local_constraints(cand, o).unwrap_or(false)
                        })
                    {
                        exclusive.push(cand.clone());
                    }
                }
            }
            if exclusive.is_empty() {
                return Ok(None);
            }
            let mut alts = Vec::with_capacity(exclusive.len());
            for v in &exclusive {
                alts.push(format!("{} ws", gbnf_json_value_literal(v)?));
            }
            return Ok(Some(self.add_rule(hint, alts.join(" | "))));
        }

        // Case 2: Handle `(plain number XOR plain integer)` -> `json-nonint-number` alongside disjoint branches.
        let mut plain_num_idxs = Vec::new();
        let mut plain_int_idxs = Vec::new();
        for (idx, (_, d)) in resolved.iter().enumerate() {
            if is_plain_single_type(d, "number") {
                plain_num_idxs.push(idx);
            } else if is_plain_single_type(d, "integer") {
                plain_int_idxs.push(idx);
            }
        }
        if plain_num_idxs.len() == 1 && plain_int_idxs.len() == 1 {
            let num_i = plain_num_idxs[0];
            let int_i = plain_int_idxs[0];
            let mut others_disjoint = true;
            for (idx, (_, d)) in resolved.iter().enumerate() {
                if idx == num_i || idx == int_i {
                    continue;
                }
                if !schemas_provably_disjoint(d, &resolved[num_i].1) {
                    others_disjoint = false;
                    break;
                }
            }
            if others_disjoint {
                let mut rem_schemas: Vec<(Value, Value)> = resolved
                    .iter()
                    .enumerate()
                    .filter(|(idx, _)| *idx != num_i && *idx != int_i)
                    .map(|(_, pair)| pair.clone())
                    .collect();
                for i in 0..rem_schemas.len() {
                    for j in (i + 1)..rem_schemas.len() {
                        if !schemas_provably_disjoint(&rem_schemas[i].1, &rem_schemas[j].1) {
                            bail!(
                                "JSON Schema 'oneOf' at '{hint}' has overlapping branches that cannot be proven mutually exclusive"
                            );
                        }
                    }
                }
                let mut alts = vec!["json-nonint-number".to_string()];
                for (idx, (orig, _)) in rem_schemas.drain(..).enumerate() {
                    if let Some(r) = self.compile_value(&orig, &format!("{hint}-oneof-{idx}"))? {
                        alts.push(r);
                    }
                }
                return Ok(Some(self.add_rule(hint, alts.join(" | "))));
            }
        }

        // Case 3: Verify pairwise disjointness across all branches.
        for i in 0..resolved.len() {
            for j in (i + 1)..resolved.len() {
                if !schemas_provably_disjoint(&resolved[i].1, &resolved[j].1) {
                    bail!(
                        "JSON Schema 'oneOf' at '{hint}' has overlapping branches that cannot be proven mutually exclusive"
                    );
                }
            }
        }

        let mut alts = Vec::with_capacity(resolved.len());
        for (idx, (orig, _)) in resolved.iter().enumerate() {
            if let Some(r) = self.compile_value(orig, &format!("{hint}-{idx}"))? {
                alts.push(r);
            }
        }
        if alts.is_empty() {
            return Ok(None);
        }
        if alts.len() == 1 {
            return Ok(alts.into_iter().next());
        }
        Ok(Some(self.add_rule(hint, alts.join(" | "))))
    }

    fn intersect_schemas(&self, a: &Value, b: &Value, hint: &str) -> Result<Option<Value>> {
        let mut seen = BTreeSet::new();
        self.intersect_schemas_with_seen(a, b, hint, &mut seen)
    }

    fn intersect_schemas_with_seen(
        &self,
        a: &Value,
        b: &Value,
        hint: &str,
        seen: &mut BTreeSet<String>,
    ) -> Result<Option<Value>> {
        let da = self.deref_shallow_with_seen(a, seen)?;
        let db = self.deref_shallow_with_seen(b, seen)?;
        match (&da, &db) {
            (Value::Bool(false), _) | (_, Value::Bool(false)) => return Ok(None),
            (Value::Bool(true), other) => return Ok(Some(other.clone())),
            (other, Value::Bool(true)) => return Ok(Some(other.clone())),
            (Value::Object(_), Value::Object(_)) => {}
            _ => bail!("invalid JSON Schema node during intersection at '{hint}'"),
        }
        let oa = da.as_object().unwrap();
        let ob = db.as_object().unwrap();
        validate_supported_keywords(oa, hint)?;
        validate_supported_keywords(ob, hint)?;

        // Flatten `allOf` on either side first.
        if let Some(arr) = oa.get("allOf").and_then(Value::as_array) {
            if arr.is_empty() {
                bail!("JSON Schema 'allOf' at '{hint}' must not be empty");
            }
            let mut base = oa.clone();
            base.remove("allOf");
            let mut cur = if has_validation_keys(&base) {
                Value::Object(base)
            } else {
                Value::Bool(true)
            };
            for item in arr {
                let Some(next) = self.intersect_schemas(&cur, item, hint)? else {
                    return Ok(None);
                };
                cur = next;
            }
            return self.intersect_schemas(&cur, &db, hint);
        }
        if let Some(arr) = ob.get("allOf").and_then(Value::as_array) {
            if arr.is_empty() {
                bail!("JSON Schema 'allOf' at '{hint}' must not be empty");
            }
            let mut base = ob.clone();
            base.remove("allOf");
            let mut cur = if has_validation_keys(&base) {
                Value::Object(base)
            } else {
                Value::Bool(true)
            };
            for item in arr {
                let Some(next) = self.intersect_schemas(&cur, item, hint)? else {
                    return Ok(None);
                };
                cur = next;
            }
            return self.intersect_schemas(&da, &cur, hint);
        }

        // Distribute over `anyOf`.
        if let Some(arr) = oa.get("anyOf").and_then(Value::as_array) {
            if arr.is_empty() {
                bail!("JSON Schema 'anyOf' at '{hint}' must not be empty");
            }
            let mut base = oa.clone();
            base.remove("anyOf");
            let base_val = if has_validation_keys(&base) {
                Value::Object(base)
            } else {
                Value::Bool(true)
            };
            let mut out_branches = Vec::new();
            for br in arr {
                if let Some(step1) = self.intersect_schemas(&base_val, br, hint)?
                    && let Some(step2) = self.intersect_schemas(&step1, &db, hint)?
                {
                    out_branches.push(step2);
                }
            }
            if out_branches.is_empty() {
                return Ok(None);
            }
            let mut res = Map::new();
            res.insert("anyOf".to_string(), Value::Array(out_branches));
            return Ok(Some(Value::Object(res)));
        }
        if let Some(arr) = ob.get("anyOf").and_then(Value::as_array) {
            if arr.is_empty() {
                bail!("JSON Schema 'anyOf' at '{hint}' must not be empty");
            }
            let mut base = ob.clone();
            base.remove("anyOf");
            let base_val = if has_validation_keys(&base) {
                Value::Object(base)
            } else {
                Value::Bool(true)
            };
            let mut out_branches = Vec::new();
            for br in arr {
                if let Some(step1) = self.intersect_schemas(&base_val, br, hint)?
                    && let Some(step2) = self.intersect_schemas(&da, &step1, hint)?
                {
                    out_branches.push(step2);
                }
            }
            if out_branches.is_empty() {
                return Ok(None);
            }
            let mut res = Map::new();
            res.insert("anyOf".to_string(), Value::Array(out_branches));
            return Ok(Some(Value::Object(res)));
        }

        // Distribute over `oneOf` (since `(X1 xor X2) and Y == (X1 and Y) xor (X2 and Y)`).
        if let Some(arr) = oa.get("oneOf").and_then(Value::as_array) {
            if arr.is_empty() {
                bail!("JSON Schema 'oneOf' at '{hint}' must not be empty");
            }
            let mut base = oa.clone();
            base.remove("oneOf");
            let base_val = if has_validation_keys(&base) {
                Value::Object(base)
            } else {
                Value::Bool(true)
            };
            let mut out_branches = Vec::new();
            for br in arr {
                if let Some(step1) = self.intersect_schemas(&base_val, br, hint)?
                    && let Some(step2) = self.intersect_schemas(&step1, &db, hint)?
                {
                    out_branches.push(step2);
                }
            }
            if out_branches.is_empty() {
                return Ok(None);
            }
            let mut res = Map::new();
            res.insert("oneOf".to_string(), Value::Array(out_branches));
            return Ok(Some(Value::Object(res)));
        }
        if let Some(arr) = ob.get("oneOf").and_then(Value::as_array) {
            if arr.is_empty() {
                bail!("JSON Schema 'oneOf' at '{hint}' must not be empty");
            }
            let mut base = ob.clone();
            base.remove("oneOf");
            let base_val = if has_validation_keys(&base) {
                Value::Object(base)
            } else {
                Value::Bool(true)
            };
            let mut out_branches = Vec::new();
            for br in arr {
                if let Some(step1) = self.intersect_schemas(&base_val, br, hint)?
                    && let Some(step2) = self.intersect_schemas(&da, &step1, hint)?
                {
                    out_branches.push(step2);
                }
            }
            if out_branches.is_empty() {
                return Ok(None);
            }
            let mut res = Map::new();
            res.insert("oneOf".to_string(), Value::Array(out_branches));
            return Ok(Some(Value::Object(res)));
        }

        let mut merged = Map::new();

        // 1. Intersect `type`.
        let ta = extract_explicit_types(oa, hint)?;
        let tb = extract_explicit_types(ob, hint)?;
        if let Some(types) = intersect_type_sets(ta.as_ref(), tb.as_ref()) {
            if types.is_empty() {
                return Ok(None);
            }
            if types.len() == 1 {
                merged.insert("type".to_string(), Value::String(types[0].clone()));
            } else {
                merged.insert(
                    "type".to_string(),
                    Value::Array(types.into_iter().map(Value::String).collect()),
                );
            }
        }

        // 2. Intersect `const` / `enum`.
        let lits_a = extract_literals_if_present(oa, hint)?;
        let lits_b = extract_literals_if_present(ob, hint)?;
        match (lits_a, lits_b) {
            (Some(la), Some(lb)) => {
                let mut common = Vec::new();
                for x in &la {
                    if lb.iter().any(|y| json_values_schema_equal(x, y))
                        && !common.iter().any(|y| json_values_schema_equal(x, y))
                    {
                        common.push(x.clone());
                    }
                }
                if common.is_empty() {
                    return Ok(None);
                }
                merged.insert("enum".to_string(), Value::Array(common));
            }
            (Some(la), None) => {
                merged.insert("enum".to_string(), Value::Array(la));
            }
            (None, Some(lb)) => {
                merged.insert("enum".to_string(), Value::Array(lb));
            }
            (None, None) => {}
        }

        // 3. Intersect string length bounds.
        let min_la = parse_nonneg_bound(oa, "minLength", MAX_STRING_BOUND, hint)?;
        let min_lb = parse_nonneg_bound(ob, "minLength", MAX_STRING_BOUND, hint)?;
        if let Some(m) = max_opt(min_la, min_lb) {
            merged.insert("minLength".to_string(), Value::Number((m as u64).into()));
        }
        let max_la = parse_nonneg_bound(oa, "maxLength", MAX_STRING_BOUND, hint)?;
        let max_lb = parse_nonneg_bound(ob, "maxLength", MAX_STRING_BOUND, hint)?;
        if let Some(m) = min_opt(max_la, max_lb) {
            merged.insert("maxLength".to_string(), Value::Number((m as u64).into()));
        }

        // 4. Intersect array bounds, `prefixItems`, and `items`.
        let min_ia = parse_nonneg_bound(oa, "minItems", MAX_ARRAY_BOUND, hint)?;
        let min_ib = parse_nonneg_bound(ob, "minItems", MAX_ARRAY_BOUND, hint)?;
        if let Some(m) = max_opt(min_ia, min_ib) {
            merged.insert("minItems".to_string(), Value::Number((m as u64).into()));
        }
        let max_ia = parse_nonneg_bound(oa, "maxItems", MAX_ARRAY_BOUND, hint)?;
        let max_ib = parse_nonneg_bound(ob, "maxItems", MAX_ARRAY_BOUND, hint)?;
        if let Some(m) = min_opt(max_ia, max_ib) {
            merged.insert("maxItems".to_string(), Value::Number((m as u64).into()));
        }

        let pfx_a = oa.get("prefixItems").and_then(Value::as_array);
        let pfx_b = ob.get("prefixItems").and_then(Value::as_array);
        let items_a = oa.get("items");
        let items_b = ob.get("items");
        let max_pfx = pfx_a.map_or(0, Vec::len).max(pfx_b.map_or(0, Vec::len));
        if max_pfx > 0 {
            let mut merged_pfx = Vec::with_capacity(max_pfx);
            for idx in 0..max_pfx {
                let sa = pfx_a
                    .and_then(|p| p.get(idx))
                    .or(items_a)
                    .unwrap_or(&Value::Bool(true));
                let sb = pfx_b
                    .and_then(|p| p.get(idx))
                    .or(items_b)
                    .unwrap_or(&Value::Bool(true));
                let cell = self
                    .intersect_schemas(sa, sb, &format!("{hint}-pfx-{idx}"))?
                    .unwrap_or(Value::Bool(false));
                merged_pfx.push(cell);
            }
            merged.insert("prefixItems".to_string(), Value::Array(merged_pfx));
        }
        match (items_a, items_b) {
            (Some(ia), Some(ib)) => {
                let cell = self
                    .intersect_schemas(ia, ib, &format!("{hint}-items"))?
                    .unwrap_or(Value::Bool(false));
                merged.insert("items".to_string(), cell);
            }
            (Some(ia), None) => {
                merged.insert("items".to_string(), ia.clone());
            }
            (None, Some(ib)) => {
                merged.insert("items".to_string(), ib.clone());
            }
            (None, None) => {}
        }

        // 5. Intersect object `properties`, `required`, and `additionalProperties`.
        let props_a = oa.get("properties").and_then(Value::as_object);
        let props_b = ob.get("properties").and_then(Value::as_object);
        let addl_a = oa.get("additionalProperties");
        let addl_b = ob.get("additionalProperties");

        if props_a.is_some() || props_b.is_some() {
            let mut all_keys = Vec::new();
            if let Some(pa) = props_a {
                for k in pa.keys() {
                    all_keys.push(k.clone());
                }
            }
            if let Some(pb) = props_b {
                for k in pb.keys() {
                    if !all_keys.contains(k) {
                        all_keys.push(k.clone());
                    }
                }
            }
            let mut merged_props = Map::new();
            for k in all_keys {
                let eff_a = props_a
                    .and_then(|p| p.get(&k))
                    .or(addl_a)
                    .unwrap_or(&Value::Bool(true));
                let eff_b = props_b
                    .and_then(|p| p.get(&k))
                    .or(addl_b)
                    .unwrap_or(&Value::Bool(true));
                let prop_merged = self
                    .intersect_schemas(eff_a, eff_b, &format!("{hint}-{k}"))?
                    .unwrap_or(Value::Bool(false));
                merged_props.insert(k, prop_merged);
            }
            merged.insert("properties".to_string(), Value::Object(merged_props));
        }

        match (addl_a, addl_b) {
            (Some(aa), Some(ab)) => {
                let addl_merged = self
                    .intersect_schemas(aa, ab, &format!("{hint}-addl"))?
                    .unwrap_or(Value::Bool(false));
                merged.insert("additionalProperties".to_string(), addl_merged);
            }
            (Some(aa), None) => {
                merged.insert("additionalProperties".to_string(), aa.clone());
            }
            (None, Some(ab)) => {
                merged.insert("additionalProperties".to_string(), ab.clone());
            }
            (None, None) => {}
        }

        let req_a = extract_required_list(oa, hint)?;
        let req_b = extract_required_list(ob, hint)?;
        if !req_a.is_empty() || !req_b.is_empty() {
            let mut merged_req = req_a;
            for r in req_b {
                if !merged_req.contains(&r) {
                    merged_req.push(r);
                }
            }
            merged.insert(
                "required".to_string(),
                Value::Array(merged_req.into_iter().map(Value::String).collect()),
            );
        }

        Ok(Some(Value::Object(merged)))
    }

    fn compile_object_map(
        &mut self,
        obj: &Map<String, Value>,
        hint: &str,
    ) -> Result<Option<String>> {
        if let Some(type_val) = obj.get("type") {
            match type_val {
                Value::String(t) => return self.compile_single_type(t, obj, hint),
                Value::Array(types) => {
                    if types.is_empty() {
                        bail!("JSON Schema 'type' array at '{hint}' must not be empty");
                    }
                    let mut type_names = Vec::with_capacity(types.len());
                    for tv in types {
                        let Some(ts) = tv.as_str() else {
                            bail!("JSON Schema 'type' array elements must be strings");
                        };
                        if !type_names.contains(&ts) {
                            type_names.push(ts);
                        }
                    }
                    if type_names.contains(&"number") && type_names.contains(&"integer") {
                        type_names.retain(|&t| t != "integer");
                    }
                    let mut alts = Vec::with_capacity(type_names.len());
                    for ts in type_names {
                        if let Some(r) = self.compile_single_type(ts, obj, hint)? {
                            alts.push(r);
                        }
                    }
                    if alts.is_empty() {
                        return Ok(None);
                    }
                    if alts.len() == 1 {
                        return Ok(alts.into_iter().next());
                    }
                    return Ok(Some(self.add_rule(hint, alts.join(" | "))));
                }
                _ => bail!("JSON Schema 'type' at '{hint}' must be a string or array of strings"),
            }
        }

        if obj.contains_key("properties")
            || obj.contains_key("additionalProperties")
            || obj.contains_key("required")
        {
            return self.compile_single_type("object", obj, hint);
        }
        if obj.contains_key("items")
            || obj.contains_key("prefixItems")
            || obj.contains_key("minItems")
            || obj.contains_key("maxItems")
        {
            return self.compile_single_type("array", obj, hint);
        }
        if obj.contains_key("minLength") || obj.contains_key("maxLength") {
            return self.compile_single_type("string", obj, hint);
        }

        Ok(Some("json-value".to_string()))
    }

    fn compile_single_type(
        &mut self,
        t: &str,
        obj: &Map<String, Value>,
        hint: &str,
    ) -> Result<Option<String>> {
        match t {
            "string" => self.compile_string(obj, hint),
            "integer" => Ok(Some("json-int".to_string())),
            "number" => Ok(Some("json-number".to_string())),
            "boolean" => Ok(Some("json-bool".to_string())),
            "null" => Ok(Some("json-null".to_string())),
            "array" => self.compile_array(obj, hint),
            "object" => self.compile_object(obj, hint),
            other => bail!("unsupported JSON Schema type '{other}' at '{hint}'"),
        }
    }

    fn compile_string(&mut self, obj: &Map<String, Value>, hint: &str) -> Result<Option<String>> {
        let min_len = parse_nonneg_bound(obj, "minLength", MAX_STRING_BOUND, hint)?.unwrap_or(0);
        let max_len = parse_nonneg_bound(obj, "maxLength", MAX_STRING_BOUND, hint)?;
        if min_len == 0 && max_len.is_none() {
            return Ok(Some("json-string".to_string()));
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
        Ok(Some(self.add_rule(hint, expr)))
    }

    fn compile_array(&mut self, obj: &Map<String, Value>, hint: &str) -> Result<Option<String>> {
        let saved_unguarded = std::mem::take(&mut self.active_refs);
        let res = self.compile_array_inner(obj, hint);
        self.active_refs = saved_unguarded;
        res
    }

    fn compile_array_inner(
        &mut self,
        obj: &Map<String, Value>,
        hint: &str,
    ) -> Result<Option<String>> {
        let min_items = parse_nonneg_bound(obj, "minItems", MAX_ARRAY_BOUND, hint)?.unwrap_or(0);
        let max_items = parse_nonneg_bound(obj, "maxItems", MAX_ARRAY_BOUND, hint)?;
        if let Some(max_n) = max_items
            && max_n < min_items
        {
            bail!("JSON Schema array at '{hint}' has maxItems ({max_n}) < minItems ({min_items})");
        }

        if let Some(pfx_val) = obj.get("prefixItems") {
            let Some(prefix_items) = pfx_val.as_array() else {
                bail!("JSON Schema 'prefixItems' at '{hint}' must be an array");
            };
            if prefix_items.len() > MAX_ARRAY_BOUND as usize {
                bail!(
                    "JSON Schema 'prefixItems' ({}) at '{hint}' exceeds maximum supported bound ({MAX_ARRAY_BOUND})",
                    prefix_items.len()
                );
            }
            if max_items == Some(0) {
                return Ok(Some(self.add_rule(hint, "\"[\" ws \"]\" ws".to_string())));
            }

            let mut eff_max = max_items;
            let mut compiled_prefix = Vec::with_capacity(prefix_items.len());
            for (idx, item) in prefix_items.iter().enumerate() {
                if eff_max.is_some_and(|m| idx >= m) {
                    break;
                }
                match self.compile_value(item, &format!("{hint}-tuple-{idx}"))? {
                    Some(rule) => compiled_prefix.push(rule),
                    None => {
                        eff_max = Some(eff_max.map_or(idx, |m| m.min(idx)));
                        break;
                    }
                }
            }

            let p_len = compiled_prefix.len();
            let tail_item = if eff_max.is_some_and(|m| m <= p_len) {
                None
            } else {
                match obj.get("items") {
                    Some(s) => match self.compile_value(s, &format!("{hint}-tail-item"))? {
                        Some(r) => Some(r),
                        None => {
                            eff_max = Some(eff_max.map_or(p_len, |m| m.min(p_len)));
                            None
                        }
                    },
                    None => Some("json-value".to_string()),
                }
            };

            if let Some(em) = eff_max
                && em < min_items
            {
                return Ok(None);
            }

            if p_len == 0 {
                let Some(t_expr) = tail_item else {
                    return Ok(Some(self.add_rule(hint, "\"[\" ws \"]\" ws".to_string())));
                };
                let expr = build_homogeneous_array_expr(&t_expr, min_items, eff_max);
                return Ok(Some(self.add_rule(hint, expr)));
            }

            let after_prefix = match (&tail_item, eff_max) {
                (Some(t), None) => format!(" (\",\" ws {t})*"),
                (Some(t), Some(em)) if em > p_len => {
                    let rem = em - p_len;
                    format!(" (\",\" ws {t}){{0,{rem}}}")
                }
                _ => String::new(),
            };

            let expr = if min_items <= p_len {
                let mut acc = after_prefix;
                for idx in (min_items..p_len).rev() {
                    let rule = &compiled_prefix[idx];
                    if idx == 0 {
                        acc = format!("({rule}{acc})?");
                    } else {
                        acc = format!(" (\",\" ws {rule}{acc})?");
                    }
                }
                if min_items == 0 {
                    format!("\"[\" ws {acc} \"]\" ws")
                } else {
                    let req_head = compiled_prefix[..min_items].join(" \",\" ws ");
                    format!("\"[\" ws {req_head}{acc} \"]\" ws")
                }
            } else {
                // `min_items > p_len`: all prefix items plus `min_items - p_len` tail items are mandatory.
                let t = tail_item.expect("tail_item guaranteed when min_items > p_len");
                let req_head = compiled_prefix.join(" \",\" ws ");
                let extra_req = min_items - p_len;
                let extra_opt = match eff_max {
                    None => format!(" (\",\" ws {t})*"),
                    Some(em) if em > min_items => {
                        let rem = em - min_items;
                        format!(" (\",\" ws {t}){{0,{rem}}}")
                    }
                    _ => String::new(),
                };
                format!("\"[\" ws {req_head} (\",\" ws {t}){{{extra_req}}}{extra_opt} \"]\" ws")
            };

            return Ok(Some(self.add_rule(hint, expr)));
        }

        if max_items == Some(0) {
            return Ok(Some(self.add_rule(hint, "\"[\" ws \"]\" ws".to_string())));
        }

        let item_opt = match obj.get("items") {
            Some(item_schema) => self.compile_value(item_schema, &format!("{hint}-item"))?,
            None => Some("json-value".to_string()),
        };

        let Some(item_expr) = item_opt else {
            return if min_items == 0 {
                Ok(Some(self.add_rule(hint, "\"[\" ws \"]\" ws".to_string())))
            } else {
                Ok(None)
            };
        };

        let expr = build_homogeneous_array_expr(&item_expr, min_items, max_items);
        Ok(Some(self.add_rule(hint, expr)))
    }

    fn compile_object(&mut self, obj: &Map<String, Value>, hint: &str) -> Result<Option<String>> {
        let saved_unguarded = std::mem::take(&mut self.active_refs);
        let res = self.compile_object_inner(obj, hint);
        self.active_refs = saved_unguarded;
        res
    }

    fn compile_object_inner(
        &mut self,
        obj: &Map<String, Value>,
        hint: &str,
    ) -> Result<Option<String>> {
        let props_opt = match obj.get("properties") {
            None => None,
            Some(v) => Some(v.as_object().ok_or_else(|| {
                anyhow::anyhow!("JSON Schema 'properties' at '{hint}' must be an object")
            })?),
        };
        let required_list = extract_required_list(obj, hint)?;
        let required_set: BTreeSet<&str> = required_list.iter().map(String::as_str).collect();
        let addl = obj.get("additionalProperties");

        // Fast paths when neither `properties` nor `required` constrains named keys.
        if props_opt.is_none() && required_list.is_empty() {
            return match addl {
                None | Some(Value::Bool(true)) => Ok(Some("json-object".to_string())),
                Some(Value::Bool(false)) => {
                    Ok(Some(self.add_rule(hint, "\"{\" ws \"}\" ws".to_string())))
                }
                Some(schema_val @ Value::Object(_)) => {
                    match self.compile_value(schema_val, &format!("{hint}-addl-val"))? {
                        Some(val_expr) => {
                            let kv = format!("json-string \":\" ws {val_expr}");
                            let kv_rule = self.add_rule(&format!("{hint}-addl"), kv);
                            let map_expr =
                                format!("\"{{\" ws ({kv_rule} (\",\" ws {kv_rule})*)? \"}}\" ws");
                            Ok(Some(self.add_rule(hint, map_expr)))
                        }
                        None => Ok(Some(self.add_rule(hint, "\"{\" ws \"}\" ws".to_string()))),
                    }
                }
                Some(_) => bail!("invalid 'additionalProperties' in JSON Schema at '{hint}'"),
            };
        }

        // Compile `additionalProperties` value schema once (used for undeclared `required` keys and additional keys).
        let addl_val_expr: Option<String> = match addl {
            None | Some(Value::Bool(true)) => Some("json-value".to_string()),
            Some(Value::Bool(false)) => None,
            Some(schema_val @ Value::Object(_)) => {
                self.compile_value(schema_val, &format!("{hint}-addl-val"))?
            }
            Some(_) => bail!("invalid 'additionalProperties' in JSON Schema at '{hint}'"),
        };

        let empty_map = Map::new();
        let props = props_opt.unwrap_or(&empty_map);

        let mut all_known_keys = BTreeSet::new();
        for k in props.keys() {
            all_known_keys.insert(k.clone());
        }
        for r in &required_list {
            all_known_keys.insert(r.clone());
        }

        let mut prop_rules: Vec<(String, bool)> =
            Vec::with_capacity(props.len() + required_list.len());
        for (prop_name, prop_schema) in props {
            let is_req = required_set.contains(prop_name.as_str());
            match self.compile_value(prop_schema, &format!("{hint}-{prop_name}"))? {
                Some(val_expr) => {
                    let kv_expr = format!("{} \":\" ws {}", gbnf_string_token(prop_name), val_expr);
                    let kv_rule = self.add_rule(&format!("{hint}-kv-{prop_name}"), kv_expr);
                    prop_rules.push((kv_rule, is_req));
                }
                None => {
                    if is_req {
                        return Ok(None);
                    }
                }
            }
        }

        // Any required key absent from `properties` is governed by `additionalProperties`.
        for req_name in &required_list {
            if !props.contains_key(req_name) {
                let Some(ref aval) = addl_val_expr else {
                    // `additionalProperties: false` (or unsatisfiable) forbids this required key!
                    return Ok(None);
                };
                let kv_expr = format!("{} \":\" ws {}", gbnf_string_token(req_name), aval);
                let kv_rule = self.add_rule(&format!("{hint}-kv-{req_name}"), kv_expr);
                prop_rules.push((kv_rule, true));
            }
        }

        // Build `additionalProperties` key-value rule only when `additionalProperties` is explicitly enabled/schema-valued
        // or when `properties` is absent (`additionalProperties` defaults to `true` in JSON Schema!).
        // Wait: when `additionalProperties` is `None` (unspecified), JSON Schema allows extra keys, EXCEPT
        // extra keys must NEVER match any declared key in `all_known_keys` (including `\uXXXX` spellings)!
        let allow_extra_keys = !matches!(addl, Some(Value::Bool(false))) && addl_val_expr.is_some();
        let addl_kv_rule = if allow_extra_keys {
            let aval = addl_val_expr.as_ref().unwrap();
            let key_rule =
                self.compile_string_excluding_keys(&all_known_keys, &format!("{hint}-addl-key"));
            let kv = format!("{key_rule} \":\" ws {aval}");
            Some(self.add_rule(&format!("{hint}-addl"), kv))
        } else {
            None
        };

        let obj_expr = self.build_ordered_object_fields(&prop_rules, addl_kv_rule.as_deref(), hint);
        Ok(Some(self.add_rule(hint, obj_expr)))
    }

    /// Emits named shared suffix & start rules ($O(N)$ grammar size in the number of optional fields)
    /// for ordered object properties `(kv_rule, is_required)`.
    fn build_ordered_object_fields(
        &mut self,
        props: &[(String, bool)],
        addl_kv: Option<&str>,
        hint: &str,
    ) -> String {
        if props.is_empty() {
            return match addl_kv {
                Some(akv) => format!("\"{{\" ws ({akv} (\",\" ws {akv})*)? \"}}\" ws"),
                None => "\"{\" ws \"}\" ws".to_string(),
            };
        }

        let n = props.len();
        let mut suf_after: Vec<Option<String>> = vec![None; n];
        let mut cur_tail = addl_kv
            .map(|akv| self.add_rule(&format!("{hint}-addl-tail"), format!("(\",\" ws {akv})*")));
        suf_after[n - 1] = cur_tail.clone();

        for idx in (1..n).rev() {
            let (ref rule_i, is_req_i) = props[idx];
            let step_expr = match (is_req_i, &cur_tail) {
                (true, None) => format!("\",\" ws {rule_i}"),
                (true, Some(t)) => format!("\",\" ws {rule_i} {t}"),
                (false, None) => format!("(\",\" ws {rule_i})?"),
                (false, Some(t)) => format!("(\",\" ws {rule_i})? {t}"),
            };
            let suf_rule = self.add_rule(&format!("{hint}-suf-{idx}"), step_expr);
            cur_tail = Some(suf_rule);
            suf_after[idx - 1] = cur_tail.clone();
        }

        let first_req = props.iter().position(|(_, req)| *req);
        let limit = first_req.unwrap_or(n - 1);

        let emit_at = |idx: usize, suf: &[Option<String>]| -> String {
            let rule_i = &props[idx].0;
            match &suf[idx] {
                Some(t) => format!("{rule_i} {t}"),
                None => rule_i.clone(),
            }
        };

        let mut nonempty = if first_req.is_some() {
            emit_at(limit, &suf_after)
        } else {
            let last_emit = emit_at(limit, &suf_after);
            match addl_kv {
                Some(akv) => self.add_rule(
                    &format!("{hint}-start-{limit}"),
                    format!("{last_emit} | {akv} (\",\" ws {akv})*"),
                ),
                None => last_emit,
            }
        };

        for idx in (0..limit).rev() {
            let here = emit_at(idx, &suf_after);
            nonempty = self.add_rule(
                &format!("{hint}-start-{idx}"),
                format!("{here} | {nonempty}"),
            );
        }

        if first_req.is_some() {
            format!("\"{{\" ws {nonempty} \"}}\" ws")
        } else {
            format!("\"{{\" ws ({nonempty})? \"}}\" ws")
        }
    }

    /// Compiles a GBNF rule matching any valid JSON string token (`"..." ws`) whose decoded
    /// Unicode character sequence is NOT in `excluded_keys` (accounting for raw UTF-8, short
    /// backslash escapes, `\uXXXX` BMP escapes, and `\uHHHH\uLLLL` UTF-16 surrogate pairs).
    fn compile_string_excluding_keys(
        &mut self,
        excluded_keys: &BTreeSet<String>,
        hint: &str,
    ) -> String {
        if excluded_keys.is_empty() {
            return "json-string".to_string();
        }

        #[derive(Default)]
        struct KeyTrie {
            terminal: bool,
            children: BTreeMap<char, usize>,
        }

        let mut nodes = vec![KeyTrie::default()];
        for key in excluded_keys {
            let mut cur = 0usize;
            for ch in key.chars() {
                let next_idx = if let Some(&existing) = nodes[cur].children.get(&ch) {
                    existing
                } else {
                    let id = nodes.len();
                    nodes.push(KeyTrie::default());
                    nodes[cur].children.insert(ch, id);
                    id
                };
                cur = next_idx;
            }
            nodes[cur].terminal = true;
        }

        let mut state_rules = vec![String::new(); nodes.len()];
        for idx in (0..nodes.len()).rev() {
            let mut alts = Vec::new();
            if !nodes[idx].terminal {
                alts.push("\"\\\"\" ws".to_string());
            }
            if nodes[idx].children.is_empty() {
                // `terminal` must be true here (leaf of non-empty trie, or empty-string key).
                alts.push("json-char+ \"\\\"\" ws".to_string());
            } else {
                let branch_chars: BTreeSet<char> = nodes[idx].children.keys().copied().collect();
                let not_branch_char = self.add_rule(
                    &format!("{hint}-not-{idx}"),
                    gbnf_json_char_excluding(&branch_chars),
                );
                alts.push(format!("{not_branch_char} json-char* \"\\\"\" ws"));
                let children_vec: Vec<(char, usize)> =
                    nodes[idx].children.iter().map(|(&c, &n)| (c, n)).collect();
                for (ch, child_idx) in children_vec {
                    let match_ch =
                        self.add_rule(&format!("{hint}-ch-{idx}"), gbnf_json_char_matching(ch));
                    let child_rule = &state_rules[child_idx];
                    alts.push(format!("{match_ch} {child_rule}"));
                }
            }
            state_rules[idx] = self.add_rule(&format!("{hint}-st-{idx}"), alts.join(" | "));
        }

        let root_state = &state_rules[0];
        self.add_rule(hint, format!("\"\\\"\" {root_state}"))
    }

    /// Detects cycles in emitted rules along paths that can consume zero input characters.
    fn check_no_zero_input_cycles(&self, root_expr: &str) -> Result<()> {
        let mut all_rules = self.rules.clone();
        all_rules.insert("root".to_string(), format!("ws {root_expr}"));

        // 1. Compute nullable rules (`ws` is the only nullable base rule).
        let mut nullable = BTreeSet::new();
        nullable.insert("ws".to_string());
        loop {
            let mut changed = false;
            for (name, expr) in &all_rules {
                if nullable.contains(name) {
                    continue;
                }
                if expr_is_nullable(expr, &nullable) {
                    nullable.insert(name.clone());
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }

        // 2. Build zero-input transition graph among user rules (`self.rules` + `root`).
        let mut edges: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        for (name, expr) in &all_rules {
            let mut out_set = BTreeSet::new();
            collect_zero_input_targets(expr, &all_rules, &nullable, &mut out_set);
            edges.insert(name.clone(), out_set);
        }

        // 3. 3-color DFS from `root` (and all rules) to reject any zero-input cycle.
        let mut state: BTreeMap<String, u8> = BTreeMap::new(); // 0=unvisited, 1=visiting, 2=done
        fn dfs(
            u: &str,
            edges: &BTreeMap<String, BTreeSet<String>>,
            state: &mut BTreeMap<String, u8>,
        ) -> bool {
            state.insert(u.to_string(), 1);
            if let Some(nbrs) = edges.get(u) {
                for v in nbrs {
                    match state.get(v).copied().unwrap_or(0) {
                        1 => return true,
                        0 if dfs(v, edges, state) => return true,
                        _ => {}
                    }
                }
            }
            state.insert(u.to_string(), 2);
            false
        }

        for rule_name in all_rules.keys() {
            if state.get(rule_name).copied().unwrap_or(0) == 0 && dfs(rule_name, &edges, &mut state)
            {
                bail!("unguarded recursive JSON Schema $ref cycle detected in compiled grammar");
            }
        }
        Ok(())
    }
}

fn build_homogeneous_array_expr(
    item_expr: &str,
    min_items: usize,
    max_items: Option<usize>,
) -> String {
    match (min_items, max_items) {
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
            format!("\"[\" ws {item_expr} (\",\" ws {item_expr}){{{tail_min},{tail_max}}} \"]\" ws")
        }
    }
}

fn extract_required_list(obj: &Map<String, Value>, hint: &str) -> Result<Vec<String>> {
    let Some(req_val) = obj.get("required") else {
        return Ok(Vec::new());
    };
    let Some(arr) = req_val.as_array() else {
        bail!("JSON Schema 'required' at '{hint}' must be an array");
    };
    let mut out = Vec::with_capacity(arr.len());
    for item in arr {
        let Some(s) = item.as_str() else {
            bail!("JSON Schema 'required' elements at '{hint}' must be strings");
        };
        if !out.iter().any(|x| x == s) {
            out.push(s.to_string());
        }
    }
    Ok(out)
}

fn extract_explicit_types(obj: &Map<String, Value>, hint: &str) -> Result<Option<Vec<String>>> {
    let Some(tv) = obj.get("type") else {
        return Ok(None);
    };
    match tv {
        Value::String(s) => Ok(Some(vec![s.clone()])),
        Value::Array(arr) => {
            if arr.is_empty() {
                bail!("JSON Schema 'type' array at '{hint}' must not be empty");
            }
            let mut out = Vec::with_capacity(arr.len());
            for item in arr {
                let Some(s) = item.as_str() else {
                    bail!("JSON Schema 'type' array elements at '{hint}' must be strings");
                };
                if !out.iter().any(|x| x == s) {
                    out.push(s.to_string());
                }
            }
            Ok(Some(out))
        }
        _ => bail!("JSON Schema 'type' at '{hint}' must be a string or array of strings"),
    }
}

fn intersect_type_sets(a: Option<&Vec<String>>, b: Option<&Vec<String>>) -> Option<Vec<String>> {
    match (a, b) {
        (None, None) => None,
        (Some(ta), None) => Some(ta.clone()),
        (None, Some(tb)) => Some(tb.clone()),
        (Some(ta), Some(tb)) => {
            let mut out = Vec::new();
            for t in ta {
                if tb.contains(t) || (t == "integer" && tb.iter().any(|x| x == "number")) {
                    if !out.contains(t) {
                        out.push(t.clone());
                    }
                } else if t == "number"
                    && tb.iter().any(|x| x == "integer")
                    && !out.iter().any(|x| x == "integer")
                {
                    out.push("integer".to_string());
                }
            }
            Some(out)
        }
    }
}

fn extract_literals_if_present(obj: &Map<String, Value>, hint: &str) -> Result<Option<Vec<Value>>> {
    let c = obj.get("const");
    let e = match obj.get("enum") {
        None => None,
        Some(v) => {
            let Some(arr) = v.as_array() else {
                bail!("JSON Schema 'enum' at '{hint}' must be an array");
            };
            if arr.is_empty() {
                bail!("JSON Schema 'enum' at '{hint}' must not be empty");
            }
            Some(arr)
        }
    };
    match (c, e) {
        (None, None) => Ok(None),
        (Some(cv), None) => Ok(Some(vec![cv.clone()])),
        (None, Some(arr)) => Ok(Some(arr.clone())),
        (Some(cv), Some(arr)) => {
            if arr.iter().any(|x| json_values_schema_equal(x, cv)) {
                Ok(Some(vec![cv.clone()]))
            } else {
                Ok(Some(Vec::new()))
            }
        }
    }
}

fn extract_pure_literals(schema: &Value) -> Option<Result<Vec<Value>>> {
    let obj = schema.as_object()?;
    if !obj.contains_key("const") && !obj.contains_key("enum") {
        return None;
    }
    if obj.contains_key("anyOf") || obj.contains_key("oneOf") || obj.contains_key("allOf") {
        return None;
    }
    Some(extract_literals_if_present(obj, "oneOf").map(|opt| opt.unwrap_or_default()))
}

fn is_plain_single_type(schema: &Value, expected: &str) -> bool {
    let Some(obj) = schema.as_object() else {
        return false;
    };
    if obj.get("type").and_then(Value::as_str) != Some(expected) {
        return false;
    }
    obj.keys().all(|k| k == "type" || is_annotation_key(k))
}

fn inferred_canonical_types(schema: &Value) -> Option<BTreeSet<&'static str>> {
    let obj = schema.as_object()?;
    if let Some(c) = obj.get("const") {
        return Some(canonical_types_for_literal(c));
    }
    if let Some(arr) = obj.get("enum").and_then(Value::as_array) {
        let mut set = BTreeSet::new();
        for item in arr {
            set.extend(canonical_types_for_literal(item));
        }
        return Some(set);
    }
    let tv = obj.get("type")?;
    let mut set = BTreeSet::new();
    let mut add_t = |t: &str| -> bool {
        match t {
            "null" => {
                set.insert("null");
                true
            }
            "boolean" => {
                set.insert("boolean");
                true
            }
            "string" => {
                set.insert("string");
                true
            }
            "array" => {
                set.insert("array");
                true
            }
            "object" => {
                set.insert("object");
                true
            }
            "integer" => {
                set.insert("integer");
                true
            }
            "number" => {
                set.insert("integer");
                set.insert("nonint_number");
                true
            }
            _ => false,
        }
    };
    match tv {
        Value::String(s) => {
            if !add_t(s) {
                return None;
            }
        }
        Value::Array(arr) => {
            for v in arr {
                let s = v.as_str()?;
                if !add_t(s) {
                    return None;
                }
            }
        }
        _ => return None,
    }
    Some(set)
}

fn canonical_types_for_literal(val: &Value) -> BTreeSet<&'static str> {
    let mut s = BTreeSet::new();
    match val {
        Value::Null => {
            s.insert("null");
        }
        Value::Bool(_) => {
            s.insert("boolean");
        }
        Value::String(_) => {
            s.insert("string");
        }
        Value::Array(_) => {
            s.insert("array");
        }
        Value::Object(_) => {
            s.insert("object");
        }
        Value::Number(_) => {
            if value_matches_primitive_type(val, "integer") {
                s.insert("integer");
            } else {
                s.insert("nonint_number");
            }
        }
    }
    s
}

fn schemas_provably_disjoint(a: &Value, b: &Value) -> bool {
    if matches!(a, Value::Bool(false)) || matches!(b, Value::Bool(false)) {
        return true;
    }
    if let (Some(ta), Some(tb)) = (inferred_canonical_types(a), inferred_canonical_types(b))
        && ta.is_disjoint(&tb)
    {
        return true;
    }
    if let (Some(Ok(la)), Some(Ok(lb))) = (extract_pure_literals(a), extract_pure_literals(b))
        && !la
            .iter()
            .any(|x| lb.iter().any(|y| json_values_schema_equal(x, y)))
    {
        return true;
    }
    // Check required object properties for a provably disjoint discriminator field.
    if let (Some(oa), Some(ob)) = (a.as_object(), b.as_object()) {
        let req_a: BTreeSet<&str> = oa
            .get("required")
            .and_then(Value::as_array)
            .map(|r| r.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();
        let req_b: BTreeSet<&str> = ob
            .get("required")
            .and_then(Value::as_array)
            .map(|r| r.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();
        if let (Some(pa), Some(pb)) = (
            oa.get("properties").and_then(Value::as_object),
            ob.get("properties").and_then(Value::as_object),
        ) {
            for k in req_a.intersection(&req_b) {
                if let (Some(sa), Some(sb)) = (pa.get(*k), pb.get(*k))
                    && schemas_provably_disjoint(sa, sb)
                {
                    return true;
                }
            }
        }
    }
    false
}

fn max_opt(a: Option<usize>, b: Option<usize>) -> Option<usize> {
    match (a, b) {
        (Some(x), Some(y)) => Some(x.max(y)),
        (Some(x), None) => Some(x),
        (None, Some(y)) => Some(y),
        (None, None) => None,
    }
}

fn min_opt(a: Option<usize>, b: Option<usize>) -> Option<usize> {
    match (a, b) {
        (Some(x), Some(y)) => Some(x.min(y)),
        (Some(x), None) => Some(x),
        (None, Some(y)) => Some(y),
        (None, None) => None,
    }
}

fn hex_nibble_gbnf(nib: u8) -> &'static str {
    match nib {
        0 => "\"0\"",
        1 => "\"1\"",
        2 => "\"2\"",
        3 => "\"3\"",
        4 => "\"4\"",
        5 => "\"5\"",
        6 => "\"6\"",
        7 => "\"7\"",
        8 => "\"8\"",
        9 => "\"9\"",
        10 => "[aA]",
        11 => "[bB]",
        12 => "[cC]",
        13 => "[dD]",
        14 => "[eE]",
        _ => "[fF]",
    }
}

fn hex4_u16_gbnf(code: u16) -> String {
    let n0 = ((code >> 12) & 0xF) as u8;
    let n1 = ((code >> 8) & 0xF) as u8;
    let n2 = ((code >> 4) & 0xF) as u8;
    let n3 = (code & 0xF) as u8;
    format!(
        "{} {} {} {}",
        hex_nibble_gbnf(n0),
        hex_nibble_gbnf(n1),
        hex_nibble_gbnf(n2),
        hex_nibble_gbnf(n3)
    )
}

fn gbnf_json_char_matching(ch: char) -> String {
    let mut alts = Vec::new();
    let code = ch as u32;
    if code >= 0x20 && code != 0x7F && ch != '"' && ch != '\\' {
        alts.push(gbnf_literal(&ch.to_string()));
    }
    match ch {
        '"' => alts.push("\"\\\\\" \"\\\"\"".to_string()),
        '\\' => alts.push("\"\\\\\" \"\\\\\"".to_string()),
        '/' => alts.push("\"\\\\\" \"/\"".to_string()),
        '\x08' => alts.push("\"\\\\\" \"b\"".to_string()),
        '\x0C' => alts.push("\"\\\\\" \"f\"".to_string()),
        '\n' => alts.push("\"\\\\\" \"n\"".to_string()),
        '\r' => alts.push("\"\\\\\" \"r\"".to_string()),
        '\t' => alts.push("\"\\\\\" \"t\"".to_string()),
        _ => {}
    }
    if code <= 0xFFFF {
        alts.push(format!("\"\\\\\" \"u\" {}", hex4_u16_gbnf(code as u16)));
    } else {
        let u = code - 0x10000;
        let hi = 0xD800u16 + ((u >> 10) as u16);
        let lo = 0xDC00u16 + ((u & 0x3FF) as u16);
        alts.push(format!(
            "\"\\\\\" \"u\" {} \"\\\\\" \"u\" {}",
            hex4_u16_gbnf(hi),
            hex4_u16_gbnf(lo)
        ));
    }
    alts.join(" | ")
}

fn nibble_char_class(nibbles: &[u8]) -> String {
    let mut s = String::from("[");
    for &n in nibbles {
        match n {
            0..=9 => s.push((b'0' + n) as char),
            10..=15 => {
                s.push((b'a' + (n - 10)) as char);
                s.push((b'A' + (n - 10)) as char);
            }
            _ => {}
        }
    }
    s.push(']');
    s
}

fn bmp_hex_complement(
    depth: usize,
    prefix_is_d: bool,
    excluded: &BTreeSet<[u8; 4]>,
) -> Option<String> {
    if excluded.is_empty() {
        return Some(match (depth, prefix_is_d) {
            (0, _) => "json-hex4-bmp".to_string(),
            (1, true) => "[0-7] [0-9a-fA-F]{2}".to_string(),
            (1, false) => "[0-9a-fA-F]{3}".to_string(),
            (2, _) => "[0-9a-fA-F]{2}".to_string(),
            (3, _) => "[0-9a-fA-F]".to_string(),
            _ => return None,
        });
    }
    if depth >= 4 {
        return None;
    }

    let valid_max: u8 = if depth == 1 && prefix_is_d { 8 } else { 16 };
    let mut free_non_d = Vec::new();
    let mut free_d = false;
    let mut constrained = Vec::new();

    for nib in 0..valid_max {
        let sub: BTreeSet<[u8; 4]> = excluded
            .iter()
            .filter(|arr| arr[depth] == nib)
            .copied()
            .collect();
        if sub.is_empty() {
            if depth == 0 && nib == 13 {
                free_d = true;
            } else {
                free_non_d.push(nib);
            }
        } else {
            constrained.push((nib, sub));
        }
    }

    let mut alts = Vec::new();
    if !free_non_d.is_empty() {
        let cc = nibble_char_class(&free_non_d);
        let rem = 3 - depth;
        if rem == 0 {
            alts.push(cc);
        } else if rem == 1 {
            alts.push(format!("{cc} [0-9a-fA-F]"));
        } else {
            alts.push(format!("{cc} [0-9a-fA-F]{{{rem}}}"));
        }
    }
    if free_d {
        alts.push("[dD] [0-7] [0-9a-fA-F]{2}".to_string());
    }
    for (nib, sub) in constrained {
        let next_is_d = depth == 0 && nib == 13;
        if let Some(sub_expr) = bmp_hex_complement(depth + 1, next_is_d, &sub) {
            alts.push(format!("{} ({sub_expr})", hex_nibble_gbnf(nib)));
        }
    }
    if alts.is_empty() {
        None
    } else {
        Some(alts.join(" | "))
    }
}

fn surrogate_hex_complement(depth: usize, excluded: &BTreeSet<[u8; 8]>) -> Option<String> {
    if excluded.is_empty() {
        return Some(match depth {
            0 => "json-surrogate-pair".to_string(),
            _ => unreachable!(),
        });
    }
    if depth >= 8 {
        return None;
    }

    let valid_nibbles: &[u8] = match depth {
        0 | 4 => &[13],
        1 => &[8, 9, 10, 11],
        5 => &[12, 13, 14, 15],
        _ => &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    };

    let prefix = match depth {
        0 => "\"\\\\\" \"u\" ",
        4 => "\"\\\\\" \"u\" ",
        _ => "",
    };

    let mut free = Vec::new();
    let mut constrained = Vec::new();
    for &nib in valid_nibbles {
        let sub: BTreeSet<[u8; 8]> = excluded
            .iter()
            .filter(|arr| arr[depth] == nib)
            .copied()
            .collect();
        if sub.is_empty() {
            free.push(nib);
        } else {
            constrained.push((nib, sub));
        }
    }

    let tail_after = |d: usize| -> String {
        match d {
            0 => "[89aAbB] [0-9a-fA-F]{2} \"\\\\\" \"u\" [dD] [c-fC-F] [0-9a-fA-F]{2}".to_string(),
            1 => "[0-9a-fA-F]{2} \"\\\\\" \"u\" [dD] [c-fC-F] [0-9a-fA-F]{2}".to_string(),
            2 => "[0-9a-fA-F] \"\\\\\" \"u\" [dD] [c-fC-F] [0-9a-fA-F]{2}".to_string(),
            3 => "\"\\\\\" \"u\" [dD] [c-fC-F] [0-9a-fA-F]{2}".to_string(),
            4 => "[c-fC-F] [0-9a-fA-F]{2}".to_string(),
            5 => "[0-9a-fA-F]{2}".to_string(),
            6 => "[0-9a-fA-F]".to_string(),
            _ => String::new(),
        }
    };

    let mut alts = Vec::new();
    if !free.is_empty() {
        let cc = nibble_char_class(&free);
        let tail = tail_after(depth);
        if tail.is_empty() {
            alts.push(format!("{prefix}{cc}"));
        } else {
            alts.push(format!("{prefix}{cc} {tail}"));
        }
    }
    for (nib, sub) in constrained {
        if let Some(sub_expr) = surrogate_hex_complement(depth + 1, &sub) {
            alts.push(format!("{prefix}{} ({sub_expr})", hex_nibble_gbnf(nib)));
        }
    }
    if alts.is_empty() {
        None
    } else {
        Some(alts.join(" | "))
    }
}

fn gbnf_json_char_excluding(excluded: &BTreeSet<char>) -> String {
    let mut alts = Vec::new();

    // 1. Raw UTF-8 char class excluding `"`, `\`, control chars, and `excluded`.
    let mut raw_cc = String::from("[^\"\\\\\\x7F\\x00-\\x1F");
    for &ch in excluded {
        let code = ch as u32;
        if code >= 0x20 && code != 0x7F && ch != '"' && ch != '\\' {
            match ch {
                ']' => raw_cc.push_str("\\]"),
                '-' => raw_cc.push_str("\\-"),
                '^' => raw_cc.push_str("\\x5E"),
                c => raw_cc.push(c),
            }
        }
    }
    raw_cc.push(']');
    alts.push(raw_cc);

    // 2. Short backslash escapes not in `excluded`.
    let short_escapes = [
        ('"', "\\\""),
        ('\\', "\\\\"),
        ('/', "/"),
        ('\x08', "b"),
        ('\x0C', "f"),
        ('\n', "n"),
        ('\r', "r"),
        ('\t', "t"),
    ];
    let mut esc_cc = String::new();
    for (decoded, gbnf_tok) in short_escapes {
        if !excluded.contains(&decoded) {
            esc_cc.push_str(gbnf_tok);
        }
    }
    if !esc_cc.is_empty() {
        alts.push(format!("\"\\\\\" [{esc_cc}]"));
    }

    // 3. `\uXXXX` BMP escapes not in `excluded`.
    let mut excluded_bmp = BTreeSet::new();
    let mut excluded_surr = BTreeSet::new();
    for &ch in excluded {
        let code = ch as u32;
        if code <= 0xFFFF {
            let u = code as u16;
            excluded_bmp.insert([
                ((u >> 12) & 0xF) as u8,
                ((u >> 8) & 0xF) as u8,
                ((u >> 4) & 0xF) as u8,
                (u & 0xF) as u8,
            ]);
        } else {
            let v = code - 0x10000;
            let hi = 0xD800u16 + ((v >> 10) as u16);
            let lo = 0xDC00u16 + ((v & 0x3FF) as u16);
            excluded_surr.insert([
                ((hi >> 12) & 0xF) as u8,
                ((hi >> 8) & 0xF) as u8,
                ((hi >> 4) & 0xF) as u8,
                (hi & 0xF) as u8,
                ((lo >> 12) & 0xF) as u8,
                ((lo >> 8) & 0xF) as u8,
                ((lo >> 4) & 0xF) as u8,
                (lo & 0xF) as u8,
            ]);
        }
    }
    if let Some(bmp_expr) = bmp_hex_complement(0, false, &excluded_bmp) {
        alts.push(format!("\"\\\\\" \"u\" ({bmp_expr})"));
    }
    if let Some(surr_expr) = surrogate_hex_complement(0, &excluded_surr) {
        alts.push(surr_expr);
    }

    alts.join(" | ")
}

/// Formats a JSON string token (`"foo"`) as a GBNF terminal literal (`"\"foo\""` + optional ws).
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

#[derive(Debug, Clone)]
enum GbnfItem {
    Terminal,
    RuleRef(String),
    Group(Vec<Vec<GbnfItem>>),
}

fn parse_gbnf_expr(expr: &str) -> Vec<Vec<GbnfItem>> {
    fn parse_seq_alts(bytes: &[u8], pos: &mut usize, stop_on_rparen: bool) -> Vec<Vec<GbnfItem>> {
        let mut alts: Vec<Vec<GbnfItem>> = vec![Vec::new()];
        while *pos < bytes.len() {
            let b = bytes[*pos];
            if b.is_ascii_whitespace() {
                *pos += 1;
                continue;
            }
            if stop_on_rparen && b == b')' {
                *pos += 1;
                break;
            }
            if b == b'|' {
                *pos += 1;
                alts.push(Vec::new());
                continue;
            }
            let item = if b == b'"' {
                *pos += 1;
                while *pos < bytes.len() {
                    if bytes[*pos] == b'\\' {
                        *pos = (*pos + 2).min(bytes.len());
                    } else if bytes[*pos] == b'"' {
                        *pos += 1;
                        break;
                    } else {
                        *pos += 1;
                    }
                }
                GbnfItem::Terminal
            } else if b == b'[' {
                *pos += 1;
                while *pos < bytes.len() {
                    if bytes[*pos] == b'\\' {
                        *pos = (*pos + 2).min(bytes.len());
                    } else if bytes[*pos] == b']' {
                        *pos += 1;
                        break;
                    } else {
                        *pos += 1;
                    }
                }
                GbnfItem::Terminal
            } else if b == b'(' {
                *pos += 1;
                let inner = parse_seq_alts(bytes, pos, true);
                GbnfItem::Group(inner)
            } else {
                let start = *pos;
                while *pos < bytes.len()
                    && (bytes[*pos].is_ascii_alphanumeric() || bytes[*pos] == b'-')
                {
                    *pos += 1;
                }
                if *pos == start {
                    *pos += 1;
                    continue;
                }
                let ident = std::str::from_utf8(&bytes[start..*pos])
                    .unwrap_or("")
                    .to_string();
                GbnfItem::RuleRef(ident)
            };

            // Check optional / repetition suffix (`?`, `*`, `+`, `{m,n}`).
            let mut can_skip = false;
            if *pos < bytes.len() {
                match bytes[*pos] {
                    b'?' | b'*' => {
                        can_skip = true;
                        *pos += 1;
                    }
                    b'+' => {
                        *pos += 1;
                    }
                    b'{' => {
                        let start = *pos + 1;
                        *pos += 1;
                        while *pos < bytes.len() && bytes[*pos] != b'}' {
                            *pos += 1;
                        }
                        let range_str = std::str::from_utf8(&bytes[start..*pos]).unwrap_or("");
                        if range_str.trim_start().starts_with("0")
                            && (range_str.trim() == "0" || range_str.contains(','))
                        {
                            can_skip = true;
                        }
                        if *pos < bytes.len() {
                            *pos += 1;
                        }
                    }
                    _ => {}
                }
            }

            let wrapped = if can_skip {
                // `(item)?` = `[ [item], [] ]`
                GbnfItem::Group(vec![vec![item], Vec::new()])
            } else {
                item
            };
            alts.last_mut().unwrap().push(wrapped);
        }
        alts
    }

    let mut pos = 0usize;
    parse_seq_alts(expr.as_bytes(), &mut pos, false)
}

fn alts_are_nullable(alts: &[Vec<GbnfItem>], nullable: &BTreeSet<String>) -> bool {
    alts.iter().any(|seq| {
        seq.iter().all(|item| match item {
            GbnfItem::Terminal => false,
            GbnfItem::RuleRef(r) => nullable.contains(r),
            GbnfItem::Group(inner) => alts_are_nullable(inner, nullable),
        })
    })
}

fn expr_is_nullable(expr: &str, nullable: &BTreeSet<String>) -> bool {
    let alts = parse_gbnf_expr(expr);
    alts_are_nullable(&alts, nullable)
}

fn collect_alts_zero_input(
    alts: &[Vec<GbnfItem>],
    user_rules: &BTreeMap<String, String>,
    nullable: &BTreeSet<String>,
    out: &mut BTreeSet<String>,
) {
    for seq in alts {
        for item in seq {
            match item {
                GbnfItem::Terminal => break,
                GbnfItem::RuleRef(r) => {
                    if user_rules.contains_key(r) {
                        out.insert(r.clone());
                    }
                    if !nullable.contains(r) {
                        break;
                    }
                }
                GbnfItem::Group(inner) => {
                    collect_alts_zero_input(inner, user_rules, nullable, out);
                    if !alts_are_nullable(inner, nullable) {
                        break;
                    }
                }
            }
        }
    }
}

fn collect_zero_input_targets(
    expr: &str,
    user_rules: &BTreeMap<String, String>,
    nullable: &BTreeSet<String>,
    out: &mut BTreeSet<String>,
) {
    let alts = parse_gbnf_expr(expr);
    collect_alts_zero_input(&alts, user_rules, nullable, out);
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
        assert!(
            gbnf.lines()
                .any(|l| l.contains("notes ::= json-string | json-null")),
            "expected notes union rule to include json-null, got:\n{gbnf}"
        );
    }

    #[test]
    fn supports_guarded_recursive_schema_and_rejects_unguarded_and_cached_alias_cycles() {
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

        let object_mutual = json!({
            "$defs": {
                "A": {"type": "object", "properties": {"b": {"$ref": "#/$defs/B"}}},
                "B": {"$ref": "#/$defs/A"}
            },
            "$ref": "#/$defs/A"
        });
        assert!(json_schema_to_gbnf(&object_mutual).is_ok());

        // Cached alias cycle: first branch caches B -> A inside guarded object, second branch
        // references B directly creating a 0-input A -> B -> A loop.
        let cached_alias_cycle = json!({
            "$defs": {
                "A": {
                    "anyOf": [
                        {"type": "object", "properties": {"b": {"$ref": "#/$defs/B"}}},
                        {"$ref": "#/$defs/B"}
                    ]
                },
                "B": {"$ref": "#/$defs/A"}
            },
            "$ref": "#/$defs/A"
        });
        let err = json_schema_to_gbnf(&cached_alias_cycle).unwrap_err();
        assert!(err.to_string().contains("cycle"));
    }

    #[test]
    fn handles_false_branches_sibling_constraints_allof_oneof_and_bounds() {
        // `false` branch in `anyOf` is dropped; `items: false` permits `[]`.
        let g1 = json_schema_to_gbnf(&json!({"anyOf": [false, {"type": "string"}]})).unwrap();
        assert!(g1.contains("root ::= ws json-string"));

        let g2 = json_schema_to_gbnf(&json!({"type": "array", "items": false})).unwrap();
        assert!(g2.contains("\"[\" ws \"]\" ws"));

        // Sibling `type: "string"` excludes `"null"` from `anyOf`.
        let g3 = json_schema_to_gbnf(&json!({
            "type": "string",
            "anyOf": [{"type": "string"}, {"type": "null"}]
        }))
        .unwrap();
        assert!(g3.contains("root ::= ws json-string"));

        // `allOf` preserves array `type` and merges overlapping property bounds (`minLength` + `maxLength`).
        let g4 = json_schema_to_gbnf(&json!({
            "allOf": [
                {"type": "object", "properties": {"name": {"type": ["string", "null"], "minLength": 2}}},
                {"type": "object", "properties": {"name": {"type": "string", "maxLength": 4}}, "required": ["name"]}
            ]
        }))
        .unwrap();
        assert!(g4.contains("json-char{2,4}"));

        // `oneOf` with `number` and `integer` compiles to `json-nonint-number`; overlapping strings fail.
        let g5 = json_schema_to_gbnf(&json!({
            "oneOf": [{"type": "number"}, {"type": "integer"}]
        }))
        .unwrap();
        assert!(g5.contains("json-nonint-number"));
        assert!(
            json_schema_to_gbnf(&json!({"oneOf": [{"type": "string"}, {"type": "string"}]}))
                .is_err()
        );

        // Unsupported numeric assertions and out-of-range bounds fail closed.
        assert!(json_schema_to_gbnf(&json!({"type": "integer", "minimum": 1})).is_err());
        assert!(json_schema_to_gbnf(&json!({"type": "string", "minLength": 4097})).is_err());
        assert!(json_schema_to_gbnf(&json!({"type": "array", "minItems": 65})).is_err());

        // `required` without `properties` and `additionalProperties: false` conflict.
        let g6 = json_schema_to_gbnf(&json!({"type": "object", "required": ["id"]})).unwrap();
        assert!(g6.contains("\\\"id\\\""));
        assert!(
            json_schema_to_gbnf(&json!({
                "type": "object",
                "properties": {},
                "required": ["id"],
                "additionalProperties": false
            }))
            .is_err()
        );

        // Local JSON pointer through array index (`#/anyOf/0`).
        let g7 = json_schema_to_gbnf(&json!({
            "anyOf": [{"type": "boolean"}, {"$ref": "#/anyOf/0"}]
        }))
        .unwrap();
        assert!(g7.contains("json-bool"));

        // `prefixItems` respects `maxItems: 0`.
        let g8 = json_schema_to_gbnf(&json!({
            "type": "array",
            "prefixItems": [{"type": "integer"}],
            "maxItems": 0
        }))
        .unwrap();
        assert!(g8.contains("\"[\" ws \"]\" ws"));
    }
}
