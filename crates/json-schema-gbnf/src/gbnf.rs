use super::*;

pub(super) fn build_homogeneous_array_expr(
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

pub(super) fn extract_required_list(obj: &Map<String, Value>, hint: &str) -> Result<Vec<String>> {
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

pub(super) fn extract_explicit_types(
    obj: &Map<String, Value>,
    hint: &str,
) -> Result<Option<Vec<String>>> {
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

pub(super) fn intersect_type_sets(
    a: Option<&Vec<String>>,
    b: Option<&Vec<String>>,
) -> Option<Vec<String>> {
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

pub(super) fn extract_literals_if_present(
    obj: &Map<String, Value>,
    hint: &str,
) -> Result<Option<Vec<Value>>> {
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

pub(super) fn extract_pure_literals(schema: &Value) -> Option<Result<Vec<Value>>> {
    let obj = schema.as_object()?;
    if !obj.contains_key("const") && !obj.contains_key("enum") {
        return None;
    }
    if obj.contains_key("anyOf") || obj.contains_key("oneOf") || obj.contains_key("allOf") {
        return None;
    }
    Some(extract_literals_if_present(obj, "oneOf").map(|opt| opt.unwrap_or_default()))
}

pub(super) fn is_plain_single_type(schema: &Value, expected: &str) -> bool {
    let Some(obj) = schema.as_object() else {
        return false;
    };
    if obj.get("type").and_then(Value::as_str) != Some(expected) {
        return false;
    }
    obj.keys().all(|k| k == "type" || is_annotation_key(k))
}

pub(super) fn inferred_canonical_types(schema: &Value) -> Option<BTreeSet<&'static str>> {
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

pub(super) fn canonical_types_for_literal(val: &Value) -> BTreeSet<&'static str> {
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

pub(super) fn schemas_provably_disjoint(a: &Value, b: &Value) -> bool {
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

pub(super) fn max_opt(a: Option<usize>, b: Option<usize>) -> Option<usize> {
    match (a, b) {
        (Some(x), Some(y)) => Some(x.max(y)),
        (Some(x), None) => Some(x),
        (None, Some(y)) => Some(y),
        (None, None) => None,
    }
}

pub(super) fn min_opt(a: Option<usize>, b: Option<usize>) -> Option<usize> {
    match (a, b) {
        (Some(x), Some(y)) => Some(x.min(y)),
        (Some(x), None) => Some(x),
        (None, Some(y)) => Some(y),
        (None, None) => None,
    }
}

pub(super) fn hex_nibble_gbnf(nib: u8) -> &'static str {
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

pub(super) fn hex4_u16_gbnf(code: u16) -> String {
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

pub(super) fn gbnf_json_char_matching(ch: char) -> String {
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

pub(super) fn nibble_char_class(nibbles: &[u8]) -> String {
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

pub(super) fn bmp_hex_complement(
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

pub(super) fn surrogate_hex_complement(
    depth: usize,
    excluded: &BTreeSet<[u8; 8]>,
) -> Option<String> {
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

pub(super) fn gbnf_json_char_excluding(excluded: &BTreeSet<char>) -> String {
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
pub(super) fn gbnf_string_token(s: &str) -> String {
    let json_encoded = serde_json::to_string(s).unwrap_or_else(|_| "\"\"".to_string());
    format!("{} ws", gbnf_literal(&json_encoded))
}

pub(super) fn gbnf_json_value_literal(v: &Value) -> Result<String> {
    let serialized = serde_json::to_string(v)?;
    Ok(gbnf_literal(&serialized))
}

pub(super) fn gbnf_literal(raw: &str) -> String {
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
pub(super) enum GbnfItem {
    Terminal,
    RuleRef(String),
    Group(Vec<Vec<GbnfItem>>),
}

pub(super) fn parse_gbnf_expr(expr: &str) -> Vec<Vec<GbnfItem>> {
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
            if let Some(last) = alts.last_mut() {
                last.push(wrapped);
            }
        }
        alts
    }

    let mut pos = 0usize;
    parse_seq_alts(expr.as_bytes(), &mut pos, false)
}

pub(super) fn alts_are_nullable(alts: &[Vec<GbnfItem>], nullable: &BTreeSet<String>) -> bool {
    alts.iter().any(|seq| {
        seq.iter().all(|item| match item {
            GbnfItem::Terminal => false,
            GbnfItem::RuleRef(r) => nullable.contains(r),
            GbnfItem::Group(inner) => alts_are_nullable(inner, nullable),
        })
    })
}

pub(super) fn expr_is_nullable(expr: &str, nullable: &BTreeSet<String>) -> bool {
    let alts = parse_gbnf_expr(expr);
    alts_are_nullable(&alts, nullable)
}

pub(super) fn collect_alts_zero_input(
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

pub(super) fn collect_zero_input_targets(
    expr: &str,
    user_rules: &BTreeMap<String, String>,
    nullable: &BTreeSet<String>,
    out: &mut BTreeSet<String>,
) {
    let alts = parse_gbnf_expr(expr);
    collect_alts_zero_input(&alts, user_rules, nullable, out);
}
