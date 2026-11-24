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
        json_schema_to_gbnf(&json!({"oneOf": [{"type": "string"}, {"type": "string"}]})).is_err()
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

/// Verifies every rule referenced on any RHS of a GBNF grammar is defined on some LHS.
fn assert_all_gbnf_rules_defined(gbnf: &str) {
    use std::collections::HashSet;

    let mut defined = HashSet::new();
    let mut rhs_list = Vec::new();
    for line in gbnf.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let (lhs, rhs) = trimmed
            .split_once("::=")
            .unwrap_or_else(|| panic!("malformed GBNF line: {trimmed}"));
        defined.insert(lhs.trim().to_string());
        rhs_list.push(rhs.to_string());
    }
    assert!(defined.contains("root"), "grammar must define `root`");

    for rhs in rhs_list {
        let bytes = rhs.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            match bytes[i] {
                b'"' => {
                    i += 1;
                    while i < bytes.len() {
                        if bytes[i] == b'\\' {
                            i += 2;
                        } else if bytes[i] == b'"' {
                            i += 1;
                            break;
                        } else {
                            i += 1;
                        }
                    }
                }
                b'[' => {
                    i += 1;
                    while i < bytes.len() {
                        if bytes[i] == b'\\' {
                            i += 2;
                        } else if bytes[i] == b']' {
                            i += 1;
                            break;
                        } else {
                            i += 1;
                        }
                    }
                }
                b'a'..=b'z' | b'A'..=b'Z' => {
                    let start = i;
                    while i < bytes.len()
                        && (bytes[i].is_ascii_alphanumeric()
                            || bytes[i] == b'-'
                            || bytes[i] == b'_')
                    {
                        i += 1;
                    }
                    let ident = &rhs[start..i];
                    assert!(
                        defined.contains(ident),
                        "GBNF references undefined rule `{ident}` in:\n{gbnf}"
                    );
                }
                _ => i += 1,
            }
        }
    }
}

#[test]
fn json_object_gbnf_is_self_contained_and_well_formed() {
    let gbnf = json_object_gbnf();
    assert!(gbnf.starts_with("root ::= ws json-object\n"));
    assert_all_gbnf_rules_defined(&gbnf);
}

#[test]
fn compiles_primitives_const_and_nullable_type_arrays() {
    for (ty, expected) in [
        ("null", "json-null"),
        ("boolean", "json-bool"),
        ("integer", "json-int"),
        ("number", "json-number"),
        ("string", "json-string"),
    ] {
        let g = json_schema_to_gbnf(&json!({"type": ty})).unwrap();
        assert!(g.contains(expected), "type={ty} missing {expected}");
        assert_all_gbnf_rules_defined(&g);
    }

    let nullable = json_schema_to_gbnf(&json!({"type": ["integer", "null"]})).unwrap();
    assert!(nullable.contains("json-int"));
    assert!(nullable.contains("json-null"));
    assert_all_gbnf_rules_defined(&nullable);

    let const_obj = json_schema_to_gbnf(&json!({
        "const": {"ok": true, "count": 3, "tags": ["a", null]}
    }))
    .unwrap();
    assert!(const_obj.contains("\\\"ok\\\""));
    assert!(const_obj.contains("true"));
    assert!(const_obj.contains("3"));
    assert_all_gbnf_rules_defined(&const_obj);
}

#[test]
fn compiles_typed_additional_properties_and_excluded_key_rules() {
    let schema = json!({
        "type": "object",
        "properties": {
            "id": {"type": "integer"},
            "label": {"type": "string"}
        },
        "required": ["id"],
        "additionalProperties": {"type": "boolean"}
    });
    let gbnf = json_schema_to_gbnf(&schema).unwrap();
    assert!(gbnf.contains("\\\"id\\\""));
    assert!(gbnf.contains("json-bool"));
    assert_all_gbnf_rules_defined(&gbnf);

    // Pure map object (`additionalProperties` without `properties`).
    let map_schema = json!({
        "type": "object",
        "additionalProperties": {"type": "number"}
    });
    let map_gbnf = json_schema_to_gbnf(&map_schema).unwrap();
    assert!(map_gbnf.contains("json-number"));
    assert_all_gbnf_rules_defined(&map_gbnf);
}

#[test]
fn compiles_string_and_array_bounds_and_prefix_items_tuples() {
    let exact_str = json_schema_to_gbnf(&json!({
        "type": "string",
        "minLength": 3,
        "maxLength": 3
    }))
    .unwrap();
    assert!(exact_str.contains("json-char{3}"));
    assert_all_gbnf_rules_defined(&exact_str);

    // minLength > maxLength is unsatisfiable.
    assert!(
        json_schema_to_gbnf(&json!({"type": "string", "minLength": 5, "maxLength": 2})).is_err()
    );

    let tuple = json_schema_to_gbnf(&json!({
        "type": "array",
        "prefixItems": [{"type": "string"}, {"type": "integer"}],
        "items": {"type": "boolean"},
        "minItems": 2,
        "maxItems": 4
    }))
    .unwrap();
    assert!(tuple.contains("json-string"));
    assert!(tuple.contains("json-int"));
    assert!(tuple.contains("json-bool"));
    assert_all_gbnf_rules_defined(&tuple);

    // minItems > maxItems fails closed.
    assert!(json_schema_to_gbnf(&json!({"type": "array", "minItems": 3, "maxItems": 1})).is_err());
}

#[test]
fn handles_json_pointer_escaping_and_rejects_invalid_refs_and_unsupported_keywords() {
    // RFC 6901 JSON Pointer escaping: `~1` -> `/`, `~0` -> `~`.
    let escaped = json!({
        "$defs": {
            "a/b~c": {"type": "integer"}
        },
        "$ref": "#/$defs/a~1b~0c"
    });
    let g = json_schema_to_gbnf(&escaped).unwrap();
    assert!(g.contains("json-int"));
    assert_all_gbnf_rules_defined(&g);

    // Missing local ref, external ref, and unsupported keywords (`format`, `pattern`) fail closed.
    let err_missing = json_schema_to_gbnf(&json!({"$ref": "#/$defs/Nope"})).unwrap_err();
    assert!(err_missing.to_string().contains("unresolved"));

    assert!(json_schema_to_gbnf(&json!({"$ref": "https://example.com/s.json"})).is_err());
    assert!(json_schema_to_gbnf(&json!({"type": "string", "format": "uuid"})).is_err());
    assert!(json_schema_to_gbnf(&json!({"type": "string", "pattern": "^[a-z]+$"})).is_err());
    assert!(json_schema_to_gbnf(&json!(false)).is_err());
}

#[test]
fn compiles_const_and_enum_with_sibling_constraints_and_escaped_literals() {
    // Rich literal escaping: control chars, quotes, backslashes, non-ASCII UTF-8, floats, negatives.
    let g = json_schema_to_gbnf(&json!({
        "enum": [
            "line1\nline2\t\"quoted\"\\slash\u{0007}café🚀",
            -42,
            3.25,
            true,
            false,
            null,
            {"k": [1, "v"]}
        ]
    }))
    .unwrap();
    assert_all_gbnf_rules_defined(&g);

    // `enum` filtered by sibling `type`, `minLength`, `maxLength`.
    let filtered = json_schema_to_gbnf(&json!({
        "type": "string",
        "minLength": 2,
        "maxLength": 4,
        "enum": ["a", "ok", "good", "toolong", 123, null]
    }))
    .unwrap();
    assert!(filtered.contains("\\\"ok\\\""));
    assert!(filtered.contains("\\\"good\\\""));
    assert!(!filtered.contains("\\\"toolong\\\""));
    assert_all_gbnf_rules_defined(&filtered);

    // `const` satisfied vs contradicted by sibling `type` / `minItems` / `required`.
    let const_arr = json_schema_to_gbnf(&json!({
        "type": "array",
        "minItems": 1,
        "maxItems": 2,
        "items": {"type": "integer"},
        "const": [1, 2]
    }))
    .unwrap();
    assert_all_gbnf_rules_defined(&const_arr);

    assert!(
        json_schema_to_gbnf(&json!({
            "type": "array",
            "minItems": 3,
            "const": [1, 2]
        }))
        .is_err()
    );

    let const_obj = json_schema_to_gbnf(&json!({
        "type": "object",
        "required": ["a"],
        "properties": {"a": {"type": "integer"}},
        "additionalProperties": false,
        "const": {"a": 1}
    }))
    .unwrap();
    assert_all_gbnf_rules_defined(&const_obj);

    // Contradicted by missing `required` key.
    assert!(
        json_schema_to_gbnf(&json!({
            "type": "object",
            "required": ["missing_key"],
            "const": {"a": 1}
        }))
        .is_err()
    );

    // Schema numeric equality: `1` == `1.0`.
    let num_eq = json_schema_to_gbnf(&json!({
        "allOf": [{"const": 1}, {"const": 1.0}]
    }))
    .unwrap();
    assert_all_gbnf_rules_defined(&num_eq);
}

#[test]
fn compiles_allof_intersections_across_objects_arrays_strings_and_enums() {
    // Intersect two objects with overlapping properties, required sets, and typed additionalProperties.
    let obj_allof = json_schema_to_gbnf(&json!({
        "allOf": [
            {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "tag": {"type": "string", "minLength": 1}
                },
                "required": ["id"],
                "additionalProperties": {"type": ["string", "boolean"]}
            },
            {
                "type": "object",
                "properties": {
                    "tag": {"type": "string", "maxLength": 8},
                    "flag": {"type": "boolean"}
                },
                "required": ["flag"],
                "additionalProperties": {"type": ["string", "integer"]}
            }
        ]
    }))
    .unwrap();
    assert!(obj_allof.contains("json-char{1,8}"));
    assert!(obj_allof.contains("\\\"id\\\""));
    assert!(obj_allof.contains("\\\"flag\\\""));
    assert_all_gbnf_rules_defined(&obj_allof);

    // Intersect two arrays with prefixItems, items, minItems, maxItems.
    let arr_allof = json_schema_to_gbnf(&json!({
        "allOf": [
            {
                "type": "array",
                "prefixItems": [{"type": ["string", "integer"]}],
                "items": {"type": ["boolean", "null"]},
                "minItems": 1
            },
            {
                "type": "array",
                "prefixItems": [{"type": "string"}],
                "items": {"type": "boolean"},
                "maxItems": 3
            }
        ]
    }))
    .unwrap();
    assert_all_gbnf_rules_defined(&arr_allof);

    // Intersect enums & consts.
    let enum_allof = json_schema_to_gbnf(&json!({
        "allOf": [
            {"enum": ["a", "b", "c"]},
            {"enum": ["b", "c", "d"]}
        ]
    }))
    .unwrap();
    assert!(enum_allof.contains("\\\"b\\\""));
    assert!(enum_allof.contains("\\\"c\\\""));
    assert!(!enum_allof.contains("\\\"a\\\""));
    assert_all_gbnf_rules_defined(&enum_allof);

    // Disjoint enum intersection is unsatisfiable.
    assert!(
        json_schema_to_gbnf(&json!({
            "allOf": [{"enum": ["a"]}, {"enum": ["b"]}]
        }))
        .is_err()
    );

    // `allOf` with `true` and `false` schemas.
    let with_true = json_schema_to_gbnf(&json!({
        "allOf": [true, {"type": "boolean"}]
    }))
    .unwrap();
    assert!(with_true.contains("json-bool"));
    assert!(json_schema_to_gbnf(&json!({"allOf": [false, {"type": "boolean"}]})).is_err());

    // `allOf` distributing over `anyOf`.
    let dist = json_schema_to_gbnf(&json!({
        "allOf": [
            {"anyOf": [{"type": "string"}, {"type": "integer"}]},
            {"type": "integer"}
        ]
    }))
    .unwrap();
    assert!(dist.contains("json-int"));
    assert_all_gbnf_rules_defined(&dist);
}

#[test]
fn compiles_oneof_disjoint_and_rejects_overlapping_branches() {
    // Disjoint primitive types + const/null.
    let g = json_schema_to_gbnf(&json!({
        "oneOf": [
            {"type": "string"},
            {"type": "boolean"},
            {"type": "null"}
        ]
    }))
    .unwrap();
    assert!(g.contains("json-string"));
    assert!(g.contains("json-bool"));
    assert!(g.contains("json-null"));
    assert_all_gbnf_rules_defined(&g);

    // `oneOf` with `const` and disjoint `enum`.
    let g2 = json_schema_to_gbnf(&json!({
        "oneOf": [
            {"const": "alpha"},
            {"enum": ["beta", "gamma"]}
        ]
    }))
    .unwrap();
    assert_all_gbnf_rules_defined(&g2);

    // `oneOf` subtracts overlapping `"alpha"` leaving only `"beta"`.
    let g_sub = json_schema_to_gbnf(&json!({
        "oneOf": [
            {"const": "alpha"},
            {"enum": ["alpha", "beta"]}
        ]
    }))
    .unwrap();
    assert!(g_sub.contains("\\\"beta\\\""));
    assert!(!g_sub.contains("\\\"alpha\\\""));
    assert_all_gbnf_rules_defined(&g_sub);

    // `oneOf` with single valid branch and a `false` branch.
    let g3 = json_schema_to_gbnf(&json!({
        "oneOf": [false, {"type": "integer"}]
    }))
    .unwrap();
    assert!(g3.contains("json-int"));
    assert_all_gbnf_rules_defined(&g3);
}

#[test]
fn compiles_object_with_excluded_keys_containing_escapes_and_unicode() {
    // Exercises `compile_string_excluding_keys` and `gbnf.rs` UTF-8 / escape / surrogate logic,
    // plus `build_ordered_object_fields` with optional + required fields.
    let schema = json!({
        "type": "object",
        "properties": {
            "": {"type": "null"},
            "a\"b\\c": {"type": "string"},
            "café": {"type": "integer"},
            "🚀": {"type": "boolean"},
            "opt1": {"type": "number"},
            "opt2": {"type": "string"}
        },
        "required": ["café"],
        "additionalProperties": {"type": "integer"}
    });
    let gbnf = json_schema_to_gbnf(&schema).unwrap();
    assert_all_gbnf_rules_defined(&gbnf);

    // All-optional object with `additionalProperties: false`.
    let all_opt = json_schema_to_gbnf(&json!({
        "type": "object",
        "properties": {
            "x": {"type": "integer"},
            "y": {"type": "string"},
            "z": {"type": "boolean"}
        },
        "additionalProperties": false
    }))
    .unwrap();
    assert_all_gbnf_rules_defined(&all_opt);

    // Empty object (`additionalProperties: false` and no properties).
    let empty_obj = json_schema_to_gbnf(&json!({
        "type": "object",
        "additionalProperties": false
    }))
    .unwrap();
    assert_all_gbnf_rules_defined(&empty_obj);
}

#[test]
fn validates_error_paths_annotations_and_boolean_schemas() {
    // `true` and annotation-only schemas compile to `json-value`.
    let g_true = json_schema_to_gbnf(&json!(true)).unwrap();
    assert!(g_true.contains("root ::= ws json-value"));
    assert_all_gbnf_rules_defined(&g_true);

    let g_annot = json_schema_to_gbnf(&json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Anything",
        "description": "Annotations only",
        "x-custom": 42
    }))
    .unwrap();
    assert!(g_annot.contains("root ::= ws json-value"));

    // Invalid root schema type.
    assert!(json_schema_to_gbnf(&json!(123)).is_err());

    // Invalid non-negative bounds (negative, float, string).
    assert!(json_schema_to_gbnf(&json!({"type": "string", "minLength": -1})).is_err());
    assert!(json_schema_to_gbnf(&json!({"type": "string", "maxLength": 1.5})).is_err());
    assert!(json_schema_to_gbnf(&json!({"type": "array", "minItems": "two"})).is_err());

    // Direct self-referential cycle at root fails closed.
    assert!(json_schema_to_gbnf(&json!({"$ref": "#"})).is_err());

    // `GrammarError::from(serde_json::Error)` conversion works.
    let serde_err = serde_json::from_str::<Value>("{bad").unwrap_err();
    let g_err: GrammarError = serde_err.into();
    assert!(!g_err.to_string().is_empty());
}

#[test]
fn compiles_oneof_discriminated_unions_number_xor_int_and_allof_distribution() {
    // Pydantic-style discriminated union (`oneOf` of objects with disjoint `const` tag).
    let disc = json_schema_to_gbnf(&json!({
        "oneOf": [
            {
                "type": "object",
                "properties": {"kind": {"const": "a"}, "val": {"type": "integer"}},
                "required": ["kind", "val"],
                "additionalProperties": false
            },
            {
                "type": "object",
                "properties": {"kind": {"const": "b"}, "msg": {"type": "string"}},
                "required": ["kind", "msg"],
                "additionalProperties": false
            }
        ]
    }))
    .unwrap();
    assert!(disc.contains("\\\"a\\\""));
    assert!(disc.contains("\\\"b\\\""));
    assert_all_gbnf_rules_defined(&disc);

    // `oneOf` with non-discriminated same-type array/string branches fails closed.
    assert!(
        json_schema_to_gbnf(&json!({
            "oneOf": [
                {"type": "array", "items": {"type": "integer"}, "maxItems": 1},
                {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 3}
            ]
        }))
        .is_err()
    );
    assert!(
        json_schema_to_gbnf(&json!({
            "oneOf": [
                {"type": "string", "maxLength": 2},
                {"type": "string", "minLength": 4, "maxLength": 6}
            ]
        }))
        .is_err()
    );

    // `oneOf` with `(number XOR integer)` + a third disjoint branch (`string`).
    let num_xor_str = json_schema_to_gbnf(&json!({
        "oneOf": [{"type": "number"}, {"type": "integer"}, {"type": "string"}]
    }))
    .unwrap();
    assert!(num_xor_str.contains("json-nonint-number"));
    assert!(num_xor_str.contains("json-string"));
    assert_all_gbnf_rules_defined(&num_xor_str);

    // `oneOf` with `(number XOR integer)` + two overlapping extra branches fails closed.
    assert!(
        json_schema_to_gbnf(&json!({
            "oneOf": [
                {"type": "number"},
                {"type": "integer"},
                {"type": "string"},
                {"type": "string"}
            ]
        }))
        .is_err()
    );

    // `oneOf` edge cases: empty, single-element, all-false.
    assert!(json_schema_to_gbnf(&json!({"oneOf": []})).is_err());
    assert!(json_schema_to_gbnf(&json!({"oneOf": [false, false]})).is_err());
    let single_oneof = json_schema_to_gbnf(&json!({"oneOf": [{"type": "boolean"}]})).unwrap();
    assert!(single_oneof.contains("json-bool"));

    // `allOf` distributing over `oneOf` on both left and right sides, plus multi-type intersection.
    let left_oneof = json_schema_to_gbnf(&json!({
        "allOf": [
            {"oneOf": [{"type": "string"}, {"type": "integer"}]},
            {"type": ["string", "boolean"]}
        ]
    }))
    .unwrap();
    assert!(left_oneof.contains("json-string"));
    assert_all_gbnf_rules_defined(&left_oneof);

    let right_oneof = json_schema_to_gbnf(&json!({
        "allOf": [
            {"type": ["string", "integer", "boolean"]},
            {"oneOf": [{"type": "string"}, {"type": "boolean"}]}
        ]
    }))
    .unwrap();
    assert!(right_oneof.contains("json-string"));
    assert!(right_oneof.contains("json-bool"));
    assert_all_gbnf_rules_defined(&right_oneof);

    // `allOf` with enum/items/additionalProperties on only one side and multi-type overlap.
    let multi_ty = json_schema_to_gbnf(&json!({
        "allOf": [
            {"type": ["string", "integer", "boolean"]},
            {"type": ["string", "boolean"]}
        ]
    }))
    .unwrap();
    assert!(multi_ty.contains("json-string"));
    assert!(multi_ty.contains("json-bool"));
    assert_all_gbnf_rules_defined(&multi_ty);

    let one_sided = json_schema_to_gbnf(&json!({
        "allOf": [
            {"type": "array", "items": {"type": "integer"}},
            {"type": "array", "minItems": 1, "maxItems": 2}
        ]
    }))
    .unwrap();
    assert_all_gbnf_rules_defined(&one_sided);

    let one_sided_obj = json_schema_to_gbnf(&json!({
        "allOf": [
            {"type": "object", "additionalProperties": {"type": "boolean"}},
            {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}
        ]
    }))
    .unwrap();
    assert_all_gbnf_rules_defined(&one_sided_obj);
}
