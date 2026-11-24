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
