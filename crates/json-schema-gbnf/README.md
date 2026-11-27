# json-schema-gbnf

Pure-Rust JSON Schema & RFC 8259 JSON-object to `llama.cpp` GBNF grammar compiler.

Extracted from [`layerstream-plugin`](https://github.com/kleinbem/layerstream-plugin) with zero C++ or `llama.cpp` build dependencies — only `serde_json` and `thiserror`.

## Supported JSON Schema features

- OpenAI `response_format` (`json_object` and `json_schema`) and Pydantic v1/v2 schemas
- `$defs`, `definitions`, `$ref` (with guarded recursion and unguarded cycle rejection)
- `anyOf`, `oneOf`, `allOf` (including schema intersection)
- `enum`, `const`, `type` (single or union), `required`, optional fields
- `additionalProperties` (including key-exclusion with Unicode `\uXXXX` handling)
- Bounded strings (`minLength`/`maxLength`), `prefixItems`, and bounded arrays (`minItems`/`maxItems`)

## Usage

```rust
use json_schema_gbnf::{json_object_gbnf, json_schema_to_gbnf};
use serde_json::json;

let schema = json!({
    "type": "object",
    "properties": {
        "capital": { "type": "string" },
        "country": { "type": "string" }
    },
    "required": ["capital", "country"],
    "additionalProperties": false
});

let gbnf = json_schema_to_gbnf(&schema)?;
assert!(gbnf.contains("root ::="));

let any_obj = json_object_gbnf();
assert!(any_obj.contains("json-object"));
# Ok::<(), json_schema_gbnf::GrammarError>(())
```

## License

Apache-2.0
