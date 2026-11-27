"""Dataclass / Pydantic JSON Schema extraction and structured output parsing (Python 3.15+)."""

import functools
import sys
import types
import typing
from collections.abc import Mapping, Sequence
from typing import Any, Literal, get_args, get_origin
lazy import dataclasses
lazy import enum
lazy import importlib
lazy import json

from ._layerstream import (
    compile_response_format as _native_compile_response_format,
    json_schema_to_gbnf,
)
from ._types import JsonDict


@functools.cache
def _get_pydantic() -> Any:
    try:
        return importlib.import_module("pydantic")
    except ImportError:
        return None


def _annotation_to_json_schema(
    tp: Any, field_label: str, *, _seen: frozenset[type[Any]] = frozenset()
) -> JsonDict:
    if tp is Any:
        return {}
    if tp is None or tp is types.NoneType:
        return {"type": "null"}
    if tp is str:
        return {"type": "string"}
    if tp is int:
        return {"type": "integer"}
    if tp is float:
        return {"type": "number"}
    if tp is bool:
        return {"type": "boolean"}
    if tp in (list, tuple, Sequence):
        return {"type": "array"}
    if tp in (dict, Mapping):
        return {"type": "object"}

    if isinstance(tp, type):
        cls = typing.cast(type[Any], tp)
        if issubclass(cls, enum.Enum):
            return {"enum": [member.value for member in cls]}
        if dataclasses.is_dataclass(cls):
            return _dataclass_to_json_schema(cls, _seen=_seen)
        if hasattr(cls, "model_json_schema"):
            schema_fn: Any = cls.model_json_schema
            return dict(typing.cast(Mapping[str, Any], schema_fn()))

    origin = get_origin(tp)
    args = get_args(tp)
    if origin is types.UnionType:
        return {"anyOf": [_annotation_to_json_schema(a, field_label, _seen=_seen) for a in args]}
    if origin is Literal:
        for a in args:
            if not isinstance(a, (str, int, float, bool, type(None))):
                raise TypeError(f"unsupported Literal value {a!r} on '{field_label}'")
        return {"enum": list(args)}
    if origin in (list, Sequence):
        return (
            {
                "type": "array",
                "items": _annotation_to_json_schema(args[0], field_label, _seen=_seen),
            }
            if args
            else {"type": "array"}
        )
    if origin is tuple:
        if not args:
            if getattr(tp, "__args__", None) == ():
                return {"type": "array", "maxItems": 0}
            return {"type": "array"}
        if len(args) == 2 and args[1] is Ellipsis:
            return {
                "type": "array",
                "items": _annotation_to_json_schema(args[0], field_label, _seen=_seen),
            }
        pfx = [_annotation_to_json_schema(a, field_label, _seen=_seen) for a in args]
        return {
            "type": "array",
            "prefixItems": pfx,
            "minItems": len(pfx),
            "maxItems": len(pfx),
        }
    if origin in (dict, Mapping):
        if not args:
            return {"type": "object"}
        if args[0] is not str:
            raise TypeError(f"JSON object keys in '{field_label}' must be 'str', got {args[0]!r}")
        return {
            "type": "object",
            "additionalProperties": _annotation_to_json_schema(args[1], field_label, _seen=_seen),
        }

    raise TypeError(f"unsupported dataclass field annotation {tp!r} on '{field_label}'")


def _resolve_dataclass_field_types(dc_cls: type[Any]) -> dict[str, Any]:
    """Resolve PEP 649 deferred annotations without `eval()`."""
    mod_ns = getattr(sys.modules.get(dc_cls.__module__), "__dict__", None)
    return typing.get_type_hints(dc_cls, globalns=mod_ns)


def _dataclass_to_json_schema(
    dc_cls: type[Any], *, _seen: frozenset[type[Any]] = frozenset()
) -> JsonDict:
    if dc_cls in _seen:
        raise ValueError(
            f"recursive dataclass '{dc_cls.__name__}' is not supported without pydantic"
        )
    if not _seen and (pydantic := _get_pydantic()) is not None:
        return dict(typing.cast(Mapping[str, Any], pydantic.TypeAdapter(dc_cls).json_schema()))

    next_seen = _seen | {dc_cls}
    resolved_types = _resolve_dataclass_field_types(dc_cls)
    props: JsonDict = {}
    required: list[str] = []
    for f in dataclasses.fields(dc_cls):
        py_t = resolved_types.get(f.name, f.type)
        props[f.name] = _annotation_to_json_schema(
            py_t, f"{dc_cls.__name__}.{f.name}", _seen=next_seen
        )
        if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
            required.append(f.name)
    return {
        "type": "object",
        "properties": props,
        "required": required,
        "additionalProperties": False,
    }


def _coerce_value_for_type(val: Any, tp: Any) -> Any:
    if val is None or isinstance(tp, str):
        return val
    if isinstance(tp, type):
        cls = typing.cast(type[Any], tp)
        if issubclass(cls, enum.Enum):
            return cls(val)
        if dataclasses.is_dataclass(cls) and isinstance(val, Mapping):
            return _instantiate_dataclass(cls, typing.cast(Mapping[str, Any], val))
        return val
    origin = get_origin(tp)
    args = get_args(tp)
    if origin is types.UnionType:
        for branch in args:
            try:
                return _coerce_value_for_type(val, branch)
            except TypeError, ValueError, KeyError:
                continue
        return val
    if origin in (list, Sequence) and args and isinstance(val, list):
        items = typing.cast(list[Any], val)
        return [_coerce_value_for_type(v, args[0]) for v in items]
    if origin is tuple and isinstance(val, list):
        items = typing.cast(list[Any], val)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce_value_for_type(v, args[0]) for v in items)
        if args and len(args) == len(items):
            return tuple(_coerce_value_for_type(v, a) for v, a in zip(items, args, strict=True))
        return tuple(items)
    if origin in (dict, Mapping) and len(args) == 2 and isinstance(val, Mapping):
        mapping = typing.cast(Mapping[Any, Any], val)
        return {k: _coerce_value_for_type(v, args[1]) for k, v in mapping.items()}
    return val


def _instantiate_dataclass(dc_cls: type[Any], data: Mapping[str, Any]) -> Any:
    if (pydantic := _get_pydantic()) is not None:
        return pydantic.TypeAdapter(dc_cls).validate_python(data)

    resolved_types = _resolve_dataclass_field_types(dc_cls)
    kwargs = {
        k: _coerce_value_for_type(v, resolved_types[k]) if k in resolved_types else v
        for k, v in data.items()
    }
    return dc_cls(**kwargs)


def compile_response_format(response_format: Any) -> tuple[str | None, Any]:
    if response_format is None:
        return None, None

    if isinstance(response_format, type):
        cls = typing.cast(type[Any], response_format)
        if hasattr(cls, "model_json_schema"):
            return json_schema_to_gbnf(cls), cls
        if dataclasses.is_dataclass(cls):
            schema = _dataclass_to_json_schema(cls)
            return json_schema_to_gbnf(schema), cls
        raise TypeError(f"unsupported response_format class '{cls.__name__}'")

    if isinstance(response_format, Mapping):
        grammar, expects_json = _native_compile_response_format(response_format, False)
        return grammar, (dict if expects_json else None)

    raise TypeError(
        "response_format must be None, a mapping ({'type': ...}), or a Pydantic/dataclass type"
    )


def parse_structured_text(raw_text: str, parser_target: Any) -> Any:
    if parser_target is None:
        return None
    if hasattr(parser_target, "model_validate_json"):
        return parser_target.model_validate_json(raw_text)
    data = json.loads(raw_text)
    if isinstance(parser_target, type) and dataclasses.is_dataclass(parser_target):
        if not isinstance(data, Mapping):
            raise ValueError("expected JSON object for dataclass response_format")
        return _instantiate_dataclass(
            typing.cast(type[Any], parser_target), typing.cast(Mapping[str, Any], data)
        )
    return data
