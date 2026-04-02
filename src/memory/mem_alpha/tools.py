"""OpenAI-style tool schemas and execution for Mem-alpha memory operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .memory_core import AlphaMemory


@dataclass
class Parameter:
    name: str
    type: str
    description: str
    required: bool = True
    enum: Optional[List[str]] = None


class ToolFunction:
    name: str
    description: str
    parameters: List[Parameter]

    @classmethod
    def execute(cls, memory: AlphaMemory, arguments: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    @classmethod
    def to_schema(cls, memory: Optional[AlphaMemory] = None) -> Optional[Dict[str, Any]]:
        properties: Dict[str, Any] = {}
        required: List[str] = []
        skip_tool = False

        for param in cls.parameters:
            param_schema: Dict[str, Any] = {
                "type": param.type,
                "description": param.description,
            }
            if param.enum:
                enum_values = param.enum.copy()
                if param.name == "memory_type" and memory is not None:
                    filtered_enum: List[str] = []
                    for enum_value in enum_values:
                        base_type = enum_value.replace("_memory", "")
                        if hasattr(memory, "is_memory_type_enabled"):
                            is_enabled = memory.is_memory_type_enabled(base_type)
                        elif base_type == "core":
                            is_enabled = getattr(memory, "including_core", False)
                        else:
                            is_enabled = True
                        if is_enabled:
                            filtered_enum.append(enum_value)
                    enum_values = filtered_enum
                    if not enum_values:
                        skip_tool = True
                        break
                param_schema["enum"] = enum_values
            properties[param.name] = param_schema
            if param.required:
                required.append(param.name)

        if skip_tool:
            return None

        return {
            "type": "function",
            "function": {
                "name": cls.name,
                "description": cls.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }


class NewMemoryInsert(ToolFunction):
    name = "new_memory_insert"
    description = (
        "Infer a new memory and append it to a memory store. Creates a new memory item with a unique ID. "
        "Note: Core memory cannot be inserted, only updated."
    )
    parameters = [
        Parameter(
            name="memory_type",
            type="string",
            description="Type of memory to insert: 'semantic' (general knowledge) or 'episodic' (specific experiences). Core memory cannot be inserted.",
            enum=["semantic_memory", "episodic_memory"],
        ),
        Parameter(
            name="content",
            type="string",
            description="Content of the memory to insert. Creates a new memory item with a unique ID.",
        ),
    ]

    @classmethod
    def execute(cls, memory: AlphaMemory, arguments: Dict[str, Any]) -> Dict[str, Any]:
        content = arguments["content"]
        new_memory = memory.new_memory_insert(
            arguments["memory_type"].replace("_memory", ""),
            content,
        )
        if new_memory is None:
            return {
                "status": "skipped",
                "message": "Memory content already exists in the memory pool, insertion skipped.",
            }
        return {"status": "ok", "new_memory": new_memory}


class MemoryUpdate(ToolFunction):
    name = "memory_update"
    description = (
        "Update an existing memory. For core memory, replaces the entire paragraph content. "
        "If core memory is empty, then directly write into the core memory. "
        "For semantic/episodic memories, updates the specific memory item by ID."
    )
    parameters = [
        Parameter(
            name="memory_type",
            type="string",
            description="Type of memory to update: 'core' (simple paragraph), 'semantic' (general knowledge), or 'episodic' (specific experiences)",
            enum=["core_memory", "semantic_memory", "episodic_memory"],
        ),
        Parameter(
            name="new_content",
            type="string",
            description="New **combined** content for the memory. For core memory, this replaces the entire paragraph. "
            "For semantic/episodic, this replaces the content of the specified memory ID.",
        ),
        Parameter(
            name="memory_id",
            type="string",
            description="ID of the memory to update. Required for semantic/episodic memories, ignored for core memory.",
            required=False,
        ),
    ]

    @classmethod
    def execute(cls, memory: AlphaMemory, arguments: Dict[str, Any]) -> Dict[str, Any]:
        new_content = arguments["new_content"]
        updated_memory = memory.memory_update(
            arguments["memory_type"].replace("_memory", ""),
            new_content,
            arguments.get("memory_id"),
        )
        return {"status": "ok", "updated_memory": updated_memory}


class MemoryDelete(ToolFunction):
    name = "memory_delete"
    description = (
        "Delete a memory. For core memory, clears the entire paragraph content. "
        "For semantic/episodic memories, deletes the specific memory item by ID."
    )
    parameters = [
        Parameter(
            name="memory_type",
            type="string",
            description="Type of memory to delete: 'core' (simple paragraph), 'semantic' (general knowledge), or 'episodic' (specific experiences)",
            enum=["core_memory", "semantic_memory", "episodic_memory"],
        ),
        Parameter(
            name="memory_id",
            type="string",
            description="ID of the memory to delete. Required for semantic/episodic memories, ignored for core memory.",
            required=False,
        ),
    ]

    @classmethod
    def execute(cls, memory: AlphaMemory, arguments: Dict[str, Any]) -> Dict[str, Any]:
        memory.memory_delete(
            arguments["memory_type"].replace("_memory", ""),
            arguments.get("memory_id"),
        )
        return {"status": "ok"}


MEMORY_TOOL_FUNCTIONS = [NewMemoryInsert, MemoryUpdate, MemoryDelete]

FUNCTION_IMPLS = {func.name: func.execute for func in MEMORY_TOOL_FUNCTIONS}


def get_memory_tool_schemas(
    memory: AlphaMemory,
    *,
    allow_memory_delete: bool = True,
) -> List[Dict[str, Any]]:
    schemas: List[Dict[str, Any]] = []
    for func in MEMORY_TOOL_FUNCTIONS:
        if not allow_memory_delete and func is MemoryDelete:
            continue
        schema = func.to_schema(memory)
        if schema is not None:
            schemas.append(schema)
    return schemas


def execute_tool(name: str, memory: AlphaMemory, arguments: Dict[str, Any]) -> Dict[str, Any]:
    fn = FUNCTION_IMPLS.get(name)
    if fn is None:
        return {"status": "error", "message": f"unknown tool: {name}"}
    try:
        return fn(memory, arguments)
    except Exception as exc:
        return {"status": "error", "message": str(exc)}
