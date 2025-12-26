"""Data processing utilities for converting raw conversation data to TRM training format."""

from dataclasses import dataclass, field
from typing import Any


# Default slot fields to extract from raw data
DEFAULT_SLOT_FIELDS = [
    "address",
    "phone",
    "device_number",
    "intent_of_user",
    "name",
    "contract_id",
]


@dataclass
class ProcessedSample:
    """Represents a single processed training sample."""

    tools: list[dict[str, Any]]
    history: list[dict[str, str]]
    decision: str  # "tool_call" or "direct_answer"
    slots: dict[str, str]
    tool: dict[str, Any]
    content: str


@dataclass
class RawDataProcessor:
    """Processor for converting raw conversation data to TRM training format."""

    tools: list[dict[str, Any]] = field(default_factory=list)
    slot_fields: list[str] = field(default_factory=lambda: DEFAULT_SLOT_FIELDS.copy())

    def process_conversation(
        self, raw_data: list[dict[str, Any]]
    ) -> list[ProcessedSample]:
        """
        Process a raw conversation into multiple training samples.

        Each decision point (assistant response or tool_call) becomes a separate sample.
        The history accumulates up to each decision point.

        Args:
            raw_data: List of conversation turns in raw format

        Returns:
            List of ProcessedSample objects
        """
        samples = []
        history = []

        for turn in raw_data:
            role = turn.get("role", "")
            content = turn.get("content", {})

            if role == "user":
                # User messages are added to history as-is
                history.append({"role": "user", "content": content})

            elif role == "assistant":
                # Assistant response = direct_answer decision
                sample = self._create_direct_answer_sample(content, history)
                samples.append(sample)

                # Add assistant response to history
                response_text = self._extract_response_text(content)
                history.append({"role": "assistant", "content": response_text})

            elif role == "tool_call":
                # Tool call = tool_call decision
                sample = self._create_tool_call_sample(content, history)
                samples.append(sample)

                # Add tool_call to history with role: tool_call
                tool_info = self._extract_tool_info(content)
                history.append({
                    "role": "tool_call",
                    "content": tool_info,
                })

            elif role == "tool_response":
                # Tool response is added to history with role: tool_response
                response_data = self._extract_tool_response(content)
                history.append({
                    "role": "tool_response",
                    "content": response_data,
                })

        return samples

    def _create_direct_answer_sample(
        self, content: dict[str, Any], history: list[dict[str, str]]
    ) -> ProcessedSample:
        """Create a sample for direct_answer decision."""
        slots = extract_slots(content, self.slot_fields)
        response_text = self._extract_response_text(content)

        return ProcessedSample(
            tools=self.tools.copy(),
            history=self._copy_history(history),
            decision="direct_answer",
            slots=slots,
            tool={},
            content=response_text,
        )

    def _create_tool_call_sample(
        self, content: dict[str, Any], history: list[dict[str, str]]
    ) -> ProcessedSample:
        """Create a sample for tool_call decision."""
        slots = extract_slots(content, self.slot_fields)
        tool_info = self._extract_tool_info(content)

        return ProcessedSample(
            tools=self.tools.copy(),
            history=self._copy_history(history),
            decision="tool_call",
            slots=slots,
            tool=tool_info,
            content="",
        )

    def _extract_response_text(self, content: dict[str, Any] | str) -> str:
        """Extract response text from content."""
        if isinstance(content, str):
            return content
        return content.get("response", "")

    def _extract_tool_info(self, content: dict[str, Any]) -> dict[str, Any]:
        """Extract tool name and arguments from tool_call content."""
        response = content.get("response", {})
        if isinstance(response, dict):
            return {
                "name": response.get("name", ""),
                "arguments": response.get("arguments", {}),
            }
        return {}

    def _extract_tool_response(self, content: dict[str, Any]) -> dict[str, Any]:
        """Extract tool response data."""
        response = content.get("response", {})
        if isinstance(response, dict):
            return response
        return {"result": response}

    def _copy_history(self, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Create a deep copy of history."""
        import copy
        return copy.deepcopy(history)


def extract_slots(content: dict[str, Any], slot_fields: list[str]) -> dict[str, str]:
    """
    Extract slot values from content.

    Args:
        content: The content dictionary containing slot fields
        slot_fields: List of field names to extract

    Returns:
        Dictionary of slot name to value
    """
    slots = {}
    for field_name in slot_fields:
        value = content.get(field_name, "")
        slots[field_name] = value if value else ""
    return slots


def build_history(
    raw_data: list[dict[str, Any]], up_to_index: int
) -> list[dict[str, Any]]:
    """
    Build conversation history up to a specific index.

    Args:
        raw_data: List of raw conversation turns
        up_to_index: Index to build history up to (exclusive)

    Returns:
        List of history messages
    """
    history = []

    for i in range(up_to_index):
        turn = raw_data[i]
        role = turn.get("role", "")
        content = turn.get("content", {})

        if role == "user":
            history.append({"role": "user", "content": content})

        elif role == "assistant":
            response = content.get("response", "") if isinstance(content, dict) else content
            history.append({"role": "assistant", "content": response})

        elif role == "tool_call":
            response = content.get("response", {})
            tool_info = {
                "name": response.get("name", "") if isinstance(response, dict) else "",
                "arguments": response.get("arguments", {}) if isinstance(response, dict) else {},
            }
            history.append({
                "role": "tool_call",
                "content": tool_info,
            })

        elif role == "tool_response":
            response = content.get("response", {})
            history.append({
                "role": "tool_response",
                "content": response,
            })

    return history


def process_raw_conversation(
    raw_data: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    slot_fields: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Process a raw conversation into training samples.

    This is a convenience function that wraps RawDataProcessor.

    Args:
        raw_data: List of conversation turns in raw format
        tools: List of tool definitions to include in samples
        slot_fields: List of slot field names to extract

    Returns:
        List of processed samples as dictionaries
    """
    processor = RawDataProcessor(
        tools=tools or [],
        slot_fields=slot_fields or DEFAULT_SLOT_FIELDS.copy(),
    )

    samples = processor.process_conversation(raw_data)

    # Convert to dictionaries
    return [
        {
            "tools": sample.tools,
            "history": sample.history,
            "decision": sample.decision,
            "slots": sample.slots,
            "tool": sample.tool,
            "content": sample.content,
        }
        for sample in samples
    ]
