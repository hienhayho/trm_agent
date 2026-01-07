"""GLiNER2-based entity extraction for tool arguments and slots.

GLiNER2 provides improved entity extraction with support for:
- LoRA adapters: domain-specific fine-tuning while keeping the base model frozen
- Full finetuning: complete model weights saved to disk
"""

import warnings
from pathlib import Path
from typing import Optional, Union

import torch
from gliner2 import GLiNER2
from transformers import AutoTokenizer


def _load_gliner2_with_fixed_tokenizer(model_path: str) -> GLiNER2:
    """Load GLiNER2 model with fixed Mistral tokenizer regex.

    Suppresses tokenizer warnings during loading and then replaces
    the tokenizer with a properly configured one.
    """
    # Suppress tokenizer warnings during initial load
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*incorrect regex pattern.*")
        warnings.filterwarnings("ignore", message=".*fix_mistral_regex.*")
        model = GLiNER2.from_pretrained(model_path)

    # Replace tokenizer with fixed version
    try:
        fixed_tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            fix_mistral_regex=True,
        )
        model.processor.tokenizer = fixed_tokenizer
    except TypeError:
        # fix_mistral_regex not supported in older transformers versions
        pass

    return model


def _is_lora_adapter(path: Path) -> bool:
    """Check if path contains a LoRA adapter (has adapter_config.json)."""
    return (path / "adapter_config.json").exists()


def _is_full_model(path: Path) -> bool:
    """Check if path contains a full model (has model weights but no adapter_config)."""
    has_safetensors = (path / "model.safetensors").exists()
    has_pytorch = (path / "pytorch_model.bin").exists()
    has_config = (path / "config.json").exists()
    return (has_safetensors or has_pytorch) and has_config and not _is_lora_adapter(path)


class GLiNER2Extractor:
    """Extract slots and tool arguments using GLiNER2 NER model.

    This class provides entity extraction capabilities using a pre-trained
    or fine-tuned GLiNER2 model. It supports:
    - LoRA adapters for efficient domain-specific extraction
    - Full finetuned models for maximum performance

    Supports loading models from:
    - HuggingFace Hub (e.g., "fastino/gliner2-base-v1")
    - Local path (full finetuned model)

    Supports loading LoRA adapters from:
    - Local path (e.g., "outputs/gliner2/adapter/final")

    Example:
        >>> # Using pre-trained model
        >>> extractor = GLiNER2Extractor()
        >>> slots, args = extractor.extract_all(
        ...     text="Tôi là Nguyễn Văn A, ở 123 Nguyễn Huệ",
        ...     tool_name="get_product_price",
        ...     tool_params={"get_product_price": ["product", "address"]}
        ... )

        >>> # Using fine-tuned LoRA adapter
        >>> extractor = GLiNER2Extractor(
        ...     adapter_path="outputs/gliner2/adapter/final"
        ... )

        >>> # Using full finetuned model
        >>> extractor = GLiNER2Extractor(
        ...     adapter_path="outputs/gliner2_full/best"  # Auto-detects full model
        ... )
    """

    def __init__(
        self,
        model_name: Union[str, Path] = "fastino/gliner2-multi-v1",
        adapter_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        threshold: float = 0.5,
        slot_fields: Optional[list[str]] = None,
    ):
        """Initialize GLiNER2 extractor.

        Args:
            model_name: HuggingFace model name or local path to base model.
                Default is GLiNER2 base model.
            adapter_path: Path to LoRA adapter directory OR full finetuned model.
                Auto-detects whether it's an adapter or full model.
                If adapter: loads on top of base model.
                If full model: loads directly (ignores model_name).
            device: Device to run model on ("cuda" or "cpu").
                Auto-detected if None.
            threshold: Confidence threshold for entity extraction.
            slot_fields: List of slot field names to always extract.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = str(model_name)
        self.adapter_path = str(adapter_path) if adapter_path else None
        self.threshold = threshold
        self.slot_fields = slot_fields or [
            "address",
            "phone",
            "device_number",
            "intent_of_user",
            "name",
            "contract_id",
        ]
        self._is_full_model = False

        # Check if adapter_path is actually a full finetuned model
        if self.adapter_path:
            adapter_path_obj = Path(self.adapter_path)
            if _is_full_model(adapter_path_obj):
                # Load full finetuned model directly
                print(f"Loading GLiNER2 full finetuned model: {self.adapter_path}")
                self.model = _load_gliner2_with_fixed_tokenizer(self.adapter_path)
                self.model = self.model.to(self.device)
                self._is_full_model = True
            elif _is_lora_adapter(adapter_path_obj):
                # Load base model + LoRA adapter
                print(f"Loading GLiNER2 base model: {self.model_name}")
                self.model = _load_gliner2_with_fixed_tokenizer(self.model_name)
                self.model = self.model.to(self.device)
                self.load_adapter(self.adapter_path)
            else:
                raise FileNotFoundError(
                    f"Path '{self.adapter_path}' is neither a LoRA adapter "
                    f"(missing adapter_config.json) nor a full model "
                    f"(missing config.json + model weights). "
                    f"Please check the path."
                )
        else:
            # Load base model only
            print(f"Loading GLiNER2 base model: {self.model_name}")
            self.model = _load_gliner2_with_fixed_tokenizer(self.model_name)
            self.model = self.model.to(self.device)

    def load_adapter(self, adapter_path: Union[str, Path]) -> None:
        """Load a LoRA adapter.

        Args:
            adapter_path: Path to adapter directory.
        """
        adapter_path = str(adapter_path)
        print(f"Loading GLiNER2 LoRA adapter: {adapter_path}")
        self.model.load_adapter(adapter_path)
        self.adapter_path = adapter_path

    def unload_adapter(self) -> None:
        """Unload the current LoRA adapter.

        Note: This only works for LoRA adapters, not full finetuned models.
        """
        if self._is_full_model:
            print("Cannot unload adapter: model is a full finetuned model, not an adapter")
            return

        if self.has_adapter:
            print("Unloading GLiNER2 adapter")
            self.model.unload_adapter()
            self.adapter_path = None

    @property
    def has_adapter(self) -> bool:
        """Check if an adapter is currently loaded (LoRA only, not full finetune)."""
        if self._is_full_model:
            return False
        return self.model.has_adapter

    @property
    def is_finetuned(self) -> bool:
        """Check if model has been finetuned (either LoRA or full)."""
        return self._is_full_model or self.has_adapter

    def extract(
        self,
        text: str,
        labels: list[str],
        threshold: Optional[float] = None,
    ) -> dict[str, list[dict]]:
        """Extract entities for given labels.

        Args:
            text: Input text to extract from.
            labels: Entity labels to extract (e.g., ["address", "phone", "product"]).
            threshold: Confidence threshold (default: self.threshold).

        Returns:
            Dict mapping label -> list of extracted entities.
            Each entity: {"text": str, "start": int, "end": int, "score": float}
        """
        if not text or not labels:
            return {label: [] for label in labels}

        threshold = threshold or self.threshold

        # GLiNER2 extract_entities returns various formats depending on version
        result = self.model.extract_entities(text, labels, threshold=threshold)

        # Handle different return formats
        # GLiNER2 returns: {"entities": {"label1": ["text1", "text2"], "label2": ["text3"]}}
        if isinstance(result, dict):
            entities_data = result.get("entities", result)

            # Format: {"label": ["text1", "text2"], ...} - dict of label -> list of strings
            if isinstance(entities_data, dict) and entities_data:
                first_value = next(iter(entities_data.values()), None)
                if isinstance(first_value, list):
                    # This is the GLiNER2 format: {"label": ["text1", "text2"]}
                    grouped: dict[str, list[dict]] = {label: [] for label in labels}
                    for label, texts in entities_data.items():
                        if label in grouped and isinstance(texts, list):
                            for text in texts:
                                if isinstance(text, str) and text:
                                    grouped[label].append({
                                        "text": text,
                                        "start": 0,
                                        "end": len(text),
                                        "score": 1.0,  # No score info in this format
                                    })
                    return grouped

            # Format: list of entity dicts [{"label": ..., "text": ..., "start": ..., "end": ..., "score": ...}]
            if isinstance(entities_data, list):
                grouped = {label: [] for label in labels}
                for ent in entities_data:
                    if isinstance(ent, dict):
                        label = ent.get("label", "")
                        if label in grouped:
                            grouped[label].append({
                                "text": ent.get("text", ""),
                                "start": ent.get("start", 0),
                                "end": ent.get("end", 0),
                                "score": ent.get("score", 1.0),
                            })
                return grouped

        elif isinstance(result, list):
            # Format: direct list of entities
            grouped = {label: [] for label in labels}
            for ent in result:
                if isinstance(ent, dict):
                    label = ent.get("label", "")
                    if label in grouped:
                        grouped[label].append({
                            "text": ent.get("text", ""),
                            "start": ent.get("start", 0),
                            "end": ent.get("end", 0),
                            "score": ent.get("score", 1.0),
                        })
            return grouped

        # Fallback: return empty groups
        return {label: [] for label in labels}

    def extract_slots(self, text: str) -> dict[str, str]:
        """Extract slot fields from text.

        Args:
            text: Input text to extract from.

        Returns:
            Dict of slot_name -> extracted value (best match per slot).
        """
        entities = self.extract(text, self.slot_fields)

        result = {}
        for slot, matches in entities.items():
            if matches:
                best = max(matches, key=lambda x: x["score"])
                result[slot] = best["text"]

        return result

    def extract_all(
        self,
        text: str,
        tool_name: Optional[str],
        tool_params: dict[str, list[str]],
    ) -> tuple[dict[str, str], dict[str, str]]:
        """Extract both slots and tool arguments.

        This method combines slot extraction and tool argument extraction
        into a single GLiNER2 call for efficiency.

        Args:
            text: Input text to extract from.
            tool_name: Name of the tool (None if direct_answer).
            tool_params: Mapping of tool names to their parameter lists.

        Returns:
            Tuple of (slots_dict, tool_args_dict).
            - slots_dict: Extracted slot values.
            - tool_args_dict: Extracted tool argument values.
        """
        # Combine all labels to extract
        labels = list(self.slot_fields)
        tool_arg_labels: list[str] = []

        if tool_name:
            tool_arg_labels = tool_params.get(tool_name, [])
            # Add tool args that aren't already slots
            for arg in tool_arg_labels:
                if arg not in labels:
                    labels.append(arg)

        # Single extraction call
        entities = self.extract(text, labels)

        # Split results into slots and tool args
        slots: dict[str, str] = {}
        tool_args: dict[str, str] = {}

        for label, matches in entities.items():
            if matches:
                best = max(matches, key=lambda x: x["score"])
                if label in self.slot_fields:
                    slots[label] = best["text"]
                elif tool_name and label in tool_arg_labels:
                    tool_args[label] = best["text"]

        return slots, tool_args
