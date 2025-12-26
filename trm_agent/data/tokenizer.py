"""TRM Tokenizer with SentencePiece and special tokens.

Special tokens for tool-calling:
- <pad>: Padding token
- <unk>: Unknown token
- <bos>: Beginning of sequence
- <eos>: End of sequence
- <user>: User turn marker
- <assistant>: Assistant turn marker
- <system>: System message marker
- <tool_call>: Start of tool call
- </tool_call>: End of tool call
- <tool_response>: Start of tool response
- </tool_response>: End of tool response
- <tool_name>: Tool name marker
- <tool_args>: Tool arguments marker
- <slot>: Slot value marker
"""

from pathlib import Path
from typing import Optional

import sentencepiece as spm


# Special tokens definition
SPECIAL_TOKENS = {
    "pad_token": "<pad>",
    "unk_token": "<unk>",
    "bos_token": "<bos>",
    "eos_token": "<eos>",
    "user_token": "<user>",
    "assistant_token": "<assistant>",
    "system_token": "<system>",
    "tool_call_start": "<tool_call>",
    "tool_call_end": "</tool_call>",
    "tool_response_start": "<tool_response>",
    "tool_response_end": "</tool_response>",
    "tool_name_token": "<tool_name>",
    "tool_args_token": "<tool_args>",
    "slot_token": "<slot>",
}

# Role token mapping
ROLE_TOKENS = {
    "user": "<user>",
    "assistant": "<assistant>",
    "system": "<system>",
    "tool_call": "<tool_call>",
    "tool_response": "<tool_response>",
}

# Role ID mapping
ROLE_IDS = {
    "user": 0,
    "assistant": 1,
    "tool_call": 2,
    "tool_response": 3,
    "system": 1,  # Treat system as assistant for now
}


class TRMTokenizer:
    """Tokenizer for TRM using SentencePiece.

    Handles tokenization of conversation history with special tokens
    for roles, tool calls, and tool responses.
    """

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        vocab_size: int = 32000,
    ):
        """Initialize tokenizer.

        Args:
            model_path: Path to SentencePiece model file (.model)
            vocab_size: Vocabulary size (used when training new tokenizer)
        """
        self.vocab_size = vocab_size
        self.sp_model = None
        self.model_path = model_path

        # Special token IDs (will be set after loading model)
        self._special_token_ids = {}

        if model_path is not None:
            self.load(model_path)

    def load(self, model_path: str | Path):
        """Load SentencePiece model."""
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"SentencePiece model not found: {model_path}")

        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(str(model_path))
        self.model_path = model_path

        # Cache special token IDs
        self._cache_special_token_ids()

    def _cache_special_token_ids(self):
        """Cache special token IDs for fast lookup."""
        for name, token in SPECIAL_TOKENS.items():
            token_id = self.sp_model.PieceToId(token)
            self._special_token_ids[name] = token_id

    @property
    def pad_token_id(self) -> int:
        """Get padding token ID."""
        return self._special_token_ids.get("pad_token", 0)

    @property
    def unk_token_id(self) -> int:
        """Get unknown token ID."""
        return self._special_token_ids.get("unk_token", 1)

    @property
    def bos_token_id(self) -> int:
        """Get beginning of sequence token ID."""
        return self._special_token_ids.get("bos_token", 2)

    @property
    def eos_token_id(self) -> int:
        """Get end of sequence token ID."""
        return self._special_token_ids.get("eos_token", 3)

    def get_role_token_id(self, role: str) -> int:
        """Get token ID for a role."""
        token = ROLE_TOKENS.get(role, "<unk>")
        return self.sp_model.PieceToId(token)

    def get_role_id(self, role: str) -> int:
        """Get role ID for role embedding."""
        return ROLE_IDS.get(role, 0)

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        """Encode text to token IDs.

        Args:
            text: Text to encode
            add_bos: Whether to add BOS token
            add_eos: Whether to add EOS token

        Returns:
            List of token IDs
        """
        if self.sp_model is None:
            raise RuntimeError("Tokenizer not loaded. Call load() first.")

        ids = self.sp_model.EncodeAsIds(text)

        if add_bos:
            ids = [self.bos_token_id] + ids
        if add_eos:
            ids = ids + [self.eos_token_id]

        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text.

        Args:
            ids: Token IDs to decode
            skip_special_tokens: Whether to skip special tokens

        Returns:
            Decoded text
        """
        if self.sp_model is None:
            raise RuntimeError("Tokenizer not loaded. Call load() first.")

        if skip_special_tokens:
            # Filter out special token IDs
            special_ids = set(self._special_token_ids.values())
            ids = [i for i in ids if i not in special_ids]

        return self.sp_model.DecodeIds(ids)

    def encode_conversation(
        self,
        history: list[dict],
        max_length: Optional[int] = None,
    ) -> dict[str, list[int]]:
        """Encode conversation history with role tokens.

        Args:
            history: List of conversation turns
                Each turn: {"role": str, "content": str | dict}
            max_length: Maximum sequence length

        Returns:
            Dictionary with:
            - input_ids: Token IDs
            - role_ids: Role IDs for each position
            - attention_mask: Attention mask
        """
        input_ids = [self.bos_token_id]
        role_ids = [0]  # BOS gets role 0

        for turn in history:
            role = turn.get("role", "user")
            content = turn.get("content", "")

            # Handle content that might be a dict (for tool_call, tool_response)
            if isinstance(content, dict):
                content = str(content)

            # Add role token
            role_token_id = self.get_role_token_id(role)
            input_ids.append(role_token_id)
            role_ids.append(self.get_role_id(role))

            # Encode content
            content_ids = self.encode(content)
            input_ids.extend(content_ids)
            role_ids.extend([self.get_role_id(role)] * len(content_ids))

        # Add EOS
        input_ids.append(self.eos_token_id)
        role_ids.append(0)

        # Truncate if needed
        if max_length is not None and len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            role_ids = role_ids[:max_length]

        # Create attention mask
        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": input_ids,
            "role_ids": role_ids,
            "attention_mask": attention_mask,
        }

    def encode_tool_call(
        self,
        tool_name: str,
        tool_args: dict,
    ) -> list[int]:
        """Encode a tool call.

        Args:
            tool_name: Name of the tool
            tool_args: Tool arguments

        Returns:
            Token IDs for the tool call
        """
        ids = []

        # Tool call start
        ids.append(self._special_token_ids["tool_call_start"])

        # Tool name
        ids.append(self._special_token_ids["tool_name_token"])
        ids.extend(self.encode(tool_name))

        # Tool arguments
        ids.append(self._special_token_ids["tool_args_token"])
        ids.extend(self.encode(str(tool_args)))

        # Tool call end
        ids.append(self._special_token_ids["tool_call_end"])

        return ids

    @classmethod
    def train(
        cls,
        input_files: list[str | Path],
        output_path: str | Path,
        vocab_size: int = 32000,
        character_coverage: float = 0.9995,
        model_type: str = "unigram",
    ) -> "TRMTokenizer":
        """Train a new SentencePiece model.

        Args:
            input_files: List of input text files for training
            output_path: Path to save the trained model (without extension)
            vocab_size: Target vocabulary size
            character_coverage: Character coverage for training
            model_type: Model type (unigram, bpe, word, char)

        Returns:
            Trained TRMTokenizer
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare user-defined symbols (exclude built-in tokens: pad, unk, bos, eos)
        builtin_tokens = {"pad_token", "unk_token", "bos_token", "eos_token"}
        user_defined_symbols = [
            token for name, token in SPECIAL_TOKENS.items()
            if name not in builtin_tokens
        ]

        # Train SentencePiece
        spm.SentencePieceTrainer.Train(
            input=",".join(str(f) for f in input_files),
            model_prefix=str(output_path),
            vocab_size=vocab_size,
            character_coverage=character_coverage,
            model_type=model_type,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            pad_piece="<pad>",
            unk_piece="<unk>",
            bos_piece="<bos>",
            eos_piece="<eos>",
            user_defined_symbols=user_defined_symbols,
        )

        # Load and return the trained tokenizer
        tokenizer = cls(vocab_size=vocab_size)
        tokenizer.load(f"{output_path}.model")

        return tokenizer

    def __len__(self) -> int:
        """Get vocabulary size."""
        if self.sp_model is None:
            return self.vocab_size
        return self.sp_model.GetPieceSize()
