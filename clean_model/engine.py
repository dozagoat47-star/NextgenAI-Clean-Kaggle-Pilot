"""Checkpoint loading and greedy response generation for the pilot model."""

from pathlib import Path

import torch

from clean_model.train import BOS_ID, EOS_ID, SPECIAL_TOKEN_COUNT, PilotCausalLM


def _prompt_text(
    messages: list[dict[str, str]],
    evidence: list[dict[str, str]],
) -> str:
    pieces = []
    for message in messages:
        if message.get('role') not in ('user', 'assistant'):
            raise ValueError('messages may only contain user/assistant roles')
        content = message.get('content')
        if not isinstance(content, str) or not content.strip():
            raise ValueError('message content must be a non-empty string')
        pieces.append('<%s>\n%s\n' % (message['role'], content))
    for source in evidence:
        title = source.get('title')
        text = source.get('text')
        if not isinstance(title, str) or not isinstance(text, str):
            raise ValueError('evidence requires title and text')
        pieces.append('<evidence:%s>\n%s\n' % (title, text))
    pieces.append('<assistant>\n')
    return ''.join(pieces)


class PilotGenerator:
    """Loads a trained pilot checkpoint and composes a response token by token."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location='cpu',
            weights_only=True,
        )
        if checkpoint.get('schema_version') != 1:
            raise ValueError('unsupported pilot checkpoint schema')
        vocabulary = checkpoint.get('vocabulary')
        if not isinstance(vocabulary, list) or not all(
                isinstance(character, str) and len(character) == 1
                for character in vocabulary):
            raise ValueError('checkpoint has an invalid character vocabulary')
        self.vocabulary = vocabulary
        self.char_to_id = {
            character: index + SPECIAL_TOKEN_COUNT
            for index, character in enumerate(vocabulary)
        }
        self.device = torch.device(
            device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model: PilotCausalLM = PilotCausalLM(
            **checkpoint['model_config'])
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.to(self.device)
        self.model.eval()

    def generate(
        self,
        messages: list[dict[str, str]],
        evidence: list[dict[str, str]] | None = None,
        max_new_tokens: int = 240,
    ) -> str:
        return generate_response(
            self.model,
            self.vocabulary,
            messages,
            evidence=evidence,
            device=self.device,
            max_new_tokens=max_new_tokens,
        )

    def __call__(self, message: str) -> str:
        return self.generate([{'role': 'user', 'content': message}])


def generation_prefix_ids(
    model: PilotCausalLM,
    vocabulary: list[str],
    messages: list[dict[str, str]],
    evidence: list[dict[str, str]] | None = None,
) -> list[int]:
    char_to_id = {
        character: index + SPECIAL_TOKEN_COUNT
        for index, character in enumerate(vocabulary)
    }
    prompt = _prompt_text(messages, evidence or [])
    prefix = [char_to_id.get(character, 2) for character in prompt]
    return [BOS_ID] + prefix[-(model.max_seq_len - 1):]


def first_token_logits(
    model: PilotCausalLM,
    vocabulary: list[str],
    messages: list[dict[str, str]],
    evidence: list[dict[str, str]] | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return next-token logits for the production prompt before generation."""
    target_device = device or next(model.parameters()).device
    input_ids = torch.tensor(
        [generation_prefix_ids(model, vocabulary, messages, evidence)],
        dtype=torch.long,
        device=target_device,
    )
    with torch.no_grad():
        return model(input_ids)[0, -1]


def generate_response(
    model: PilotCausalLM,
    vocabulary: list[str],
    messages: list[dict[str, str]],
    evidence: list[dict[str, str]] | None = None,
    device: torch.device | None = None,
    max_new_tokens: int = 240,
) -> str:
    """Greedily generate a response using the production inference path."""
    if max_new_tokens < 1:
        raise ValueError('max_new_tokens must be positive')
    target_device = device or next(model.parameters()).device
    prefix_ids = generation_prefix_ids(
        model, vocabulary, messages, evidence)
    generated = []
    with torch.no_grad():
        for step in range(max_new_tokens):
            if step == 0:
                logits = first_token_logits(
                    model, vocabulary, messages, evidence, target_device)
            else:
                input_ids = torch.tensor(
                    [prefix_ids + generated],
                    dtype=torch.long,
                    device=target_device,
                )
                input_ids = input_ids[:, -model.max_seq_len:]
                logits = model(input_ids)[0, -1]
            next_id = int(logits.argmax(dim=-1).item())
            if next_id == EOS_ID:
                break
            if next_id >= SPECIAL_TOKEN_COUNT:
                generated.append(next_id)
    text = ''.join(
        vocabulary[token_id - SPECIAL_TOKEN_COUNT]
        for token_id in generated
        if SPECIAL_TOKEN_COUNT <= token_id
        < SPECIAL_TOKEN_COUNT + len(vocabulary)
    )
    return text.strip()
