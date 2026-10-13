"""Train a small character-level causal LM on approved pilot records."""

import argparse
import hashlib
import json
import math
import os
import random
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

try:
    import torch
    from torch import nn
except ImportError as exc:
    raise RuntimeError(
        'Training requires PyTorch. Install dependencies from requirements.txt.'
    ) from exc

from clean_model.data import (
    build_supervised_examples,
    dataset_fingerprint,
    load_jsonl,
    split_by_group,
)


PAD_ID = 0
EOS_ID = 1
UNK_ID = 2
BOS_ID = 3
SPECIAL_TOKEN_COUNT = 4
GENERATION_EVAL_TRAIN_EXAMPLES = 4
GENERATION_EVAL_VALIDATION_EXAMPLES = 16


def _example_text(example: dict[str, Any]) -> tuple[str, str]:
    pieces = []
    for message in example['context']:
        pieces.append('<%s>\n%s\n' % (message['role'], message['content']))
    for evidence in example['evidence']:
        pieces.append(
            '<evidence:%s>\n%s\n'
            % (evidence['title'], evidence['text']))
    pieces.append('<assistant>\n')
    return ''.join(pieces), example['target']


def build_vocabulary(
    examples: Iterable[dict[str, Any]],
) -> list[str]:
    """Build a deterministic character vocabulary from training data only."""
    characters = set()
    for example in examples:
        prefix, target = _example_text(example)
        characters.update(prefix)
        characters.update(target)
    return sorted(characters)


def encode_example(
    example: dict[str, Any],
    vocabulary: list[str],
    max_seq_len: int,
) -> tuple[list[int], list[int]]:
    """Encode one context/target pair and mask loss to assistant target tokens."""
    if max_seq_len < 4:
        raise ValueError('max_seq_len must be at least 4')
    char_to_id = {character: index + SPECIAL_TOKEN_COUNT
                  for index, character in enumerate(vocabulary)}
    prefix, target = _example_text(example)
    target_ids = [char_to_id.get(character, UNK_ID) for character in target]
    available_prefix = max_seq_len - len(target_ids) - 2
    if available_prefix < 1:
        target_ids = target_ids[:max_seq_len - 3]
        available_prefix = 1
    prefix_ids = [
        char_to_id.get(character, UNK_ID)
        for character in prefix[-available_prefix:]
    ]
    tokens = [BOS_ID] + prefix_ids + target_ids + [EOS_ID]
    if len(tokens) > max_seq_len:
        raise ValueError('encoded sequence exceeds max_seq_len')

    inputs = tokens[:-1]
    labels = [-100] * len(inputs)
    target_start = 1 + len(prefix_ids)
    first_target_label = target_start - 1
    labels[first_target_label:] = tokens[target_start:]
    if not any(label != -100 for label in labels):
        raise ValueError('example has no assistant target tokens')
    return inputs, labels


if torch is not None:
    class PilotCausalLM(nn.Module):
        """Small causal Transformer; response tokens are scored by the trainer."""

        def __init__(
            self,
            vocab_size: int,
            d_model: int = 64,
            num_heads: int = 4,
            num_layers: int = 2,
            max_seq_len: int = 1024,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            if d_model % num_heads:
                raise ValueError('d_model must be divisible by num_heads')
            self.max_seq_len = max_seq_len
            self.token_embedding = nn.Embedding(vocab_size, d_model)
            self.position_embedding = nn.Embedding(max_seq_len, d_model)
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
                activation='gelu',
            )
            self.transformer = nn.TransformerEncoder(
                layer, num_layers=num_layers, enable_nested_tensor=False)
            self.final_norm = nn.LayerNorm(d_model)
            self.output = nn.Linear(d_model, vocab_size, bias=False)
            self.output.weight = self.token_embedding.weight

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            batch_size, sequence_length = input_ids.shape
            if sequence_length > self.max_seq_len:
                raise ValueError('input exceeds model max_seq_len')
            positions = torch.arange(
                sequence_length, device=input_ids.device).unsqueeze(0)
            hidden = (self.token_embedding(input_ids)
                      + self.position_embedding(positions))
            causal_mask = torch.triu(
                torch.ones(
                    sequence_length, sequence_length,
                    dtype=torch.bool, device=input_ids.device),
                diagonal=1)
            padding_mask = input_ids.eq(PAD_ID)
            hidden = self.transformer(
                hidden, mask=causal_mask, src_key_padding_mask=padding_mask)
            return self.output(self.final_norm(hidden))


def _batches(
    encoded: list[tuple[list[int], list[int]]],
    batch_size: int,
    shuffle: bool,
    rng: random.Random,
):
    order = list(range(len(encoded)))
    if shuffle:
        rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        rows = [encoded[index] for index in order[start:start + batch_size]]
        width = max(len(inputs) for inputs, _ in rows)
        input_batch = torch.full(
            (len(rows), width), PAD_ID, dtype=torch.long)
        label_batch = torch.full(
            (len(rows), width), -100, dtype=torch.long)
        for index, (inputs, labels) in enumerate(rows):
            input_batch[index, :len(inputs)] = torch.tensor(inputs)
            label_batch[index, :len(labels)] = torch.tensor(labels)
        yield input_batch, label_batch


def _response_loss_sum_and_count(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Return summed assistant-target loss and its scored-token count."""
    token_count = int(labels.ne(-100).sum().item())
    if token_count == 0:
        return logits.new_zeros(()), 0
    loss_sum = nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=-100,
        reduction='sum',
    )
    return loss_sum, token_count


def _weighted_response_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    first_token_weight: float,
) -> torch.Tensor:
    """Average response-token loss with optional extra weight on each first token."""
    if first_token_weight == 1.0:
        loss_sum, token_count = _response_loss_sum_and_count(logits, labels)
        if token_count == 0:
            return logits.new_zeros(())
        return loss_sum / token_count

    scored = labels.ne(-100)
    if not bool(scored.any()):
        return logits.new_zeros(())
    first_target = scored & scored.cumsum(dim=1).eq(1)
    token_losses = nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=-100,
        reduction='none',
    ).reshape_as(labels)
    weights = torch.ones_like(token_losses)
    weights = weights.masked_fill(first_target, first_token_weight)
    weights = weights * scored
    return (token_losses * weights).sum() / weights.sum()


def _response_accuracy_counts(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, int]:
    """Count argmax hits separately for target characters and EOS."""
    scored = labels.ne(-100)
    eos = scored & labels.eq(EOS_ID)
    unknown = scored & labels.eq(UNK_ID)
    characters = scored & labels.ge(SPECIAL_TOKEN_COUNT)
    predictions = logits.argmax(dim=-1)
    return {
        'correct_target_characters': int(
            predictions.eq(labels).logical_and(characters).sum().item()),
        'target_characters': int(characters.sum().item()),
        'correct_unknown_target_tokens': int(
            predictions.eq(labels).logical_and(unknown).sum().item()),
        'unknown_target_tokens': int(unknown.sum().item()),
        'correct_eos': int(
            predictions.eq(labels).logical_and(eos).sum().item()),
        'eos_tokens': int(eos.sum().item()),
    }


def _evaluate(
    model: PilotCausalLM,
    examples: list[tuple[list[int], list[int]]],
    batch_size: int,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for input_ids, labels in _batches(
                examples, batch_size, False, random.Random(0)):
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            logits = model(input_ids)
            loss_sum, token_count = _response_loss_sum_and_count(
                logits, labels)
            if token_count == 0:
                continue
            total_loss += float(loss_sum.item())
            total_tokens += token_count
    if total_tokens == 0:
        raise ValueError('evaluation split has no assistant target tokens')
    return total_loss / total_tokens


def _evaluate_token_accuracy(
    model: PilotCausalLM,
    examples: list[tuple[list[int], list[int]]],
    batch_size: int,
    device: torch.device,
) -> dict[str, float | int]:
    """Measure teacher-forced argmax accuracy on characters and EOS."""
    model.eval()
    totals = {
        'correct_target_characters': 0,
        'target_characters': 0,
        'correct_unknown_target_tokens': 0,
        'unknown_target_tokens': 0,
        'correct_eos': 0,
        'eos_tokens': 0,
        'correct_first_target_tokens': 0,
        'first_target_tokens': 0,
        'correct_known_first_target_characters': 0,
        'known_first_target_characters': 0,
        'unknown_first_target_tokens': 0,
    }
    with torch.no_grad():
        for input_ids, labels in _batches(
                examples, batch_size, False, random.Random(0)):
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            logits = model(input_ids)
            counts = _response_accuracy_counts(logits, labels)
            for key in (
                    'correct_target_characters',
                    'target_characters',
                    'correct_unknown_target_tokens',
                    'unknown_target_tokens',
                    'correct_eos',
                    'eos_tokens'):
                totals[key] += counts[key]
            scored = labels.ne(-100)
            first_targets = scored & scored.cumsum(dim=1).eq(1)
            known_first_targets = first_targets & labels.ge(
                SPECIAL_TOKEN_COUNT)
            unknown_first_targets = first_targets & labels.eq(UNK_ID)
            first_predictions = logits.argmax(dim=-1)
            totals['correct_first_target_tokens'] += int(
                (first_predictions.eq(labels) & first_targets)
                .sum().item())
            totals['first_target_tokens'] += int(first_targets.sum().item())
            totals['correct_known_first_target_characters'] += int(
                (first_predictions.eq(labels) & known_first_targets)
                .sum().item())
            totals['known_first_target_characters'] += int(
                known_first_targets.sum().item())
            totals['unknown_first_target_tokens'] += int(
                unknown_first_targets.sum().item())
    if totals['target_characters'] == 0:
        raise ValueError('evaluation split has no assistant target characters')
    totals['target_character_accuracy'] = (
        totals['correct_target_characters'] / totals['target_characters'])
    totals['eos_accuracy'] = (
        totals['correct_eos'] / totals['eos_tokens']
        if totals['eos_tokens'] else 0.0)
    totals['first_target_token_accuracy'] = (
        totals['correct_first_target_tokens'] / totals['first_target_tokens']
        if totals['first_target_tokens'] else 0.0)
    totals['known_first_target_character_accuracy'] = (
        totals['correct_known_first_target_characters']
        / totals['known_first_target_characters']
        if totals['known_first_target_characters'] else 0.0)
    return totals


def _evaluate_generation(
    model: PilotCausalLM,
    vocabulary: list[str],
    examples: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, int | float]:
    """Measure free-generation exact matches without affecting model selection."""
    from clean_model.engine import generate_response

    model.eval()
    exact_matches = 0
    prefix_characters_total = 0
    target_characters_total = 0
    details = []
    with torch.no_grad():
        for example in examples:
            target = example['target'].strip()
            generated = generate_response(
                model,
                vocabulary,
                example['context'],
                evidence=example['evidence'],
                device=device,
                max_new_tokens=len(target) + 1,
            )
            common_prefix_length = 0
            for expected, actual in zip(target, generated):
                if expected != actual:
                    break
                common_prefix_length += 1
            exact_match = generated == target
            if exact_match:
                first_mismatch_position = None
                mismatch_kind = 'none'
            elif common_prefix_length < min(len(target), len(generated)):
                first_mismatch_position = common_prefix_length + 1
                mismatch_kind = 'character'
            elif len(generated) < len(target):
                first_mismatch_position = len(generated) + 1
                mismatch_kind = 'early_termination'
            else:
                first_mismatch_position = len(target) + 1
                mismatch_kind = 'overgeneration'
            exact_matches += int(exact_match)
            prefix_characters_total += common_prefix_length
            target_characters_total += len(target)
            details.append({
                'record_id': example['record_id'],
                'exact_match': exact_match,
                'target_characters': len(target),
                'generated_characters': len(generated),
                'matching_prefix_characters': common_prefix_length,
                'first_mismatch_position': first_mismatch_position,
                'mismatch_kind': mismatch_kind,
            })
    return {
        'exact_matches': exact_matches,
        'example_count': len(examples),
        'exact_match_rate': (
            exact_matches / len(examples) if examples else 0.0),
        'matching_prefix_character_rate': (
            prefix_characters_total / target_characters_total
            if target_characters_total else 0.0),
        'examples': details,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def train(
    data_path: str | Path,
    output_dir: str | Path,
    seed: int = 7,
    epochs: int = 80,
    patience: int = 12,
    batch_size: int = 4,
    learning_rate: float = 3e-4,
    d_model: int = 64,
    num_heads: int = 4,
    num_layers: int = 2,
    max_seq_len: int = 1024,
    include_test: bool = False,
    first_token_loss_weight: float = 1.0,
    allow_draft_data: bool = False,
) -> dict[str, Any]:
    """Train in a new directory; draft data requires explicit opt-in."""
    if torch is None:
        raise RuntimeError('Training requires PyTorch.')
    if epochs < 1 or patience < 1 or batch_size < 1:
        raise ValueError('epochs, patience, and batch_size must be positive')
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('learning_rate must be finite and positive')
    if (not math.isfinite(first_token_loss_weight)
            or first_token_loss_weight < 1.0):
        raise ValueError('first_token_loss_weight must be finite and at least 1')

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=False)

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    source_path = Path(data_path)
    require_approved = not allow_draft_data
    records = load_jsonl(
        source_path, require_approved=require_approved)
    splits = split_by_group(
        records, seed=seed, require_approved=require_approved)
    active_split_names = ('train', 'validation', 'test') if include_test else (
        'train', 'validation')
    supervised = {
        name: list(build_supervised_examples(
            splits[name], require_approved=require_approved))
        for name in active_split_names
    }
    vocabulary = build_vocabulary(supervised['train'])
    if not vocabulary:
        raise ValueError('training split produced an empty character vocabulary')
    encoded = {
        name: [
            encode_example(example, vocabulary, max_seq_len)
            for example in examples
        ]
        for name, examples in supervised.items()
    }
    validation_generation_examples = supervised['validation']
    if len(validation_generation_examples) > GENERATION_EVAL_VALIDATION_EXAMPLES:
        validation_generation_examples = random.Random(seed).sample(
            validation_generation_examples,
            GENERATION_EVAL_VALIDATION_EXAMPLES,
        )
    generation_examples = {
        'train': supervised['train'][:GENERATION_EVAL_TRAIN_EXAMPLES],
        'validation': validation_generation_examples,
    }
    model_config = {
        'vocab_size': SPECIAL_TOKEN_COUNT + len(vocabulary),
        'd_model': d_model,
        'num_heads': num_heads,
        'num_layers': num_layers,
        'max_seq_len': max_seq_len,
        'dropout': 0.1,
    }
    model = PilotCausalLM(**model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.01)
    rng = random.Random(seed)

    best_validation_loss = math.inf
    best_epoch = None
    best_state = None
    epochs_without_improvement = 0
    optimizer_steps_total = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_tokens = 0
        optimizer_steps_this_epoch = 0
        for input_ids, labels in _batches(
                encoded['train'], batch_size, True, rng):
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids)
            loss_sum, token_count = _response_loss_sum_and_count(
                logits, labels)
            if token_count == 0:
                continue
            loss = _weighted_response_loss(
                logits, labels, first_token_loss_weight)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    'Non-finite training loss at epoch %d' % epoch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer_steps_total += 1
            optimizer_steps_this_epoch += 1
            total_loss += float(loss_sum.item())
            total_tokens += token_count

        if total_tokens == 0:
            raise ValueError('training split has no assistant target tokens')
        if optimizer_steps_this_epoch == 0:
            raise ValueError('training split produced no optimizer steps')
        train_loss = total_loss / total_tokens
        validation_loss = _evaluate(
            model, encoded['validation'], batch_size, device)
        train_token_accuracy = _evaluate_token_accuracy(
            model, encoded['train'], batch_size, device)
        validation_token_accuracy = _evaluate_token_accuracy(
            model, encoded['validation'], batch_size, device)
        train_generation = _evaluate_generation(
            model, vocabulary, generation_examples['train'], device)
        validation_generation = _evaluate_generation(
            model, vocabulary, generation_examples['validation'], device)
        history.append({
            'epoch': epoch,
            'train_response_loss': train_loss,
            'validation_response_loss': validation_loss,
            'optimizer_steps': optimizer_steps_this_epoch,
            'train_token_accuracy': train_token_accuracy,
            'validation_token_accuracy': validation_token_accuracy,
            'train_generation': train_generation,
            'validation_generation': validation_generation,
        })
        print(
            'epoch %d/%d train_response_loss=%.4f '
            'validation_response_loss=%.4f '
            'train_char_accuracy=%.4f '
            'validation_char_accuracy=%.4f '
            'train_generation_exact=%d/%d '
            'validation_generation_exact=%d/%d'
            % (
                epoch,
                epochs,
                train_loss,
                validation_loss,
                train_token_accuracy['target_character_accuracy'],
                validation_token_accuracy['target_character_accuracy'],
                train_generation['exact_matches'],
                train_generation['example_count'],
                validation_generation['exact_matches'],
                validation_generation['example_count'],
            ),
            flush=True,
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is None or best_epoch is None:
        raise RuntimeError('Training produced no usable best checkpoint')
    model.load_state_dict(best_state)
    model.to(device)
    test_loss = (
        _evaluate(model, encoded['test'], batch_size, device)
        if include_test else None)

    checkpoint_path = output_path / 'pilot_model.pt'
    checkpoint = {
        'schema_version': 1,
        'model_config': model_config,
        'vocabulary': vocabulary,
        'model_state': best_state,
    }
    with tempfile.NamedTemporaryFile(
            mode='wb', dir=output_path, delete=False,
            prefix='pilot_model.', suffix='.tmp') as temporary:
        temporary_path = Path(temporary.name)
        torch.save(checkpoint, temporary)
    temporary_path.replace(checkpoint_path)

    metadata = {
        'schema_version': 1,
        'experiment': 'clean_model_character_causal_lm_pilot',
        'interpretation': (
            'Small character-level training experiment; not evidence of '
            'useful Turkish generalization or production quality.'),
        'data_path': str(source_path.resolve()),
        'data_sha256': _sha256(source_path),
        'dataset_fingerprint': dataset_fingerprint(
            records, require_approved=require_approved),
        'draft_records_included': allow_draft_data,
        'review_status_counts': {
            status: sum(
                record['review']['status'] == status for record in records)
            for status in ('approved', 'draft')
        },
        'split_seed': seed,
        'split_counts': {name: len(rows) for name, rows in splits.items()},
        'split_kind_counts': {
            name: {
                kind: sum(row['kind'] == kind for row in rows)
                for kind in ('dialogue', 'grounded_qa')
            }
            for name, rows in splits.items()
        },
        'split_group_ids': {
            name: sorted({row['group_id'] for row in rows})
            for name, rows in splits.items()
        },
        'target_objective': 'assistant_response_characters_and_eos_only',
        'first_token_loss_weight': first_token_loss_weight,
        'tokenizer': 'training-split-only Unicode character vocabulary',
        'torch_version': torch.__version__,
        'device': str(device),
        'model_config': model_config,
        'training': {
            'seed': seed,
            'epochs_requested': epochs,
            'epochs_completed': len(history),
            'optimizer_steps_total': optimizer_steps_total,
            'early_stopping_patience': patience,
            'early_stopping_monitor': 'validation_response_loss',
            'checkpoint_selection_monitor': 'validation_response_loss',
            'generation_eval_train_record_ids': [
                example['record_id']
                for example in generation_examples['train']
            ],
            'generation_eval_validation_record_ids': [
                example['record_id']
                for example in generation_examples['validation']
            ],
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'weight_decay': 0.01,
            'first_token_loss_weight': first_token_loss_weight,
        },
        'best_epoch': best_epoch,
        'best_validation_response_loss': best_validation_loss,
        'heldout_test_response_loss': test_loss,
        'test_split_evaluated': include_test,
        'history': history,
        'checkpoint': checkpoint_path.name,
        'checkpoint_sha256': _sha256(checkpoint_path),
    }
    metadata_path = output_path / 'training_metadata.json'
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    print('Best validation response loss: %.4f' % best_validation_loss)
    if test_loss is not None:
        print('Held-out test response loss (exploratory): %.4f' % test_loss)
    else:
        print('Held-out test split not evaluated.')
    print('Checkpoint: %s' % checkpoint_path)
    print('Metadata: %s' % metadata_path)
    print(metadata['interpretation'])
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Run a small causal-LM training pipeline experiment')
    parser.add_argument('--data', default='data/pilot_v1.jsonl')
    parser.add_argument('--output', default='artifacts/pilot_run')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--patience', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--d-model', type=int, default=64)
    parser.add_argument('--num-heads', type=int, default=4)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--max-seq-len', type=int, default=1024)
    parser.add_argument(
        '--first-token-loss-weight',
        type=float,
        default=1.0,
        help=(
            'relative loss weight for each example’s first response token; '
            '1.0 preserves the standard token-mean objective'),
    )
    parser.add_argument(
        '--include-test',
        action='store_true',
        help=(
            'evaluate the held-out test split after training; use only after '
            'model and evaluation decisions are fixed'),
    )
    parser.add_argument(
            '--allow-draft-data',
            action='store_true',
            help=(
                'explicitly train on draft records; use only for a deliberately '
                'selected, licensed experimental source whose individual content '
                'has not been independently reviewed'),
    )
    args = parser.parse_args()
    train(
        args.data,
        args.output,
        seed=args.seed,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        max_seq_len=args.max_seq_len,
        include_test=args.include_test,
        first_token_loss_weight=args.first_token_loss_weight,
        allow_draft_data=args.allow_draft_data,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
