"""Train a byte-level BPE language model, then fine-tune it for dialogue."""

import argparse
import csv
import hashlib
import json
import math
import random
import tempfile
from pathlib import Path
from collections.abc import Callable, Iterable
from typing import Any

import torch
from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers, trainers
from torch import nn

from clean_model.data import (
    build_supervised_examples,
    dataset_fingerprint,
    load_jsonl,
    split_by_group,
)
from clean_model.engine import PilotGenerator, _prompt_text
from clean_model.train import (
    BOS_ID,
    EOS_ID,
    PAD_ID,
    PilotCausalLM,
    _batches,
    _response_loss_sum_and_count,
    _example_text,
)


SPECIAL_TOKEN_TEXT = ('<pad>', '<eos>', '<unk>', '<bos>')
DEFAULT_VOCAB_SIZE = 2048


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _read_sentences(path: Path) -> list[str]:
    expected_columns = (
        'sentence_id', 'sentence', 'variant', 'sentence_domain', 'source',
        'is_used', 'clips_count',
    )
    with path.open('r', encoding='utf-8-sig', newline='') as source:
        reader = csv.DictReader(source, delimiter='\t', strict=True)
        if tuple(reader.fieldnames or ()) != expected_columns:
            raise ValueError('Unexpected prepared Common Voice header: %s' % path)
        sentences = []
        for row in reader:
            if row.get(None) is not None or row.get('sentence') is None:
                raise ValueError('Malformed Common Voice row in %s' % path)
            sentence = row['sentence']
            if sentence.strip():
                sentences.append(sentence)
    if not sentences:
        raise ValueError('Common Voice split contains no sentences: %s' % path)
    return sentences


def _read_common_voice_manifest(directory: Path) -> dict[str, Any]:
    manifest_path = directory / 'manifest.json'
    with manifest_path.open('r', encoding='utf-8') as source:
        manifest = json.load(source)
    split_counts = manifest.get('split_counts', {})
    if (manifest.get('dataset')
            != 'Common Voice Scripted Speech 27.0 Turkish'
            or manifest.get('license') != 'CC0-1.0'
            or not isinstance(split_counts, dict)
            or not split_counts.get('train')
            or not split_counts.get('validation')):
        raise ValueError(
            'Common Voice manifest must identify the Turkish CC0 train and '
            'validation splits')
    if (directory / 'test.tsv').exists():
        raise ValueError(
            'Use the train/validation-only Common Voice derivative; test.tsv '
            'must not be present')
    return manifest


def _chat_tokenizer_texts(
    examples: Iterable[dict[str, Any]],
) -> Iterable[str]:
    for example in examples:
        prefix, target = _example_text(example)
        yield prefix
        yield target


def train_subword_tokenizer(
    common_voice_train_path: str | Path,
    dialogue_examples: list[dict[str, Any]],
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    min_frequency: int = 2,
) -> Tokenizer:
    """Train a byte-level BPE tokenizer from train splits only."""
    if vocab_size < 260:
        raise ValueError('vocab_size must be at least 260')
    if min_frequency < 1:
        raise ValueError('min_frequency must be positive')

    byte_level = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer = Tokenizer(models.BPE(unk_token='<unk>'))
    tokenizer.pre_tokenizer = byte_level
    tokenizer.decoder = decoders.ByteLevel()
    special_tokens = [
        AddedToken(token, special=True, normalized=False)
        for token in SPECIAL_TOKEN_TEXT
    ]
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        initial_alphabet=byte_level.alphabet(),
    )

    def training_texts() -> Iterable[str]:
        yield from (
            sentence for sentence in _read_sentences(
                Path(common_voice_train_path))
        )
        yield from _chat_tokenizer_texts(dialogue_examples)

    tokenizer.train_from_iterator(training_texts(), trainer=trainer)
    expected_ids = {
        token: index for index, token in enumerate(SPECIAL_TOKEN_TEXT)
    }
    actual_ids = {
        token: tokenizer.token_to_id(token) for token in SPECIAL_TOKEN_TEXT
    }
    if actual_ids != expected_ids:
        raise ValueError('BPE special-token IDs do not match model constants')
    return tokenizer


def encode_dialogue_example(
    example: dict[str, Any],
    tokenizer: Tokenizer,
    max_seq_len: int,
) -> tuple[list[int], list[int]]:
    """Encode a chat example and score only its assistant response and EOS."""
    if max_seq_len < 4:
        raise ValueError('max_seq_len must be at least 4')
    prefix, target = _example_text(example)
    prefix_ids = tokenizer.encode(prefix).ids
    target_ids = tokenizer.encode(target).ids
    available_prefix = max_seq_len - len(target_ids) - 2
    if available_prefix < 1:
        target_ids = target_ids[:max_seq_len - 3]
        available_prefix = 1
    prefix_ids = prefix_ids[-available_prefix:]
    tokens = [BOS_ID] + prefix_ids + target_ids + [EOS_ID]
    inputs = tokens[:-1]
    labels = [-100] * len(inputs)
    first_target_label = len(prefix_ids)
    labels[first_target_label:] = tokens[first_target_label + 1:]
    if not any(label != -100 for label in labels):
        raise ValueError('example has no assistant target tokens')
    return inputs, labels


def generation_prefix_ids(
    model: PilotCausalLM,
    tokenizer: Tokenizer,
    messages: list[dict[str, str]],
    evidence: list[dict[str, str]] | None = None,
) -> list[int]:
    """Encode the inference prompt using the same context truncation as SFT."""
    prompt = _prompt_text(messages, evidence or [])
    prefix_ids = tokenizer.encode(prompt).ids
    return [BOS_ID] + prefix_ids[-(model.max_seq_len - 1):]


def _encode_sentence_stream(
    sentences: list[str],
    tokenizer: Tokenizer,
    token_budget: int,
    seed: int,
) -> tuple[list[int], int]:
    if token_budget < 2:
        raise ValueError('token budget must be at least 2')
    order = list(range(len(sentences)))
    random.Random(seed).shuffle(order)
    token_stream: list[int] = []
    selected = 0
    for index in order:
        encoded = tokenizer.encode(sentences[index]).ids + [EOS_ID]
        if len(token_stream) + len(encoded) > token_budget:
            continue
        token_stream.extend(encoded)
        selected += 1
    if len(token_stream) < 2:
        raise ValueError('token budget selected fewer than two tokens')
    return token_stream, selected


def _lm_batches(
    token_stream: list[int],
    context_length: int,
    batch_size: int,
    shuffle: bool,
    rng: random.Random,
):
    if context_length < 2 or batch_size < 1:
        raise ValueError('context_length must be >= 2 and batch_size positive')
    block_width = context_length + 1
    blocks = [
        token_stream[start:start + block_width]
        for start in range(0, len(token_stream) - 1, context_length)
    ]
    blocks = [block for block in blocks if len(block) >= 2]
    if shuffle:
        rng.shuffle(blocks)
    for start in range(0, len(blocks), batch_size):
        selected = blocks[start:start + batch_size]
        width = max(len(block) - 1 for block in selected)
        inputs = torch.full(
            (len(selected), width), PAD_ID, dtype=torch.long)
        labels = torch.full(
            (len(selected), width), -100, dtype=torch.long)
        for row_index, block in enumerate(selected):
            inputs[row_index, :len(block) - 1] = torch.tensor(block[:-1])
            labels[row_index, :len(block) - 1] = torch.tensor(block[1:])
        yield inputs, labels


def _evaluate_lm(
    model: PilotCausalLM,
    token_stream: list[int],
    context_length: int,
    batch_size: int,
    device: torch.device,
) -> tuple[float, int]:
    model.eval()
    loss_total = 0.0
    token_total = 0
    with torch.no_grad():
        for input_ids, labels in _lm_batches(
                token_stream, context_length, batch_size, False,
                random.Random(0)):
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            logits = model(input_ids)
            count = int(labels.ne(-100).sum().item())
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=-100,
                reduction='sum',
            )
            loss_total += float(loss.item())
            token_total += count
    if not token_total:
        raise ValueError('Common Voice evaluation split has no target tokens')
    return loss_total / token_total, token_total


def _evaluate_sft(
    model: PilotCausalLM,
    examples: list[tuple[list[int], list[int]]],
    batch_size: int,
    device: torch.device,
) -> tuple[float, int]:
    model.eval()
    loss_total = 0.0
    token_total = 0
    with torch.no_grad():
        for input_ids, labels in _batches(
                examples, batch_size, False, random.Random(0)):
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            loss_sum, count = _response_loss_sum_and_count(
                model(input_ids), labels)
            loss_total += float(loss_sum.item())
            token_total += count
    if not token_total:
        raise ValueError('dialogue validation split has no target tokens')
    return loss_total / token_total, token_total


def _fit_stage(
    model: PilotCausalLM,
    train_batches: Callable[
        [int], Iterable[tuple[torch.Tensor, torch.Tensor]]],
    validation_fn: Callable[[], tuple[float, int]],
    epochs: int,
    patience: int,
    learning_rate: float,
    device: torch.device,
    stage_name: str,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]], int, float]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.01)
    best_loss = math.inf
    best_epoch = 0
    best_state = None
    waiting = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        loss_total = 0.0
        token_total = 0
        step_count = 0
        for input_ids, labels in train_batches(epoch):
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids)
            if stage_name == 'dialogue_finetune':
                loss_sum, token_count = _response_loss_sum_and_count(
                    logits, labels)
                if token_count == 0:
                    continue
                loss = loss_sum / token_count
            else:
                token_count = int(labels.ne(-100).sum().item())
                if token_count == 0:
                    continue
                loss_sum = nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    labels.reshape(-1),
                    ignore_index=-100,
                    reduction='sum',
                )
                loss = loss_sum / token_count
            if not torch.isfinite(loss):
                raise RuntimeError(
                    '%s produced non-finite loss at epoch %d'
                    % (stage_name, epoch))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            loss_total += float(loss_sum.item())
            token_total += token_count
            step_count += 1
        if not token_total or not step_count:
            raise ValueError('%s produced no optimizer steps' % stage_name)

        validation_loss, validation_tokens = validation_fn()
        history.append({
            'epoch': epoch,
            'train_loss': loss_total / token_total,
            'validation_loss': validation_loss,
            'train_tokens': token_total,
            'validation_tokens': validation_tokens,
            'optimizer_steps': step_count,
        })
        print(
            '%s epoch %d/%d train_loss=%.4f validation_loss=%.4f steps=%d'
            % (stage_name, epoch, epochs, loss_total / token_total,
               validation_loss, step_count),
            flush=True,
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            waiting = 0
        else:
            waiting += 1
            if waiting >= patience:
                break
    if best_state is None:
        raise RuntimeError('%s produced no selected checkpoint' % stage_name)
    return best_state, history, best_epoch, best_loss


def train_two_stage(
    common_voice_dir: str | Path,
    dialogue_data_path: str | Path,
    output_dir: str | Path,
    seed: int = 7,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    pretrain_epochs: int = 4,
    finetune_epochs: int = 12,
    patience: int = 3,
    batch_size: int = 16,
    learning_rate: float = 3e-4,
    finetune_learning_rate: float = 1e-4,
    d_model: int = 128,
    num_heads: int = 4,
    num_layers: int = 4,
    max_seq_len: int = 256,
    train_token_budget: int = 2_000_000,
    validation_token_budget: int = 250_000,
    allow_draft_data: bool = False,
) -> dict[str, Any]:
    """Pretrain on Common Voice, then fine-tune on group-split dialogues."""
    if torch is None:
        raise RuntimeError('Training requires PyTorch')
    if pretrain_epochs < 1 or finetune_epochs < 1 or patience < 1:
        raise ValueError('epochs and patience must be positive')
    if batch_size < 1 or max_seq_len < 4:
        raise ValueError('batch_size must be positive and max_seq_len >= 4')
    if (not math.isfinite(learning_rate) or learning_rate <= 0
            or not math.isfinite(finetune_learning_rate)
            or finetune_learning_rate <= 0):
        raise ValueError('learning rates must be finite and positive')

    corpus_path = Path(common_voice_dir)
    dialogue_path = Path(dialogue_data_path)
    output_path = Path(output_dir)
    if output_path.exists():
        raise ValueError('Output directory already exists: %s' % output_path)
    output_path.mkdir(parents=True)

    corpus_manifest = _read_common_voice_manifest(corpus_path)
    train_sentences = _read_sentences(corpus_path / 'train.tsv')
    validation_sentences = _read_sentences(corpus_path / 'validation.tsv')
    require_approved = not allow_draft_data
    records = load_jsonl(
        dialogue_path, require_approved=require_approved)
    splits = split_by_group(
        records, seed=seed, require_approved=require_approved)
    train_dialogue_examples = list(build_supervised_examples(
        splits['train'], require_approved=require_approved))
    validation_dialogue_examples = list(build_supervised_examples(
        splits['validation'], require_approved=require_approved))

    tokenizer = train_subword_tokenizer(
        corpus_path / 'train.tsv',
        train_dialogue_examples,
        vocab_size=vocab_size,
    )
    train_tokens, selected_train_sentences = _encode_sentence_stream(
        train_sentences, tokenizer, train_token_budget, seed)
    validation_tokens, selected_validation_sentences = _encode_sentence_stream(
        validation_sentences, tokenizer, validation_token_budget, seed + 1)
    encoded_dialogue = {
        'train': [
            encode_dialogue_example(
                example, tokenizer, max_seq_len)
            for example in train_dialogue_examples
        ],
        'validation': [
            encode_dialogue_example(
                example, tokenizer, max_seq_len)
            for example in validation_dialogue_examples
        ],
    }

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_config = {
        'vocab_size': tokenizer.get_vocab_size(),
        'd_model': d_model,
        'num_heads': num_heads,
        'num_layers': num_layers,
        'max_seq_len': max_seq_len,
        'dropout': 0.1,
    }
    model = PilotCausalLM(**model_config).to(device)

    def pretrain_batches(epoch: int):
        return _lm_batches(
            train_tokens, max_seq_len, batch_size, True,
            random.Random(seed + epoch))

    pretrained_state, pretrain_history, pretrain_best_epoch, pretrain_best_loss = (
        _fit_stage(
            model,
            pretrain_batches,
            lambda: _evaluate_lm(
                model, validation_tokens, max_seq_len, batch_size, device),
            pretrain_epochs,
            patience,
            learning_rate,
            device,
            'language_pretrain',
        ))
    model.load_state_dict(pretrained_state)

    def finetune_batches(epoch: int):
        return _batches(
            encoded_dialogue['train'], batch_size, True,
            random.Random(seed + 10_000 + epoch))

    finetuned_state, finetune_history, finetune_best_epoch, finetune_best_loss = (
        _fit_stage(
            model,
            finetune_batches,
            lambda: _evaluate_sft(
                model, encoded_dialogue['validation'], batch_size, device),
            finetune_epochs,
            patience,
            finetune_learning_rate,
            device,
            'dialogue_finetune',
        ))
    model.load_state_dict(finetuned_state)

    tokenizer_json = tokenizer.to_str()
    checkpoint_path = output_path / 'pilot_model.pt'
    with tempfile.NamedTemporaryFile(
            mode='wb', dir=output_path, delete=False,
            prefix='pilot_model.', suffix='.tmp') as temporary:
        torch.save({
            'schema_version': 2,
            'tokenizer_type': 'byte_level_bpe',
            'tokenizer_json': tokenizer_json,
            'model_config': model_config,
            'model_state': finetuned_state,
        }, temporary)
        temporary_path = Path(temporary.name)
    temporary_path.replace(checkpoint_path)
    tokenizer_path = output_path / 'tokenizer.json'
    tokenizer_path.write_text(tokenizer_json + '\n', encoding='utf-8')

    generator = PilotGenerator(checkpoint_path, device='cpu')
    review_examples = validation_dialogue_examples[:8]
    generation_samples = [{
        'record_id': example['record_id'],
        'context': example['context'],
        'reference': example['target'],
        'generated': generator.generate(
            example['context'],
            evidence=example['evidence'],
            max_new_tokens=80,
        ),
    } for example in review_examples]
    (output_path / 'generation_samples.json').write_text(
        json.dumps(generation_samples, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )

    metadata = {
        'schema_version': 1,
        'experiment': 'common_voice_bpe_pretrain_dialogue_finetune',
        'interpretation': (
            'From-scratch byte-level BPE language-model pretraining followed '
            'by dialogue fine-tuning; experimental, not evidence of useful '
            'generalization or production readiness.'),
        'data': {
            'common_voice_dataset': corpus_manifest['dataset'],
            'common_voice_license': corpus_manifest['license'],
            'common_voice_manifest_sha256': _sha256(
                corpus_path / 'manifest.json'),
            'common_voice_train_sha256': _sha256(
                corpus_path / 'train.tsv'),
            'common_voice_validation_sha256': _sha256(
                corpus_path / 'validation.tsv'),
            'common_voice_test_read': False,
            'common_voice_train_sentences_selected': selected_train_sentences,
            'common_voice_validation_sentences_selected':
                selected_validation_sentences,
            'dialogue_sha256': _sha256(dialogue_path),
            'dialogue_fingerprint': dataset_fingerprint(
                records, require_approved=require_approved),
            'dialogue_draft_records_included': allow_draft_data,
            'dialogue_review_status_counts': {
                status: sum(
                    row['review']['status'] == status for row in records)
                for status in ('approved', 'draft')
            },
            'dialogue_split_seed': seed,
            'dialogue_split_counts': {
                name: len(rows) for name, rows in splits.items()
            },
            'dialogue_split_group_ids': {
                name: sorted({row['group_id'] for row in rows})
                for name, rows in splits.items()
            },
            'dialogue_test_split_read': False,
        },
        'tokenizer': {
            'type': 'byte_level_bpe',
            'vocab_size': tokenizer.get_vocab_size(),
            'requested_vocab_size': vocab_size,
            'training_splits': [
                'Common Voice train',
                'dialogue train',
            ],
            'sha256': _sha256(tokenizer_path),
        },
        'device': str(device),
        'torch_version': torch.__version__,
        'model_config': model_config,
        'pretraining': {
            'token_budget': train_token_budget,
            'validation_token_budget': validation_token_budget,
            'train_tokens': len(train_tokens),
            'validation_tokens': len(validation_tokens),
            'epochs_completed': len(pretrain_history),
            'best_epoch': pretrain_best_epoch,
            'best_validation_loss': pretrain_best_loss,
            'history': pretrain_history,
        },
        'dialogue_finetuning': {
            'train_examples': len(train_dialogue_examples),
            'validation_examples': len(validation_dialogue_examples),
            'epochs_completed': len(finetune_history),
            'best_epoch': finetune_best_epoch,
            'best_validation_loss': finetune_best_loss,
            'history': finetune_history,
        },
        'checkpoint_sha256': _sha256(checkpoint_path),
        'test_split_evaluated': False,
        'generation_sample_count': len(generation_samples),
    }
    (output_path / 'training_metadata.json').write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    print('Checkpoint:', checkpoint_path)
    print('Metadata:', output_path / 'training_metadata.json')
    print('Held-out Common Voice and dialogue test splits not read/evaluated.')
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            'Pretrain a small byte-level BPE LM on Common Voice and fine-tune '
            'it on Turkish dialogues'))
    parser.add_argument(
        '--common-voice',
        default='data/raw/common_voice_27_active_trainval_001',
    )
    parser.add_argument(
        '--dialogue-data',
        default='data/raw/turkish_daily_dialogues_sft_001/train.jsonl',
    )
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--vocab-size', type=int, default=DEFAULT_VOCAB_SIZE)
    parser.add_argument('--pretrain-epochs', type=int, default=4)
    parser.add_argument('--finetune-epochs', type=int, default=12)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--finetune-learning-rate', type=float, default=1e-4)
    parser.add_argument('--d-model', type=int, default=128)
    parser.add_argument('--num-heads', type=int, default=4)
    parser.add_argument('--num-layers', type=int, default=4)
    parser.add_argument('--max-seq-len', type=int, default=256)
    parser.add_argument('--train-token-budget', type=int, default=2_000_000)
    parser.add_argument(
        '--validation-token-budget', type=int, default=250_000)
    parser.add_argument('--allow-draft-data', action='store_true')
    args = parser.parse_args()
    train_two_stage(
        common_voice_dir=args.common_voice,
        dialogue_data_path=args.dialogue_data,
        output_dir=args.output,
        seed=args.seed,
        vocab_size=args.vocab_size,
        pretrain_epochs=args.pretrain_epochs,
        finetune_epochs=args.finetune_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        finetune_learning_rate=args.finetune_learning_rate,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        max_seq_len=args.max_seq_len,
        train_token_budget=args.train_token_budget,
        validation_token_budget=args.validation_token_budget,
        allow_draft_data=args.allow_draft_data,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
