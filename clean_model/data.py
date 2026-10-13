"""Validation and deterministic splitting for the clean-model dataset."""

import argparse
import hashlib
import json
import math
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Iterator


SCHEMA_VERSION = 1
ROLES = ('user', 'assistant')


def validate_record(record: Any, require_approved: bool = True) -> None:
    """Raise ValueError unless a JSONL record meets the selected review gate."""
    if not isinstance(record, dict):
        raise ValueError('record must be a JSON object')
    if record.get('schema_version') != SCHEMA_VERSION:
        raise ValueError('schema_version must be 1')

    for field in ('id', 'group_id'):
        if not isinstance(record.get(field), str) or not record[field].strip():
            raise ValueError('%s must be a non-empty string' % field)

    kind = record.get('kind')
    if kind not in ('dialogue', 'grounded_qa'):
        raise ValueError("kind must be 'dialogue' or 'grounded_qa'")

    messages = record.get('messages')
    if not isinstance(messages, list) or not messages:
        raise ValueError('messages must be a non-empty list')
    roles = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError('messages[%d] must be an object' % index)
        role = message.get('role')
        content = message.get('content')
        if role == 'system' and index == 0:
            pass
        elif role in ROLES:
            roles.append(role)
        else:
            raise ValueError(
                'messages[%d].role must follow user/assistant order' % index)
        if not isinstance(content, str) or not content.strip():
            raise ValueError(
                'messages[%d].content must be a non-empty string' % index)

    if not roles or roles[0] != 'user' or roles[-1] != 'assistant':
        raise ValueError('messages must start with user and end with assistant')
    for previous, current in zip(roles, roles[1:]):
        if previous == current:
            raise ValueError('user and assistant messages must alternate')
    if kind == 'grounded_qa' and len(roles) != 2:
        raise ValueError('grounded_qa records must contain one user/assistant turn')

    provenance = record.get('provenance')
    if not isinstance(provenance, dict):
        raise ValueError('provenance must be an object')
    for field in ('dataset', 'license', 'reference'):
        if not isinstance(provenance.get(field), str) or not provenance[field].strip():
            raise ValueError('provenance.%s must be a non-empty string' % field)

    review = record.get('review')
    if not isinstance(review, dict) or review.get('status') not in (
            'draft', 'approved'):
        raise ValueError('review.status must be draft or approved')
    if require_approved and review['status'] != 'approved':
        raise ValueError('only records with review.status=approved are accepted')
    if review['status'] == 'approved':
        if not isinstance(review.get('reviewed_by'), str) or not review['reviewed_by'].strip():
            raise ValueError('review.reviewed_by must be a non-empty string')
        reviewed_at = review.get('reviewed_at')
        if not isinstance(reviewed_at, str):
            raise ValueError('review.reviewed_at must be an ISO date')
        try:
            date.fromisoformat(reviewed_at)
        except ValueError as exc:
            raise ValueError('review.reviewed_at must be an ISO date') from exc

    evidence = record.get('evidence', [])
    if not isinstance(evidence, list):
        raise ValueError('evidence must be a list')
    if kind == 'grounded_qa' and not evidence:
        raise ValueError('grounded_qa records require at least one evidence source')
    source_ids = set()
    for index, source in enumerate(evidence):
        if not isinstance(source, dict):
            raise ValueError('evidence[%d] must be an object' % index)
        for field in (
                'source_id', 'title', 'text', 'text_type', 'license',
                'reference', 'retrieved_at'):
            if not isinstance(source.get(field), str) or not source[field].strip():
                raise ValueError(
                    'evidence[%d].%s must be a non-empty string'
                    % (index, field))
        try:
            date.fromisoformat(source['retrieved_at'])
        except ValueError as exc:
            raise ValueError(
                'evidence[%d].retrieved_at must be an ISO date'
                % index) from exc
        if source['text_type'] not in ('verbatim', 'paraphrase'):
            raise ValueError(
                'evidence[%d].text_type must be verbatim or paraphrase'
                % index)
        if source['source_id'] in source_ids:
            raise ValueError('evidence source_id values must be unique per record')
        source_ids.add(source['source_id'])
    if kind == 'dialogue' and evidence:
        raise ValueError('dialogue records must not include factual evidence')


def load_jsonl(
    path: str | Path,
    require_approved: bool = True,
) -> list[dict[str, Any]]:
    """Load and validate a JSONL dataset, reporting the exact failing line."""
    records = []
    seen_ids = set()
    with Path(path).open('r', encoding='utf-8') as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                validate_record(record, require_approved=require_approved)
                if record['id'] in seen_ids:
                    raise ValueError('duplicate record id %r' % record['id'])
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError('%s:%d: %s' % (path, line_number, exc)) from exc
            seen_ids.add(record['id'])
            records.append(record)
    if not records:
        raise ValueError('%s: dataset contains no records' % path)
    return records


def dataset_fingerprint(
    records: Iterable[dict[str, Any]],
    require_approved: bool = True,
) -> str:
    """Hash canonical record JSON in its current order."""
    digest = hashlib.sha256()
    for record in records:
        validate_record(record, require_approved=require_approved)
        encoded = json.dumps(
            record, ensure_ascii=False, sort_keys=True,
            separators=(',', ':')).encode('utf-8')
        digest.update(len(encoded).to_bytes(8, 'little'))
        digest.update(encoded)
    return digest.hexdigest()


def build_supervised_examples(
    records: Iterable[dict[str, Any]],
    require_approved: bool = True,
) -> Iterator[dict[str, Any]]:
    """Yield prior conversation/evidence as context and assistant text as target."""
    for record in records:
        validate_record(record, require_approved=require_approved)
        history = []
        for message in record['messages']:
            if message['role'] == 'assistant':
                yield {
                    'record_id': record['id'],
                    'group_id': record['group_id'],
                    'kind': record['kind'],
                    'context': [dict(item) for item in history],
                    'evidence': [dict(item) for item in record.get('evidence', [])],
                    'target': message['content'],
                }
            history.append(message)


def split_by_group(
    records: list[dict[str, Any]],
    seed: int = 7,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    require_approved: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    """Deterministically split records without separating a group across sets."""
    if not math.isfinite(train_fraction) or not math.isfinite(validation_fraction):
        raise ValueError('split fractions must be finite')
    test_fraction = 1.0 - train_fraction - validation_fraction
    if train_fraction <= 0 or validation_fraction <= 0 or test_fraction <= 0:
        raise ValueError('train, validation, and test fractions must be positive')

    group_kinds = {}
    group_sizes = {}
    for record in records:
        validate_record(record, require_approved=require_approved)
        group_kinds.setdefault(record['group_id'], set()).add(record['kind'])
        group_sizes[record['group_id']] = (
            group_sizes.get(record['group_id'], 0) + 1)
    if len(group_kinds) < 3:
        raise ValueError(
            'at least three distinct group_id values are required for '
            'train/validation/test splits')

    group_splits = {}
    strata = {}
    for group_id, kinds in group_kinds.items():
        strata.setdefault(tuple(sorted(kinds)), []).append(group_id)

    for stratum_groups in strata.values():
        ordered_groups = sorted(
            stratum_groups,
            key=lambda group_id: (
                -group_sizes[group_id],
                hashlib.sha256(
                    ('%d\0%s' % (seed, group_id)).encode('utf-8')).digest()))
        group_count = len(ordered_groups)
        if group_count == 1:
            group_splits[ordered_groups[0]] = 'train'
            continue
        elif group_count == 2:
            group_splits[ordered_groups[0]] = 'train'
            group_splits[ordered_groups[1]] = 'validation'
            continue

        stratum_size = sum(group_sizes[group] for group in ordered_groups)
        targets = (
            stratum_size * train_fraction,
            stratum_size * validation_fraction,
            stratum_size * test_fraction,
        )
        names = ('train', 'validation', 'test')
        assigned_sizes = [0, 0, 0]
        assigned_groups = [0, 0, 0]
        for index, group_id in enumerate(ordered_groups):
            remaining_groups = group_count - index
            empty_splits = [
                split_index for split_index, count
                in enumerate(assigned_groups) if count == 0
            ]
            candidates = (
                empty_splits
                if remaining_groups == len(empty_splits)
                else range(len(names))
            )
            split_index = max(
                candidates,
                key=lambda candidate: (
                    (targets[candidate] - assigned_sizes[candidate])
                    / max(targets[candidate], 1),
                    -candidate,
                ))
            group_splits[group_id] = names[split_index]
            assigned_sizes[split_index] += group_sizes[group_id]
            assigned_groups[split_index] += 1

    result = {'train': [], 'validation': [], 'test': []}
    for record in records:
        result[group_splits[record['group_id']]].append(record)
    empty_splits = [name for name, items in result.items() if not items]
    if empty_splits:
        raise ValueError(
            'group/kind distribution leaves empty split(s): %s'
            % ', '.join(empty_splits))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Validate and fingerprint a clean-model JSONL dataset')
    parser.add_argument('--input', required=True, help='Path to curated JSONL')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument(
        '--audit-drafts',
        action='store_true',
        help='Audit draft records without making a train/validation/test split')
    args = parser.parse_args()

    try:
        records = load_jsonl(
            args.input, require_approved=not args.audit_drafts)
    except ValueError as exc:
        parser.error(str(exc))
    kind_counts = {}
    review_counts = {}
    for record in records:
        kind_counts[record['kind']] = kind_counts.get(record['kind'], 0) + 1
        status = record['review']['status']
        review_counts[status] = review_counts.get(status, 0) + 1
    print('Dataset records: %d' % len(records))
    print('Dataset SHA-256: %s' % dataset_fingerprint(
        records, require_approved=not args.audit_drafts))
    print('Kinds: %s' % json.dumps(kind_counts, sort_keys=True))
    print('Review statuses: %s' % json.dumps(review_counts, sort_keys=True))
    if review_counts.get('draft'):
        print('Split preview: not run; approve records before splitting.')
    else:
        try:
            splits = split_by_group(records, seed=args.seed)
        except ValueError as exc:
            parser.error(str(exc))
        print('Group-safe split counts: %s' % json.dumps(
            {name: len(items) for name, items in splits.items()}, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
