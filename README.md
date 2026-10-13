# Turkish model — Kaggle GPU experiments

This public repository contains only the files needed for a small Kaggle
training experiment. Use the two-stage notebook for the current run:

- [`kaggle_turkish_two_stage.ipynb`](./kaggle_turkish_two_stage.ipynb) trains
  the student from random initialization, first on Common Voice Turkish and
  then on dialogue data;
- the BPE two-stage trainer, data and inference helpers;
- a CC0 Common Voice train/validation-only archive;
- the train-only, repetition-capped Turkish Daily Dialogues derivative;
- provenance manifests, file hashes, and license information.

## Start training

1. Import [`kaggle_turkish_two_stage.ipynb`](./kaggle_turkish_two_stage.ipynb)
   into Kaggle from GitHub.
2. In notebook settings, enable **Internet** and select a **GPU** accelerator.
3. Choose **Run All**. Setup checks file hashes, installs the BPE tokenizer
   dependency, and refuses to continue without a GPU.
4. Save/download the output under `/kaggle/working` before the Kaggle session
   ends.

No GitHub token or Kaggle API token is needed. Internet access is required to
clone the source and install `tokenizers`.

## Data and limitations

The training file contains 2,480 synthetic, AI-assisted Turkish conversations
from the upstream train split only. The upstream validation/test splits are
not included. Whole conversations were selected with a deterministic cap on
exact repeated messages; this does not remove paraphrased templates.

Records are marked `draft`: the source and license were selected for this
experiment, but each conversation has not been independently human-reviewed.
The trainer's `--allow-draft-data` option is explicit and recorded in the run
metadata.

The byte-level BPE tokenizer is trained only on Common Voice train text and
dialogue train examples. The model is trained from scratch on Common Voice
and then fine-tuned on the group-safe dialogue train split. Common Voice is
Wikipedia-weighted, and sentence-level splitting cannot prevent related
sentences crossing splits. The dialogue set is small, synthetic, AI-assisted
and not individually human-reviewed. Validation is used for
monitoring/checkpoint selection; neither held-out test split is evaluated.
This remains an experiment, not a capable or production-ready chat model.

The older
[`kaggle_turkish_dialogue_pilot.ipynb`](./kaggle_turkish_dialogue_pilot.ipynb)
is the character-level baseline; use it only to reproduce the previous run.

## Attribution and license

The language pretraining data is the train/validation-only derivative of
[Common Voice Scripted Speech 27.0 Turkish](https://commonvoice.mozilla.org/),
licensed under **CC0-1.0**. Its prepared manifest is included in the archive;
the parent held-out test split is not packaged.

Source: [Turkish Daily Dialogues 5K](https://huggingface.co/datasets/3nesdeniz/turkish-daily-dialogues-5k)
by 3nesdeniz. Dataset license: **CC BY 4.0**. This repository contains a
train-only, deterministic repetition-capped and schema-converted derivative;
that modification is identified in the included manifest. Retain attribution
and the [`LICENSE`](./data/raw/turkish_daily_dialogues_sft_001/LICENSE) when
redistributing the data. The source project's code license is not changed by
this pilot repository.
