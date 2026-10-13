# Turkish Daily Dialogues — Kaggle GPU pilot

This public repository contains only the files needed for a small Kaggle
training experiment:

- the clean-model character-level trainer and its data/inference helpers;
- the train-only, repetition-capped Turkish Daily Dialogues derivative;
- its provenance manifest and CC BY 4.0 legal text; and
- a Kaggle notebook that clones this repository, verifies file hashes, and
  starts the experiment.

## Start training

1. Open [`kaggle_turkish_dialogue_pilot.ipynb`](./kaggle_turkish_dialogue_pilot.ipynb)
   in Kaggle using **File → Import Notebook → GitHub** (or Kaggle's equivalent
   GitHub notebook import).
2. In notebook settings, enable **Internet** and select a **GPU** accelerator.
3. Choose **Run All**. The first code cell clones this public repository and
   checks the source and file hashes. The later cells configure and start
   training.
4. Save/download the output under `/kaggle/working` before the Kaggle session
   ends.

No GitHub token, Kaggle API token, or other credential is needed. The Kaggle
notebook stops if it cannot find a GPU. Internet access is required to clone
the source.

## Data and limitations

The training file contains 2,480 synthetic, AI-assisted Turkish conversations
from the upstream train split only. The upstream validation/test splits are
not included. Whole conversations were selected with a deterministic cap on
exact repeated messages; this does not remove paraphrased templates.

Records are marked `draft`: the source and license were selected for this
experiment, but each conversation has not been independently human-reviewed.
The trainer's `--allow-draft-data` option is explicit and recorded in the run
metadata.

The model is a small character-level pilot. A GPU should reduce runtime, but
does not establish conversational quality or fix limitations in the data.
The validation split is used only for monitoring/checkpoint selection; the
held-out test split is not evaluated by this notebook.

## Attribution and license

Source: [Turkish Daily Dialogues 5K](https://huggingface.co/datasets/3nesdeniz/turkish-daily-dialogues-5k)
by 3nesdeniz. Dataset license: **CC BY 4.0**. This repository contains a
train-only, deterministic repetition-capped and schema-converted derivative;
that modification is identified in the included manifest. Retain attribution
and the [`LICENSE`](./data/raw/turkish_daily_dialogues_sft_001/LICENSE) when
redistributing the data. The source project's code license is not changed by
this pilot repository.
