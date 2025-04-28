# Revisiting COMET: Adapting to New Multilingual Encoders

This repository contains the code for the paper **"Revisiting COMET: Adapting to New Multilingual Encoders."**  
It is adapted from the [original COMET repository](https://github.com/Unbabel/COMET).

---

## Quick Installation

First, create a virtual environment and install the required dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Data

Before running any scripts, download the `data/` folder from [this link](https://drive.google.com/file/d/1yhMx111AOwBaN5zw1jt9wd6h1cd6jY54/view?usp=sharing) and place it at the root of the repository.

---

## Training

We define `$config_model` as either:
- `configs/ranking_model.yaml`
- `configs/regression_model.yaml`  
depending on the architecture you want to train.

(No experiments were conducted with the referenceless model, but it should work.)

To change the multilingual encoder, edit the `pretrained_model` line in `$config_model`.  
The encoders we tested are listed in `encoder_config.py`.

Then, train the model with:

```bash
python comet/cli/train.py --cfg "$config_model" --seed_everything 12
```

### Hyperparameter Search

For hyperparameter tuning, specify the search space in `configs/search_space.json` and run:

```bash
python comet/cli/train.py --cfg "$config_model" --seed_everything 12 --search --search_space configs/models/search_space.json --n_jobs 2
```

where `n_jobs` is the number of concurrent hyperparameter searches to run (using **Optuna**).

---

## Scoring

Once your model is trained, save the checkpoint path as `$checkpoint_path`.

To score on the **WMT24 Metrics Shared Task** and the three challenge sets (**AfriMTE**, **IndicMTE**, and **BioMQM**), run:

```bash
python evaluation/prepare_scores.py --baseline name_of_your_metric --set all --checkpoint $checkpoint_path
```

You can also score only on WMT24 or on the challenge sets separately by specifying `--set wmt24` or `--set challenge`.

**Important**: Your metric name (`--baseline`) should start with `comet`, e.g., `comet_rank_nllb` or `comet_reg_glot500`.

To perform LLM inference with **Gemma 3 27B**, run:

```bash
python evaluation/prepare_scores.py --baseline llm --set challenge
```

- Partial results are saved in `results/scores/partial/` (which you can delete after final aggregation).
- Full results are saved in `results/scores/`.

---

## Meta-Evaluation

Once scoring is complete, you can run the meta-evaluation.

Use the `--evalset` argument to select among:
- `wmt24_prim`
- `wmt24_sec`
- `afrimte`
- `indicmte`
- `biomqm`

and the `--k` argument to set the number of resampling runs (for WMT24 sets).

**Example**:

```bash
python evaluation/meta_evaluation.py --evalset wmt24_sec --k 200
```

- Using `k=200` takes about 4 hours for the primary tasks and about 18 hours for the secondary tasks.
