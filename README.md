# Neural Multi-Objective Combinatorial Optimization for Flexible Job Shop Scheduling Problems

## Introduction

This repository contains the code for our paper **[Neural Multi-Objective Combinatorial Optimization for Flexible Job Shop Scheduling Problems](https://openreview.net/forum?id=YAgOaYedLQ)**, accepted at the Fourteenth International Conference on Learning Representations (ICLR 2026). The code builds upon [this](https://github.com/wrqccc/FJSP-DRL) repository.

To cite our work, please refer to the BibTeX entry below:

```bibtex
@inproceedings{
    smit2026neural,
    title={Neural Multi-Objective Combinatorial Optimization for Flexible Job Shop Scheduling Problems},
    author={Igor G. Smit and Yaoxin Wu and Pavel Troubil and Yingqian Zhang and Wim P.M. Nuijten},
    booktitle={The Fourteenth International Conference on Learning Representations},
    year={2026},
    url={https://openreview.net/forum?id=YAgOaYedLQ}
}
```

## File Guide

The main code of our method is organized as follows:

- common_utils.py — Shared helpers (dispatching rules, sampling, utilities).
- data_utils.py — Data generation/loading, reference points, format conversion.
- enums.py — Enumerations for objectives and model architectures.
- fjsp_env_same_op_nums.py — FJSP environment with fixed operations per job.
- fjsp_env_various_op_nums.py — FJSP environment with variable operations per job.
- mo_fjsp_env_same_op_nums.py — Multi-objective env (fixed ops/job) with reward variants.
- mo_fjsp_env_various_op_nums.py — Multi-objective env (variable ops/job) with reward variants.
- multipliers.py — Normalization multipliers for objectives per dataset/size.
- params.py — Argument parsing and global config assembly.
- ortools_solver.py — OR-Tools baseline solver and exporter.
- print_test_result.py — Aggregate and export test results to Excel.
- read_or_tools_solutions.py — Read OR-Tools solutions and compute hypervolume stats.
- mo_train.py — Multi-objective training loop (trainer subclass).
- mo_test_trained_model.py — Multi-objective evaluation (greedy/sampling, Pareto, HV).
- model/main_model.py — Model definitions (DANIEL variants, MO variants).
- model/attention_layer.py — Core attention blocks for operations/machines.
- model/mo_attention_layer.py — Preference-aware attention blocks.
- model/sub_layers.py — MLP/Actor/Critic/Hypersupport layers.
- model/PPO.py — PPO implementation and policy selection.
- datasets in `data/` — Provided training/validation/test instances and reference points.

For the MOEA/D and NSGA-II baselines, the code is organized in the `evolutionary_algorithms/` directory, with separate scripts for configuration generation and algorithm execution. To run these, use the `run_moead.py` and `run_genetic_algorithm.py` scripts, which read from the generated configuration files in `evolutionary_algorithms/configs/`. `generate_configs.py` shows several examples for how to generate config files for different datasets, instance sizes, and objective combinations. The `configs` folder also contains some example configuration files.

## Training & Testing Entry Points

- Training (multi-objective): `mo_train.py`
- Testing (multi-objective): `mo_test_trained_model.py`
- Baseline: `ortools_solver.py`

## Most Relevant Command Line Arguments

For the main experiments, the following arguments are most relevant. For a full list, see `params.py`.

- `--model_source` — Dataset source for training (SD1, SD2, FFSP, JSSP).
- `--data_source` — Dataset source for testing (SD1, SD2, FFSP, JSSP, BenchData).
- `--n_j` and `--n_m` — Number of jobs and machines for training/validation data.
- `--max_updates` — Max training updates (e.g., 1500 for multi-objective).
- `test_data` — Test dataset name (e.g., "10x5" for 10 jobs, 5 machines).
- `test_mode` — Whether to use sampling (true) or greedy (false) during testing.
- `test_model` — Name of the trained model to load for testing. For details on naming conventions, see the `_set_model_name` method in `mo_train.py`.
- `objective_fn` — Objectives to optimize (e.g., makespan costs or makespan flowtime costs).
- `num_preferences_test` — Number of preference points to evaluate during testing.
- `model_architecture` — Model architecture to use. The architecture names match to the paper names as follows:
  - `mo_daniel_enc_weight_input` → WI-DAN
  - `mo_daniel_conditional_op_and_mch` → DCAN
  - `mo_daniel_conditional_op_and_mch_fea_input` → WI-DCAN
  - `hyper_daniel` → HYPER

For the ablation studies, the following arguments are used:

- `use_simple_reward` — Whether to use simple reward mode (true) or not (false) in the multi-objective environments.
- `use_lb_features` — Whether to include lower bound features in the model input (true) or not (false).

## Example Commands (fill placeholders)

### Train (multi-objective)
  
```shell
python mo_train.py --model_source=SD1 --data_source=SD1 --n_j=10 --n_m=5 --model_architecture="mo_daniel_conditional_op_and_mch" --max_updates=1500 --objective_fn makespan costs
```

### Test (multi-objective, sampling)

```shell
python mo_test_trained_model.py --data_source=SD1 --model_source=SD1 --test_data=10x5 --test_mode=true --test_model=10x5_mo_daniel_conditional_op_and_mch_mak_cos_MO --model_architecture=mo_daniel_conditional_op_and_mch --objective_fn makespan costs
```

### Test (multi-objective, greedy)

```shell
python mo_test_trained_model.py --data_source=SD1 --model_source=SD1 --test_data=10x5 --test_mode=false --test_model=10x5_mo_daniel_conditional_op_and_mch_mak_cos_MO --model_architecture=mo_daniel_conditional_op_and_mch --objective_fn makespan costs
```

### OR-Tools baseline

```shell
python ortools_solver.py --objective_fn makespan costs --max_solve_time 60 --data_source SD1 --test_data 10x5
```
