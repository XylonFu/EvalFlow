# EvalFlow - EvalScope Plugin for Model Deployment and Evaluation

## Introduction

EvalFlow is a plugin for EvalScope that simplifies model deployment and evaluation through a unified command-line interface. It handles both model serving and benchmark evaluation in one workflow.

## Basic Usage

```bash
python evalflow.py \
    --conda_env /path/to/conda_env \
    --work_dir /path/to/work_dir \
    --eval_model_path /path/to/model \
    --eval_model_name model_name \
    --judge_model_path /path/to/judge_model \
    --judge_model_name judge_model_name
```

## Parameters

### Required Parameters
- `--conda_env`: Path to conda environment with dependencies
- `--work_dir`: Directory for evaluation results
- `--eval_model_path`: Path to model being evaluated (can specify multiple)
- `--eval_model_name`: Name of model being evaluated (can specify multiple)
- `--judge_model_path`: Path to judge model
- `--judge_model_name`: Name of judge model

### Optional Parameters
- `--reuse`: Reuse cached results (True/False)
- `--datasets`: Space-separated list of datasets to evaluate
- `--eval_max_model_length`: Maximum context length (default: 32768)
- `--eval_max_new_tokens`: Maximum generation length (default: 2048)
- `--eval_max_num_seqs`: Maximum concurrent sequences (default: 200)
- `--eval_template`: Chat template path
- `--eval_temperature`: Sampling temperature (default: 0.0)
- `--eval_devices`: Comma-separated GPU devices (default: "0")
- `--judge_max_model_length`: Judge model context length (default: 8192)
- `--judge_max_num_seqs`: Judge model concurrent sequences (default: 200)
- `--judge_devices`: Judge model GPU devices (default: "0")
- `--eval_backend`: Evaluation backend ("VLMEvalKit" or "Native")
- `--deploy_backend`: Serving backend ("vLLM" or "LMDeploy")
