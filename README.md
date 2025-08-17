# EvalFlow

EvalFlow is a plugin for [EvalScope](https://github.com/modelscope/evalscope) that simplifies model deployment and evaluation through a unified command-line interface. It handles both model serving and benchmark evaluation in one workflow.

## Basic Usage

### Evaluate Local Models (vLLM/LMDeploy)
```bash
python evalflow.py \
    --conda_env /path/to/conda_env \
    --work_dir /path/to/work_dir \
    --eval_model_path /path/to/eval_model \
    --eval_model_name eval_model_name \
    --judge_model_name judge_model_name \
    --judge_api_url judge_api_url \
    --judge_api_key judge_api_key
```

### Evaluate Remote API Models
```bash
python evalflow.py \
    --work_dir /path/to/work_dir \
    --deploy_backend remote \
    --remote_api_url https://api.example.com/v1 \
    --eval_model_name eval_model_name \
    --judge_model_name judge_model_name \
    --judge_api_url judge_api_url \
    --judge_api_key judge_api_key
```

## Parameters

### Required Parameters
- `--work_dir`: Working directory for evaluation results and temporary files (required)
- `--eval_model_name`: Name(s) of the model(s) being evaluated (can specify multiple) (required)
- `--judge_model_name`: Served name of the judge model (required)
- `--judge_api_url`: API URL for judge service (required); for `VLMEvalKit`, ensure it ends with `/v1/chat/completions`; for `Native`, ensure it ends with `/v1`.
- `--judge_api_key`: API KEY for judge service (required)

### Conditional Required Parameters
- `--conda_env`: Required only when `deploy_backend` is `vllm` or `lmdeploy`
- `--eval_model_path`: Required when `deploy_backend` is `vllm` or `lmdeploy`
- `--remote_api_url`: Required when `deploy_backend` is `remote` (supports multiple URLs)

### Optional Parameters - General
- `--reuse`: Whether to reuse existing results (True/False, default: True)
- `--datasets`: Space-separated list of datasets to evaluate (default: LogicVista MathVista_MINI WeMath MathVision MathVerse_MINI DynaMath)

### Optional Parameters - Evaluation Model
- `--eval_max_model_length`: Maximum context length for evaluation model (default: 32768)
- `--eval_max_new_tokens`: Maximum new tokens to generate (default: 2048)
- `--eval_max_num_seqs`: Maximum concurrent sequences for evaluation (default: 200)
- `--eval_template`: Path to chat template file (default: None)
- `--eval_temperature`: Sampling temperature (0.0 for deterministic) (default: 0.0)
- `--eval_devices`: Comma-separated GPU devices for evaluation (default: "0")
- `--eval_host`: Host address for evaluation service (default: "127.0.0.1")
- `--eval_port`: Port for evaluation service (default: 8000)
- `--eval_api_key`: API KEY for evaluation service (default: "EMPTY")

### Optional Parameters - Judge Model
- `--judge_max_num_seqs`: Maximum concurrent sequences for judging (default: 200)

### Optional Parameters - Backend Configuration
- `--eval_backend`: Evaluation backend (`VLMEvalKit` or `Native`, default: `VLMEvalKit`)
- `--vlmevalkit_mode`: VLMEvalKit evaluation mode (`all` or `infer`, default: `all`)
- `--deploy_backend`: Model serving backend (`vllm`, `lmdeploy`, or `remote`, default: `vllm`)
