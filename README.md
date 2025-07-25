# EvalFlow

EvalFlow is a plugin for [EvalScope](https://github.com/modelscope/evalscope) that simplifies model deployment and evaluation through a unified command-line interface. It handles both model serving and benchmark evaluation in one workflow.

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
- `--conda_env`: Path to conda environment with dependencies (required)
- `--work_dir`: Working directory for evaluation results and temporary files (required)
- `--eval_model_path`: Path(s) to the model(s) being evaluated (can specify multiple) (required)
- `--eval_model_name`: Name(s) of the model(s) being evaluated (can specify multiple) (required)
- `--judge_model_path`: Path to the judge/grader model (required)
- `--judge_model_name`: Name of the judge/grader model (required)

### Optional Parameters - General
- `--reuse`: Whether to reuse existing results (True/False, default: True)
- `--datasets`: Space-separated list of datasets to evaluate (default: LogicVista MMMU_DEV_VAL MathVista_MINI WeMath MathVision MathVerse_MINI DynaMath)

### Optional Parameters - Evaluation Model
- `--eval_max_model_length`: Maximum context length for evaluation model (default: 32768)
- `--eval_max_new_tokens`: Maximum new tokens to generate (default: 2048)
- `--eval_max_num_seqs`: Maximum concurrent sequences for evaluation (default: 200)
- `--eval_template`: Path to chat template file (default: None)
- `--eval_temperature`: Sampling temperature (0.0 for deterministic) (default: 0.0)
- `--eval_devices`: Comma-separated GPU devices for evaluation (default: "0")
- `--eval_host`: Host address for evaluation service (default: "127.0.0.1")
- `--eval_port`: Port for evaluation service (default: 8000)
- `--eval_api_key`: API key for evaluation service (default: "EMPTY")

### Optional Parameters - Judge Model
- `--judge_max_model_length`: Maximum context length for judge model (default: 8192)
- `--judge_max_num_seqs`: Maximum concurrent sequences for judging (default: 200)
- `--judge_devices`: Comma-separated GPU devices for judge model (default: "0")
- `--judge_host`: Host address for judge service (default: "10.7.91.121")
- `--judge_port`: Port for judge service (default: 8000)
- `--judge_api_key`: API key for judge service (default: "EmpTY")

### Optional Parameters - Backend
- `--eval_backend`: Evaluation backend ("VLMEvalKit" or "Native", default: "VLMEvalKit")
- `--deploy_backend`: Model serving backend ("vLLM" or "LMDeploy", default: "vLLM")
