from evalscope.run import run_task
from evalscope.config import TaskConfig
from vllm_utils import start_vllm_server, start_lmdeploy_server, wait_server, stop_server
import os


def run_evaluation(args, model_path, model_name):
    devices = [int(d) for d in args.eval_devices.split(',')]

    if args.deploy_backend == 'vllm':
        eval_server = start_vllm_server(
            conda_env_path=args.conda_env,
            model_path=model_path,
            served_model_name=model_name,
            devices=devices,
            tensor_parallel_size=len(devices),
            max_model_len=args.eval_max_model_length,
            max_num_seqs=args.eval_max_num_seqs,
            port=args.eval_port,
            chat_template=args.eval_template,
        )
    elif args.deploy_backend == 'lmdeploy':
        eval_server = start_lmdeploy_server(
            conda_env_path=args.conda_env,
            model_path=model_path,
            served_model_name=model_name,
            devices=devices,
            tensor_parallel_size=len(devices),
            max_model_len=args.eval_max_model_length,
            max_num_seqs=args.eval_max_num_seqs,
            port=args.eval_port,
            chat_template=args.eval_template,
        )

    try:
        wait_server(port=args.eval_port, timeout=600)

        if args.eval_backend == 'VLMEvalKit':
            task_cfg = TaskConfig(
                work_dir=args.work_dir,
                use_cache=args.work_dir,
                eval_backend=args.eval_backend,
                eval_config={
                    "data": args.datasets,
                    "mode": "all",
                    "reuse": args.reuse,
                    "nproc": args.eval_max_num_seqs,
                    "model": [{
                        "type": model_name,
                        "name": "CustomAPIModel",
                        "api_base": f"http://{args.eval_host}:{args.eval_port}/v1/chat/completions",
                        "key": args.eval_api_key,
                        "max_tokens": args.eval_max_new_tokens,
                        "temperature": args.eval_temperature
                    }],
                    "OPENAI_API_BASE": args.judge_api_url,
                    "OPENAI_API_KEY": args.judge_api_key,
                    "LOCAL_LLM": args.judge_model_name
                }
            )
        elif args.eval_backend == 'Native':
            EVAL_PROMPT_TEMPLATE = (
                "{query} Please reason step by step, and put your final answer within \\boxed{{}}."
            )
            dataset_args = {
                dataset: {"prompt_template": EVAL_PROMPT_TEMPLATE}
                for dataset in args.datasets
            }
            task_cfg = TaskConfig(
                work_dir=args.work_dir,
                use_cache=args.work_dir,
                eval_backend=args.eval_backend,
                eval_type="service",
                datasets=args.datasets,
                dataset_args=dataset_args,
                eval_batch_size=args.eval_max_num_seqs,
                model=model_name,
                model_id=model_name,
                api_url=f"http://{args.eval_host}:{args.eval_port}/v1",
                api_key=args.eval_api_key,
                generation_config={
                    "max_tokens": args.eval_max_new_tokens,
                    "temperature": args.eval_temperature
                },
                judge_strategy="llm",
                judge_worker_num=args.judge_max_num_seqs,
                judge_model_args={
                    "api_url": args.judge_api_url,
                    "api_key": args.judge_api_key,
                    "model_id": args.judge_model_name
                }
            )

        run_task(task_cfg=task_cfg)

    finally:
        stop_server(eval_server, devices)
