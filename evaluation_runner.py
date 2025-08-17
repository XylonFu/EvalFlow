from evalscope.run import run_task
from evalscope.config import TaskConfig
from vllm_utils import start_vllm_server, start_lmdeploy_server, start_swift_server, wait_server, stop_server
import os


def run_evaluation(args, model_identifier, model_name):
    devices = None
    eval_server = None

    if args.deploy_backend != 'remote':
        devices = [int(d) for d in args.eval_devices.split(',')]
        
        if args.deploy_backend == 'vllm':
            eval_server = start_vllm_server(
                conda_env_path=args.conda_env,
                model_path=model_identifier,
                served_model_name=model_name,
                devices=devices,
                tensor_parallel_size=len(devices),
                max_model_len=args.eval_max_model_length,
                max_num_seqs=args.eval_max_num_seqs,
                host=args.eval_host,
                port=args.eval_port,
                api_key=args.eval_api_key,
                chat_template=args.eval_template,
            )
        elif args.deploy_backend == 'lmdeploy':
            eval_server = start_lmdeploy_server(
                conda_env_path=args.conda_env,
                model_path=model_identifier,
                served_model_name=model_name,
                devices=devices,
                tensor_parallel_size=len(devices),
                max_model_len=args.eval_max_model_length,
                max_num_seqs=args.eval_max_num_seqs,
                host=args.eval_host,
                port=args.eval_port,
                api_key=args.eval_api_key,
                chat_template=args.eval_template,
            )
        elif args.deploy_backend == 'swift':
            eval_server = start_swift_server(
                conda_env_path=args.conda_env,
                model_path=model_identifier,
                served_model_name=model_name,
                devices=devices,
                tensor_parallel_size=len(devices),
                max_model_len=args.eval_max_model_length,
                max_num_seqs=args.eval_max_num_seqs,
                host=args.eval_host,
                port=args.eval_port,
                api_key=args.eval_api_key,
                infer_backend=args.swift_infer_backend,
                template=args.eval_template,
                system=args.eval_system
            )
        
        wait_server(port=args.eval_port, timeout=600)

    if args.deploy_backend == 'remote':
        api_url = model_identifier
        api_base = f"{api_url}/v1/chat/completions"
    else:
        api_url = f"http://{args.eval_host}:{args.eval_port}/v1"
        api_base = f"http://{args.eval_host}:{args.eval_port}/v1/chat/completions"

    try:
        if args.eval_backend == 'VLMEvalKit':
            task_cfg = TaskConfig(
                work_dir=args.work_dir,
                use_cache=args.work_dir,
                eval_backend=args.eval_backend,
                eval_config={
                    "data": args.datasets,
                    "mode": args.vlmevalkit_mode,
                    "reuse": args.reuse,
                    "nproc": args.eval_max_num_seqs,
                    "model": [{
                        "type": model_name,
                        "name": "CustomAPIModel",
                        "api_base": api_base,
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
                api_url=api_url,
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
        if args.deploy_backend != 'remote' and eval_server is not None:
            stop_server(eval_server, devices)
