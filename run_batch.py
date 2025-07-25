import argparse
from vllm_utils import start_vllm_server, wait_server, stop_server
from evaluation_runner import run_evaluation


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--conda_env', required=True)
    parser.add_argument('--work_dir', required=True)
    parser.add_argument('--reuse', type=bool, default=True)
    
    parser.add_argument('--datasets', nargs='+', default=[
        'LogicVista', 'MMMU_DEV_VAL', 'MathVista_MINI', 'WeMath', 'MathVision', 'MathVerse_MINI', 'DynaMath',
    ])
    
    parser.add_argument('--judge_model_path', required=True)
    parser.add_argument('--judge_model_name', required=True)
    parser.add_argument('--judge_max_model_length', type=int, default=8192)
    parser.add_argument('--judge_max_num_seqs', type=int, default=200)
    parser.add_argument('--judge_devices', type=str, default='0')
    parser.add_argument('--judge_host', type=str, default='10.7.91.121')
    parser.add_argument('--judge_port', type=int, default=8000)
    parser.add_argument('--judge_api_key', type=str, default='EmpTY')
    
    parser.add_argument('--eval_model_path', required=True, nargs='+')
    parser.add_argument('--eval_model_name', required=True, nargs='+')
    parser.add_argument('--eval_max_model_length', type=int, default=32768)
    parser.add_argument('--eval_max_new_tokens', type=int, default=2048)
    parser.add_argument('--eval_max_num_seqs', type=int, default=200)
    parser.add_argument('--eval_template', type=str, default=None)
    parser.add_argument('--eval_temperature', type=float, default=0.0)
    parser.add_argument('--eval_devices', type=str, default='0')
    parser.add_argument('--eval_host', type=str, default='127.0.0.1')
    parser.add_argument('--eval_port', type=int, default=8000)
    parser.add_argument('--eval_api_key', type=str, default='EMPTY')

    parser.add_argument('--eval_backend', type=str, default='VLMEvalKit', choices=['VLMEvalKit', 'Native'])
    parser.add_argument('--deploy_backend', type=str, default='vllm', choices=['vllm', 'lmdeploy'])

    return parser.parse_args()


def main():
    args = parse_args()

    try:
        for model_path, model_name in zip(args.eval_model_path, args.eval_model_name):
            run_evaluation(args, model_path, model_name)

    finally:
        pass


if __name__ == "__main__":
    main()
