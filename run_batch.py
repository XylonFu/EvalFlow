import argparse
from evaluation_runner import run_evaluation


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--conda_env', type=str)
    parser.add_argument('--work_dir', required=True)
    parser.add_argument('--reuse', type=str2bool, default=True)

    parser.add_argument('--datasets', nargs='+', default=[
        'LogicVista', 'MathVista_MINI', 'WeMath', 'MathVision', 'MathVerse_MINI', 'DynaMath',
    ])
    
    parser.add_argument('--judge_model_name', type=str)
    parser.add_argument('--judge_max_num_seqs', type=int, default=200)
    parser.add_argument('--judge_api_url', type=str)
    parser.add_argument('--judge_api_key', type=str)
    
    parser.add_argument('--eval_model_path', nargs='+', default=None)
    parser.add_argument('--eval_model_name', required=True, nargs='+')
    parser.add_argument('--eval_max_model_length', type=int, default=32768)
    parser.add_argument('--eval_max_new_tokens', type=int, default=2048)
    parser.add_argument('--eval_max_num_seqs', type=int, default=200)
    parser.add_argument('--eval_template', type=str, default=None)
    parser.add_argument('--eval_system', type=str, default=None)
    parser.add_argument('--eval_temperature', type=float, default=0.0)
    parser.add_argument('--eval_devices', type=str, default='0')
    parser.add_argument('--eval_host', type=str, default='127.0.0.1')
    parser.add_argument('--eval_port', type=int, default=8000)
    parser.add_argument('--eval_api_key', type=str, default='EMPTY')

    parser.add_argument('--eval_backend', type=str, default='VLMEvalKit', choices=['VLMEvalKit', 'Native'])
    parser.add_argument('--vlmevalkit_mode', type=str, default='all', choices=['all', 'infer'])

    parser.add_argument('--deploy_backend', type=str, default='vllm', choices=['vllm', 'lmdeploy', 'swift', 'remote'])
    parser.add_argument('--swift_infer_backend', type=str, default='pytorch', choices=['pytorch', 'vllm', 'lmdeploy'])
    parser.add_argument('--remote_api_url', nargs='+', default=None)

    args = parser.parse_args()
    
    if args.deploy_backend == 'remote':
        if args.remote_api_url is None or len(args.remote_api_url) != len(args.eval_model_name):
            parser.error("When using remote backend, --remote_api_url must be provided and have the same number of URLs as --eval_model_name")
        if args.eval_model_path is not None:
            print("Warning: --eval_model_path is ignored when using remote backend")
    else:
        if not args.conda_env:
            parser.error("--conda_env is required when not using remote backend")
        if args.eval_model_path is None or len(args.eval_model_path) != len(args.eval_model_name):
            parser.error("--eval_model_path must be provided and match the number of --eval_model_name when not using remote backend")
    
    if args.eval_backend == 'VLMEvalKit' and args.vlmevalkit_mode != 'infer':
        if not args.judge_model_name or not args.judge_api_url or not args.judge_api_key:
            parser.error("When eval_backend is VLMEvalKit and vlmevalkit_mode is not 'infer', judge_model_name, judge_api_url, and judge_api_key are required.")
    elif args.eval_backend == 'Native':
        if not args.judge_model_name or not args.judge_api_url or not args.judge_api_key:
            parser.error("When eval_backend is Native, judge_model_name, judge_api_url, and judge_api_key are required.")

    return args


def main():
    args = parse_args()

    try:
        if args.deploy_backend == 'remote':
            identifiers = args.remote_api_url
        else:
            identifiers = args.eval_model_path

        for identifier, model_name in zip(identifiers, args.eval_model_name):
            run_evaluation(args, identifier, model_name)

    finally:
        pass


if __name__ == "__main__":
    main()
