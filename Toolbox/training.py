import os
import time
import argparse
import subprocess
import yaml

from utils import get_demo_cwd, get_demo, get_runtime_env


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default=os.path.join(os.path.dirname(__file__), 'config.yaml'), type=str)
    parser.add_argument("--category", default='ucmh', type=str, help='scmh/ucmh/cmrss')
    parser.add_argument("--dataset_name", default='flickr', type=str, help='coco/nuswide/flickr')
    parser.add_argument("--model_name", default='djsrh', type=str, help='djsrh/jdsh/dgcpn/cirh')
    parser.add_argument("--code_len", default=16, type=int, help='16/32/64/128')
    args = parser.parse_args()
    
    # merge config
    assert os.path.isfile(args.cfg), "cfg file: {} not found".format(args.cfg)
    with open(args.cfg, 'r') as yaml_file:
        cfg = yaml.safe_load(yaml_file)
    category_cfg = cfg.get(args.category, 'scmh')
    model_cfg = category_cfg.get(args.model_name.lower(), None)
    if model_cfg is None:
        raise ModuleNotFoundError(f'{args.model_name.lower()} may not be in the {args.category} category, please check!')
    args_dict = vars(args)
    args_dict.update(cfg)
    args_dict.update(category_cfg)
    args_dict.update(model_cfg)
    args = argparse.Namespace(**args_dict)
    
    return args


if __name__ == '__main__':
    args = parse_args()

    cwd = get_demo_cwd(args=args)
    demo = get_demo(args=args)
    env = get_runtime_env()
    process = subprocess.Popen(args=demo, env=env, cwd=cwd)

    # wait for the end of training
    while True:
        if process.poll() is not None:
            print('Training have finished.')
            break
        time.sleep(1)