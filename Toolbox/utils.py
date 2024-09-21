import os
import platform


def get_shallow_demo(args, executor_relative_path: str) -> list:
    demo = [
        os.path.join(args.matlab_home, executor_relative_path), # matlab executor
        '-nojvm', '-nodesktop', '-r', # matlab param
        f"demo_{args.model_name.upper()}('{str(args.code_len)}', '{args.dataset_name.lower()}'); quit;" # target method
    ]

    return demo


def get_deep_demo(args, executor_relative_path: str) -> list:
    demo = [
        os.path.join(args.conda_home, args.executor_env_name, executor_relative_path), # python executor
        f'demo_{args.model_name.upper()}.py', # target method
        '--dataname', args.dataset_name, '--bits', str(args.code_len), # bash param
    ]

    return demo


def get_demo(args) -> list:
    executor_relative_path = get_executor_relative_path(args.learning_principle)
    
    if args.learning_principle == 'shallow':
        demo = get_shallow_demo(args, executor_relative_path)
    elif args.learning_principle == 'deep':
        demo = get_deep_demo(args, executor_relative_path)
    else:
        raise ValueError('learning_principle can only specify shallow or deep, please check!')

    return demo


def get_executor_relative_path(learning_principle: str):
    if learning_principle == 'shallow':
        executor_platform = 'matlab'
    elif learning_principle == 'deep':
        executor_platform = 'python'
    else:
        raise ValueError('learning_principle can only specify shallow or deep, please check!')
    
    if platform.system() == 'Linux':
        executor_relative_path = os.path.join('bin', executor_platform)
    elif platform.system() == 'Darwin':
        executor_relative_path = os.path.join('bin', executor_platform)
    elif platform.system() == 'Windows':
        if executor_platform == 'python':
            executor_relative_path = os.path.join(f'{executor_platform}.exe')
        elif executor_platform == 'matlab':
            executor_relative_path = os.path.join('bin', f'{executor_platform}.exe')
    else:
        executor_relative_path = os.path.join('bin', executor_platform)

    return executor_relative_path


def get_demo_cwd(args) -> str:
    if args.learning_principle == 'shallow':
        cwd = os.path.join(args.model_name.upper(), 'code')
    elif args.learning_principle == 'deep':
        cwd = os.path.join(args.model_name.upper())
    else:
        raise ValueError('learning_principle can only specify shallow or deep, please check!')
    
    return cwd


def get_runtime_env():
    # copy
    env = os.environ.copy()
    
    # set runtime env
    env['CUDA_VISIBLE_DEVICES'] = '3'
    
    return env
