def print_args(args):
    """优雅地打印命令行参数"""
    
    print("-" * 20, "args", "-" * 20)
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("-" * 18, "args end", "-" * 18)
