import os
import yaml

def get_cfg(cfg_filename):
    """获取配置"""
    try:
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Loader
    # 获取当前文件所在目录
    curPath = os.path.dirname(os.path.realpath(__file__))
    # curPath = os.path.join(curPath, "../config", cfg_filename)
    # 获取yaml文件路径
    yamlPath = os.path.join(curPath, "../config", cfg_filename)

    with open(yamlPath, encoding="utf-8") as f:
        cfg = yaml.load(f, Loader)
    
    return cfg


def print_yml_cfg(cfg):
    """打印从yml文件加载的配置"""

    print("-" * 20, "yml cfg", "-" * 20)
    for k, v in cfg.items():
        print(f"{k}: {v}")
    print("-" * 18, "yml cfg end", "-" * 18)


if __name__ == '__main__':
    cfg = get_cfg("cbs_refl.yml")["cifar-10"]
    print_yml_cfg(cfg)