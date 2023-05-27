import os
import yaml
import argparse
from mergedeep import merge


def merge_configs(*cfgs):
    return merge(dict(cfgs[0]), *cfgs[1:])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/default.yaml')

    args = parser.parse_args()
    default_config_path = os.path.join(os.path.dirname(args.config), 'default.yaml')

    with open(default_config_path) as f:
        default_config = yaml.safe_load(f)

    with open(args.config) as f:
        input_config = yaml.safe_load(f)

    config = merge_configs(default_config, input_config)

    print(f'INPUT PARAMETERS: {config}')

    return config
