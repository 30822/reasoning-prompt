import yaml
import sys
import os
from pathlib import Path


def _get_project_root():
    """return the project root"""
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    return project_root


def _get_config_path():
    """return the absolute path of the config file"""
    return _get_project_root() / "conf.d" / "conf.yaml"


def get_openai_key():
    try:
        config_path = _get_config_path()
        with open(config_path) as f:
            config = yaml.safe_load(f)
            return config["openai"]["key"]
    except FileNotFoundError:
        raise FileNotFoundError("conf.d/conf.yaml file not found. Please create one based on conf.example")
    except KeyError:
        raise KeyError("OpenAI key not found in conf.yaml. Please check configuration format")


def get_anthropic_key():
    try:
        config_path = _get_config_path()
        with open(config_path) as f:
            config = yaml.safe_load(f)
            return config["anthropic"]["key"]
    except FileNotFoundError:
        raise FileNotFoundError("conf.d/conf.yaml file not found. Please create one based on conf.example")
    except KeyError:
        raise KeyError("Anthropic key not found in conf.yaml. Please check configuration format")


def get_openrouter_key():
    try:
        config_path = _get_config_path()
        with open(config_path) as f:
            config = yaml.safe_load(f)
            return config["openrouter"]["key"]
    except FileNotFoundError:
        raise FileNotFoundError("conf.d/conf.yaml file not found. Please create one based on conf.example")
    except KeyError:
        raise KeyError("OpenRouter key not found in conf.yaml. Please check configuration format")


def load_dataset(data_name):
    dataset = {'train':[], 'test':[], 'valid':[]}
    project_root = _get_project_root()
    for key in dataset:
        try:
            data_path = project_root / "resources" / "data" / f"{data_name}-{key}.txt"
            with open(data_path, 'r') as infile:
                for episode_idx, line in enumerate(infile):
                    data_item = eval(line.strip('\n'))
                    data_item['episode_idx'] = episode_idx
                    dataset[key].append(data_item)
        except FileNotFoundError:
            continue
    return dataset