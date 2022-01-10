import os
from typing import Dict, List

from pydantic.main import BaseModel
from yaml import load
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


class Bucket(BaseModel):
    description: str
    project: str
    task: str
    tags: List[str]


class Config(BaseModel):
    base_url: str
    api_key: str
    buckets: Dict[str, Bucket]


def get_config() -> Config:
    config_dir = get_config_dir()
    config_filename = os.path.join(config_dir, 'config.yaml')
    with open(config_filename, 'r') as config_file:
        config_dict = load(config_file, Loader=Loader)
    config = Config.parse_obj(config_dict)
    if 'stop' in config.buckets:
        raise ValueError(
            f'"stop" is reserved, and cannot be used as a bucket name'
        )
    return config


def get_config_dir() -> str:
    config_dir = os.environ.get('DCL_CONFIG_DIR')
    if config_dir is None:
        home = os.environ['HOME']
        config_dir = os.path.join(home, '.config', 'dimagi-clockify-cli')
    return config_dir
