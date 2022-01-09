import os
from unittest.mock import patch

from dimagi_clockify_cli.config import get_config_dir


def test_get_config_dir_default():
    with patch.dict(os.environ, {'HOME': '/home/nicola'}, clear=True):
        config_dir = get_config_dir()
    assert config_dir == '/home/nicola/.config/dimagi-clockify-cli'


def test_get_config_dir_alt():
    with patch.dict(os.environ, {'DCL_CONFIG_DIR': '/etc/dcl'}, clear=True):
        config_dir = get_config_dir()
    assert config_dir == '/etc/dcl'
