import config

import json


def load_config(config_path='examples/config/coco.json'):
    assert isinstance(config_path, str)
    # only support json format at present
    assert config_path.endswith("json")
    with open(config_path, "r") as f:
        obj = json.load(f)
    for k,v in obj.iteritems():
        config.__dict__[k] = v

