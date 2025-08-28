import yaml
import os

def load_config(config_path=None):
    # Allow for default location or user-specified
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

settings = load_config()