import sys
import yaml
from explaignn.library.utils import get_out_path, get_gnn_string

def get_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    config_path = sys.argv[1]
    config = get_config(config_path)
    try:
        i = sys.argv[2]
        print(get_out_path(config,i))
    except IndexError:
        print(get_gnn_string(config))
    
