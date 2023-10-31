import yaml

def parse_yaml(file_path):
    with open(file_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
            return config
        except yaml.YAMLError as exc:
            print(exc)
            return None

if __name__ == "__main__":
    # Example usage:
    config = parse_yaml("config.yaml")
    if config:
        print(config)