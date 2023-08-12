import yaml


class ConfigManager:
    """
    通过读取yaml文件读取所有配置
    """

    def __init__(self, config_path):
        with open(config_path, 'r') as stream:
            try:
                self.config = yaml.safe_load()
            except yaml.YAMLError as e:
                print(e)
                self.config = {}

    def get(self, *args):
        config = dict.fromkeys(args)
        for arg in args:
            config[arg] = self.config[arg]

        return config
