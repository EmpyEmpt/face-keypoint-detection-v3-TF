import yaml

import wandb

from src.train import train
from src.optimization import optimize


def main():
    wandb.login(key=open('secrets/wandb_key.txt', 'r').read(), relogin=True)

    config = yaml.safe_load(open('config.yaml', 'r'))

    with wandb.init(project=config['wandb']['project'],
                    name=config['wandb']['name'],
                    config=config):
        model = train(config)

    with wandb.init(project=config['wandb']['project'],
                    name='Optimization',
                    config=config):
        optimize(config, model)


if __name__ == '__main__':
    main()
