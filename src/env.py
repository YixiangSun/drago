from src.custom_dmcontrol_env.dmcontrol_env import make_env as DMMakeEnv
from src.custom_minigrid_env.minigrid_env import make_env as MGMakeEnv

def make_env(cfg):
    if cfg.env == 'dmcontrol':
        return DMMakeEnv(cfg)
    elif cfg.env == 'minigrid':
        return MGMakeEnv(cfg)
    else:
        raise ValueError(f'Unknown env: {cfg.env}')