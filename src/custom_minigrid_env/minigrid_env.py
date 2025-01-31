import numpy as np
import gymnasium as gym
import warnings
from minigrid.wrappers import *
from scipy import sparse

warnings.filterwarnings("ignore", category=DeprecationWarning)

class ImageObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.num_directions =  env.observation_space['direction'].n
        image = env.observation_space['image']
        new_shape = (image.shape[0] * image.shape[1] * image.shape[2] + self.num_directions,)
        
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=new_shape,
            dtype='uint8'
        )

    def observation(self, obs):
        # One-hot encode the direction
        direction_one_hot = float(obs['direction']) / float(self.num_directions)
        
        # Flatten the image part of the observation
        flattened_image = obs['image'].flatten()
        
        # Concatenate the flattened image and one-hot encoded direction
        combined_obs = np.concatenate([flattened_image, [direction_one_hot]])
        
        return combined_obs

class StateObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.dir_dim = float(env.observation_space['direction'].n)
        # self.pos_dim = env.width + env.height
        # new_shape = (self.dir_dim + self.pos_dim, )
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(3,),
            dtype='float32'
        )
    
    def observation(self, obs):
        
        # Use continuous observation space
        normalized_agent_pos = np.array(self.agent_pos) / self.size * 2.0 - 1
        normalized_direction = float(obs['direction']) / self.dir_dim * 2.0 - 1
        
        return np.array([*normalized_agent_pos, normalized_direction])


class OneHotActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self._env = env
        self.action_space = gym.spaces.Box(
            low=0, high=1, shape=(3,), dtype='float32'
        )

    def action(self, action):
        # Convert one-hot encoded action to integer action
        action = np.argmax(action)
        return action
    
    def reset(self):
        return self._env.reset()[0]

def make_env(cfg):
    """
    Make MiniGrid environment for TD-MPC experiments.
    """
    env = gym.make(str(cfg.task), agent_poses=cfg.agent_poses, goal_poses=cfg.goal_poses,
                   task_idx=cfg.task_idx, max_episode_steps=cfg.episode_length, size=cfg.size, 
                   render_mode=cfg.render_mode, dense_reward=cfg.dense_reward, 
                   goal_radius=cfg.goal_radius, put_blocks=cfg.put_blocks)
    if cfg.modality == "pixels":
        # NOTE: Image observation not tested yet
        env = ImageObservationWrapper(env)
    elif cfg.modality == "state" or cfg.modality == "dqn":
        env = StateObservationWrapper(env)
        if cfg.modality == "dqn":
            return env
    else:
        raise ValueError(f"Invalid modality: {cfg.modality}")
    env = OneHotActionWrapper(env)

    # Convenience
    cfg.obs_shape = tuple(int(x) for x in env.observation_space.shape)
    cfg.action_shape = tuple(int(x) for x in env.action_space.shape)
    cfg.action_dim = env.action_space.shape[0]

    return env
