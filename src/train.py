from copy import deepcopy
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
import torch
import numpy as np
import gym
gym.logger.set_level(40)
import time
import random
from pathlib import Path
from cfg import parse_cfg
from src.env import make_env
from src.algorithm.drago import DRAGO
from algorithm.helper import Episode, ReplayBuffer, PseudoCounts, ContPseudoCounts
import logger
import torch
import time

torch.backends.cudnn.benchmark = True
__CONFIG__, __LOGS__ = 'cfgs', 'logs'


def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def evaluate(cfg, buffer, env, agent, pre_rollout_agent, pseudo_counts, step, env_step, video):
	"""Evaluate a trained agent and optionally save a video."""
	episode_rewards = []
	for i in range(cfg.num_episodes):
		if video: video.init(env, enabled=(i==0))
		obs, _, _, ep_reward, t = env.reset(), False, False, 0, 0
		if pre_rollout_agent is not None:
			for pre_t in range(cfg.pre_rollout_steps):
				action = pre_rollout_agent.learner.plan(buffer, pseudo_counts, obs, eval_mode=True, step=step, t0=pre_t==0, key='learner')
				obs, _, _, _, _ = env.step(action.cpu().numpy())
				if video: video.record(env)
		while t < cfg.episode_length:
			action = agent.learner.plan(buffer, pseudo_counts, obs, eval_mode=True, step=step, t0=t==0, key='learner')
			next_obs, reward, _, _, info = env.step(action.cpu().numpy())
			obs = next_obs
			ep_reward += reward
			if video: video.record(env)
			t += 1
		assert t == cfg.episode_length
		episode_rewards.append(ep_reward)
		if video: video.save(env_step)
	total_reward = np.nanmean(episode_rewards)
	return total_reward


def train(cfg):
	"""Training script for DRAGO. Requires a CUDA-enabled device."""
	assert torch.cuda.is_available()
	set_seed(cfg.seed)
	fp = cfg.ckpt
	start_idx = cfg.task_idx
	pre_rollout_agent = None
	cfg.time_limit = cfg.episode_length * cfg.action_repeat
	
	for task_idx in range(start_idx, len(cfg.tasks)):
		work_dir = Path().cwd() / __LOGS__ / cfg.task / cfg.exp_name\
			/ f"task_idx_{task_idx}" / f"seed_{cfg.seed}"
		if cfg.env == 'metaworld':
			cfg.coffee_tasks = cfg.coffee_tasks_list[task_idx]
		cfg.task_idx = task_idx
		cfg.horizon = cfg.horizons[task_idx]
		cfg.task = cfg.tasks[task_idx]
		cfg.ckpt = fp
		rollout_steps = cfg.episode_length

		# If use reviewer, double the logged rollout steps
		if cfg.ckpt is not None and cfg.use_reviewer:
			rollout_steps = cfg.episode_length * 2

		env, learner_buffer, reviewer_buffer = make_env(cfg), ReplayBuffer(cfg), ReplayBuffer(cfg)
		
		cfg.obs_dim = cfg.obs_shape[0]
		if not cfg.use_encoder:
			cfg.latent_dim = cfg.obs_shape[0]

		if cfg.pre_rollout_ckpt is not None:
			pre_rollout_cfg = deepcopy(cfg)
			pre_rollout_cfg.ckpt = cfg.pre_rollout_ckpt
			pre_rollout_cfg.load_policy = True
			pre_rollout_agent = DRAGO(pre_rollout_cfg)
			
		agent = DRAGO(cfg)
		pseudo_counts = PseudoCounts(cfg) if cfg.env=='minigrid' else\
			  ContPseudoCounts(cfg.latent_dim)
		
		# Run training
		L = logger.Logger(work_dir, cfg)

		episode_idx, start_time = 0, time.time()
		for step in range(0, cfg.train_steps+rollout_steps, rollout_steps):
			# Collect trajectory
			obs = env.reset()
			# Pre-rollout in finetuning
			if cfg.pre_rollout_ckpt is not None:
				for t in range(cfg.pre_rollout_steps):
					action = pre_rollout_agent.learner.plan(learner_buffer, pseudo_counts,\
								 obs, eval_mode=True, step=step, t0=t==0, key='learner')
					obs, _, _, _, _ = env.step(action.cpu().numpy())

			# learner rollout:
			episode = Episode(cfg, obs)
			while len(episode) < cfg.episode_length:
				action = agent.learner.plan(learner_buffer, pseudo_counts,\
							  obs, step=step, t0=episode.first, key='learner')
				obs, reward, terminated, truncated, info = env.step(action.cpu().numpy())
				episode += (obs, action, reward, terminated, truncated)
			assert len(episode) == cfg.episode_length
			learner_buffer += episode
			
			# reviewer rollout
			if cfg.ckpt is not None and cfg.use_reviewer:
				reviewer_episode_reward = 0
				episode_old_reward = 0
				episode_cost = 0
				obs = env.reset()
				review_episode = Episode(cfg, obs)
				encoded_obs = agent.reviewer.model._encoder(torch.tensor([obs],\
				 dtype=torch.float32).to(cfg.device))
				pseudo_counts.update(encoded_obs)
				
				while len(review_episode) < cfg.episode_length:
					action = agent.reviewer.plan(reviewer_buffer, pseudo_counts,\
								  obs, step=step, t0=review_episode.first, key='reviewer')
					next_obs, _, terminated, truncated, _ = env.step(action.cpu().numpy())

					# recompute reviewer reward
					reward, old_wm_reward, cost = agent.reviewer_reward(
						torch.tensor([obs], dtype=torch.float32).to(cfg.device),  
						torch.tensor([next_obs], dtype=torch.float32).to(cfg.device),
						action.unsqueeze(0).to(cfg.device))
					obs = next_obs

					# update pseudo counts
					encoded_obs = agent.reviewer.model._encoder(
						torch.tensor([obs], dtype=torch.float32).to(cfg.device)
						)
					pseudo_counts.update(encoded_obs)

					review_episode += (obs, action, reward, terminated, truncated)
					reviewer_episode_reward += reward
					episode_old_reward += old_wm_reward
					episode_cost -= cost

				assert len(review_episode) == cfg.episode_length
				reviewer_buffer += review_episode

			# Update model
			train_metrics = {}
			if step >= cfg.seed_steps:
				for i in range(cfg.episode_length):
					update_reviewer = cfg.ckpt is not None and cfg.use_reviewer
					train_metrics.update(agent.update(learner_buffer, reviewer_buffer,  
										step+i, update_reviewer=update_reviewer))
			# Log training episode
			episode_idx += 1
			env_step = int(step*cfg.action_repeat)
			common_metrics = {
				'episode': episode_idx,
				'step': step,
				'env_step': env_step,
				'total_time': time.time() - start_time,
				'episode_reward': episode.cumulative_reward,
				'reviewer_episode_reward': reviewer_episode_reward if cfg.use_reviewer and cfg.ckpt is not None else None,
				'old_model_episode_reward': episode_old_reward if cfg.use_reviewer and cfg.ckpt is not None else None,
				'learner_model_episode_cost': episode_cost if cfg.use_reviewer and cfg.ckpt is not None else None}
			
			train_metrics.update(common_metrics)
			L.log(train_metrics, category='train')

			# Save agent periodically
			if env_step % cfg.save_freq == 0:
				_model_dir = work_dir / 'models'
				if not os.path.exists(_model_dir):
					os.makedirs(_model_dir)
				if cfg.save_model:
					fp = f'{_model_dir}/model_{env_step}.pt'
					torch.save(agent.state_dict(), fp)

			# Evaluate periodically
			if env_step % cfg.eval_freq == 0:
				common_metrics['episode_reward'] = evaluate(
					cfg, reviewer_buffer, env, agent, pre_rollout_agent, pseudo_counts, 
					step, env_step, L.video,)
				L.log(common_metrics, category='eval')

		L.finish(agent)
		print('Training completed successfully')


if __name__ == '__main__':
	
	train(parse_cfg(Path().cwd() / __CONFIG__))