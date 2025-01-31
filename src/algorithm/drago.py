import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
import algorithm.helper as h


class TOLD(nn.Module):
	"""Task-Oriented Latent Dynamics (TOLD) model used in TD-MPC."""
	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		self._encoder = h.enc(self.cfg)
		act_fn = 'crelu' if self.cfg.use_crelu else nn.ELU()
		self._dynamics = h.mlp(self.cfg.latent_dim+self.cfg.action_dim, 
						 self.cfg.mlp_dim, self.cfg.latent_dim, act_fn=act_fn)
		self._reward = h.mlp(self.cfg.latent_dim+self.cfg.action_dim, self.cfg.mlp_dim, 1)
		self._pi = h.mlp(self.cfg.latent_dim, self.cfg.mlp_dim, self.cfg.action_dim)
		self._Q1, self._Q2 = h.q(self.cfg), h.q(self.cfg)
		self.apply(h.orthogonal_init)
		for m in [self._reward, self._Q1, self._Q2]:
			m[-1].weight.data.fill_(0)
			m[-1].bias.data.fill_(0)


	def track_q_grad(self, enable=True):
		"""Utility function. Enables/disables gradient tracking of Q-networks."""
		for m in [self._Q1, self._Q2]:
			h.set_requires_grad(m, enable)

	def h(self, obs):
		"""Encodes an observation into its latent representation (h)."""
		return self._encoder(obs)

	def next(self, z, a):
		"""Predicts next latent state (d) and single-step reward (R)."""
		x = torch.cat([z, a], dim=-1)
		x_p = x.detach()

		next_z = self._dynamics(x).clamp(-100, 100)
		if self.cfg.env == 'minigrid' and self.cfg.modality == 'state':
			next_z = torch.tanh(next_z) ## normalize the state prediction
		return next_z, self._reward(x_p)

	def pi(self, z, std=0, eval_mode=False):
		"""Samples an action from the learned policy (pi)."""
		if self.cfg.env == 'dmcontrol' or self.cfg.env == 'metaworld':
			mu = torch.tanh(self._pi(z))
			if std > 0:
				std = torch.ones_like(mu) * std
				out = h.TruncatedNormal(mu, std).sample(clip=0.3) # avoid clip
				return out
			return mu
		else:
			# use gumbel softmax for minigrid to generate discrete actions
			return h.gumbel_softmax(self._pi(z), deterministic=eval_mode)


	def Q(self, z, a):
		"""Predict state-action value (Q)."""
		z = z.detach()
		x = torch.cat([z, a], dim=-1)
		return self._Q1(x), self._Q2(x)


class TDMPC():
	"""Implementation of TD-MPC learning + inference."""
	def __init__(self, cfg):
		self.cfg = cfg
		self.device = torch.device('cuda')
		self.std = h.linear_schedule(cfg.std_schedule, 0)
		self.model = TOLD(cfg).cuda()
		self.model_target = deepcopy(self.model)
		self.optim = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
		self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.cfg.lr)
		self.aug = h.RandomShiftsAug(cfg)
		self.model.eval()
		self.model_target.eval()

	def load(self, d):
		"""Load model from checkpoint."""
		self.model.load_state_dict(d['model'])
		self.model_target.load_state_dict(d['model_target'])

	def state_dict(self):
		"""Retrieve state dict of TOLD model, including slow-moving target network."""
		return {'model': self.model.state_dict(),
		  		'model_target': self.model_target.state_dict(),
		  		'dynamics': self.model._dynamics.state_dict(),
				'dynamics_target': self.model_target._dynamics.state_dict(),
				'encoder': self.model._encoder.state_dict(),
				'encoder_target': self.model_target._encoder.state_dict(),
				'reward': self.model._reward.state_dict(),}

	def save(self, fp):
		"""Save state dict of TOLD model to filepath."""
		torch.save(self.state_dict(), fp)

	@torch.no_grad()
	def estimate_value(self, z, actions, horizon, pseudo_counts=None, eval_mode=False, key='learner'):
		"""Estimate value of a trajectory starting at latent state z and executing given actions."""
		G, discount = 0, 1
		for t in range(horizon):
			z, reward = self.model.next(z, actions[t])
			if key == 'reviewer':
				discounted_reward = reward * torch.tensor(pseudo_counts.get_intrinsic_rewards(z))
				discounted_reward[torch.where(reward < 0)] = reward[torch.where(reward < 0)]
				reward = discounted_reward
			G += discount * reward
			discount *= self.cfg.discount
		G += int(self.cfg.use_q) * discount * torch.min(
			*self.model.Q(z, self.model.pi(z, self.cfg.min_std, eval_mode=eval_mode)))
		return G

	def sample_from_N(self, mean, n):
		k_int = torch.multinomial(mean, n, replacement=True)
		k_onehot = torch.nn.functional.one_hot(k_int, num_classes=self.cfg.action_dim).to(self.cfg.device)
		return k_onehot
	
	def get_reward(self, z, goal_pos):
		z = (z + 1) * self.cfg.size // 2
		z = torch.round(z)
		matches = (z[:, :2] == goal_pos.cuda()).all(dim=1).float()
		reward = matches.reshape(-1, 1)
		return reward

	@torch.no_grad()
	def plan(self, replay_buffer, pseudo_counts, obs, eval_mode=False, step=None, t0=True, key='learner'):
		"""
		Plan next action using TD-MPC inference.
		obs: raw input observation.
		eval_mode: uniform sampling and action noise is disabled during evaluation.
		step: current time step. determines e.g. planning horizon.
		t0: whether current step is the first step of an episode.
		"""
		# Seed steps
		if (step < self.cfg.seed_steps or replay_buffer.idx < self.cfg.batch_size) and not eval_mode:
			if self.cfg.env == 'dmcontrol' or self.cfg.env == 'metaworld':
				return torch.empty(self.cfg.action_dim, dtype=torch.float32, device=self.device).uniform_(-1, 1)
			elif self.cfg.env == 'minigrid':
				return torch.nn.functional.one_hot(torch.tensor(
					[np.random.choice(self.cfg.action_dim, p=[0.2, 0.2, 0.6])]),
					    num_classes=self.cfg.action_dim).to(self.cfg.device)[0]
			
		# Sample policy trajectories
		obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
		horizon = int(min(self.cfg.horizon, h.linear_schedule(self.cfg.horizon_schedule, step)))
		num_pi_trajs = int(self.cfg.mixture_coef * self.cfg.num_samples)
		if num_pi_trajs > 0:
			pi_actions = torch.empty(horizon, num_pi_trajs, self.cfg.action_dim, device=self.device)
			z = self.model.h(obs).repeat(num_pi_trajs, 1)
			for t in range(horizon):
				pi_actions[t] = self.model.pi(z, self.cfg.min_std, eval_mode=eval_mode)
				z, _ = self.model.next(z, pi_actions[t])

		# Initialize state and parameters
		z = self.model.h(obs).repeat(self.cfg.num_samples+num_pi_trajs, 1)
		if horizon == 0:
			return self.model.pi(z[0], self.cfg.min_std, eval_mode=eval_mode)
		
		if self.cfg.env == 'dmcontrol' or self.cfg.env == 'metaworld':
			# DMControl: CEM with Gaussian policy
		
			mean = torch.zeros(horizon, self.cfg.action_dim, device=self.device)
			std = 2*torch.ones(horizon, self.cfg.action_dim, device=self.device)
			if not t0 and hasattr(self, '_prev_mean'):
				mean[:-1] = self._prev_mean[1:]

			# Iterate CEM
			for _ in range(self.cfg.iterations):
				actions = torch.clamp(mean.unsqueeze(1) + std.unsqueeze(1) * \
					torch.randn(horizon, self.cfg.num_samples, self.cfg.action_dim, device=std.device), -1, 1)
				if num_pi_trajs > 0:
					actions = torch.cat([actions, pi_actions], dim=1)

				# Compute elite actions
				value = self.estimate_value(z, actions, horizon, pseudo_counts, eval_mode=eval_mode, key=key).nan_to_num_(0)
				elite_idxs = torch.topk(value.squeeze(1), self.cfg.num_elites, dim=0).indices
				elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

				# Update parameters
				max_value = elite_value.max(0)[0]
				score = torch.exp(self.cfg.temperature*(elite_value - max_value))
				score /= score.sum(0)
				_mean = torch.sum(score.unsqueeze(0) * elite_actions, dim=1) / (score.sum(0) + 1e-9)
				_std = torch.sqrt(torch.sum(score.unsqueeze(0) *\
								 (elite_actions - _mean.unsqueeze(1)) ** 2, dim=1) / (score.sum(0) + 1e-9))
				_std = _std.clamp_(self.std, 2)
				mean, std = self.cfg.momentum * mean + (1 - self.cfg.momentum) * _mean, _std
			# Outputs
			score = score.squeeze(1).cpu().numpy()
			actions = elite_actions[:, np.random.choice(np.arange(score.shape[0]), p=score)]
			self._prev_mean = mean
			mean, std = actions[0], _std[0]
			a = mean
			if not eval_mode:
				a += std * torch.randn(self.cfg.action_dim, device=std.device)
			return a
		else:
			# Minigrid: CEM with discrete policy
			mean = torch.ones(horizon, self.cfg.action_dim, device=self.device)
			mean /= len(mean)

			# Iterate CEM
			for _ in range(self.cfg.iterations):
				actions = self.sample_from_N(mean, self.cfg.num_samples)
				if num_pi_trajs > 0:
					actions = torch.cat([actions, pi_actions], dim=1)

				# Compute elite actions
				value = self.estimate_value(z, actions, horizon, pseudo_counts, eval_mode=eval_mode, key=key).nan_to_num_(0)
				elite_idxs = torch.topk(value.squeeze(1), self.cfg.num_elites, dim=0).indices
				elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

				max_value = elite_value.max(0)[0]
				score = torch.exp(self.cfg.temperature * (elite_value - max_value))
				score /= score.sum(0)  # [num_elite, 1]
				new_mean = torch.sum(score.unsqueeze(0) * elite_actions, dim=1) / (score.sum(0) + 1e-9)
				mean = self.cfg.momentum * mean + (1 - self.cfg.momentum) * new_mean
				
			# Outputs
			score = score.squeeze(1).cpu().numpy()
			actions = elite_actions[:, np.random.choice(np.arange(score.shape[0]), p=score)]
			self._prev_mean = mean
			mean = actions[0].cpu().numpy()
			
			if eval_mode:
				a = torch.nn.functional.one_hot(torch.tensor(
					[mean.argmax().item()]), num_classes=self.cfg.action_dim).to(self.cfg.device)[0]
			else:
				a = torch.nn.functional.one_hot(torch.tensor(
					[np.random.choice(len(mean), p=mean)]), num_classes=self.cfg.action_dim).to(self.cfg.device)[0]
			assert len(a) == 3

			return a

	def update_pi(self, zs):
		"""Update policy using a sequence of latent states."""
		self.pi_optim.zero_grad(set_to_none=True)
		self.model.track_q_grad(False)

		# Loss is a weighted sum of Q-values
		pi_loss = 0
		for t,z in enumerate(zs):
			z = z[:self.cfg.batch_size]
			a = self.model.pi(z, self.cfg.min_std)
			Q = torch.min(*self.model.Q(z, a))
			pi_loss += -Q.mean() * (self.cfg.rho ** t)

		pi_loss.backward()
		torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm, error_if_nonfinite=False)
		self.pi_optim.step()
		self.model.track_q_grad(True)
		return pi_loss.item()
	
	def update_wm(self, obs, next_obs, action):
		self.optim.zero_grad(set_to_none=True)
		self.model.train()
		if self.cfg.modality == 'pixel':
			obs = self.aug(obs)
		z = self.model.h(obs)
		if self.cfg.modality == 'pixel':
			next_obs = self.aug(next_obs)
		next_z = self.model.h(next_obs)
		forgetting_loss = 0
		pred = self.model._dynamics(torch.cat([z, action], dim=-1))
		forgetting_loss = torch.mean(h.mse(pred, next_z))
		forgetting_loss.backward()
		self.optim.step()

	@torch.no_grad()
	def _td_target(self, next_obs, reward):
		"""Compute the TD-target from a reward and the observation at the following time step."""
		next_z = self.model.h(next_obs)
		td_target = reward + self.cfg.discount * \
			torch.min(*self.model_target.Q(next_z, self.model.pi(next_z, self.cfg.min_std)))
		return td_target

	def update(self, replay_buffer, obs, next_obses, action, reward, idxs, weights, step, key='learner'):
		"""update method called in DRAGO."""
		self.optim.zero_grad(set_to_none=True)
		self.std = h.linear_schedule(self.cfg.std_schedule, step)
		self.model.train()

		# Representation
		if self.cfg.modality == 'pixel':
			obs = self.aug(obs)
		z = self.model.h(obs)
		zs = [z.detach()]

		mean_score, consistency_loss, reward_loss, value_loss, priority_loss = 0, 0, 0, 0, 0

		for t in range(self.cfg.horizon):

			# Predictions
			Q1, Q2 = self.model.Q(z, action[t])
			z, reward_pred = self.model.next(z, action[t])
			with torch.no_grad():
				if self.cfg.modality == 'pixel':
					next_obs = self.aug(next_obses[t])
				else:
					next_obs = next_obses[t]
				next_z = self.model_target.h(next_obs)
				prediction_error = torch.mean((z - next_z) ** 2, dim=1, keepdim=True)
				familiarity = -torch.log(prediction_error)
				curiosity_reward = prediction_error
				total_reward = reward[t] + self.cfg.curiosity_coef * curiosity_reward +\
					self.cfg.familiarity_coef * familiarity
				td_target = self._td_target(next_obs, total_reward)
			zs.append(z.detach())

			# Losses
			rho = (self.cfg.rho ** t)
			# update the world model together at the reviewer
			if key == 'learner' and self.cfg.use_reviewer and self.cfg.ckpt is not None:
				# do not update the world model of the learner from its role out if there is a reviewer
				consistency_loss += torch.zeros(1, device=self.device)
			else:
				# update the world model together at the reviewer
				consistency_loss += rho * torch.mean(h.mse(z, next_z), dim=1, keepdim=True)
				mean_score += -torch.log(torch.mean(h.mse(z, next_z), dim=1, keepdim=True)).detach()
			# calculate mean wm score for sigmoid threshold
			if not self.cfg.first_step_qr_only or t == 0:
				reward_loss += rho * h.mse(reward_pred, reward[t])
				value_loss += rho * (h.mse(Q1, td_target) + h.mse(Q2, td_target))
			if key == 'reviewer':
				priority_loss += rho * ((reward[t] + 2 + h.l1(Q1, td_target) +\
							  h.l1(Q2, td_target))[:self.cfg.batch_size])
			else:
				priority_loss += rho * (h.l1(Q1, td_target) + h.l1(Q2, td_target))[:self.cfg.batch_size]

		if isinstance(mean_score, torch.Tensor):
			mean_score = mean_score.mean() / self.cfg.horizon

		# optimize model
		total_loss = self.cfg.consistency_coef * consistency_loss.clamp(max=1e4) + \
					 self.cfg.reward_coef * reward_loss.clamp(max=1e4) + \
					 self.cfg.value_coef * value_loss.clamp(max=1e4)
		weighted_loss = total_loss.squeeze(1) * weights
		weighted_loss = weighted_loss.mean()
		weighted_loss.register_hook(lambda grad: grad * (1/self.cfg.horizon))
		weighted_loss.backward()
		grad_norm = torch.nn.utils.clip_grad_norm_(
			self.model.parameters(), self.cfg.grad_clip_norm, error_if_nonfinite=False)
		self.optim.step()
		replay_buffer.update_priorities(idxs[:self.cfg.batch_size], priority_loss.clamp(max=1e4).detach())

		# Update policy + target network
		pi_loss = self.update_pi(zs)
		if step % self.cfg.update_freq == 0:
			h.ema(self.model, self.model_target, self.cfg.tau)

		self.model.eval()
		return {f'{key}_consistency_loss': float(consistency_loss.mean().item()),
				f'{key}_reward_loss': float(reward_loss.mean().item()),
				f'{key}_value_loss': float(value_loss.mean().item()),
				f'{key}_pi_loss': pi_loss,
				f'{key}_total_loss': float(total_loss.mean().item()),
				f'{key}_weighted_loss': float(weighted_loss.mean().item()),
				f'{key}_grad_norm': float(grad_norm),
				f'{key}_mean_prediction_score': float(mean_score)}


class DRAGO():
	"""Implementation of DRAGO model."""
	def __init__(self, cfg):
		self.cfg = cfg
		self.device = torch.device('cuda')
		self.sigmoid_threshold = cfg.sigmoid_threshold
		self.new_sigmoid_threshold = cfg.sigmoid_threshold
		self.learner = TDMPC(cfg)
		self.reviewer = None
		self.vae_encoder, self.vae_decoder = h.vae(cfg)
		self.vae_encoder = self.vae_encoder.to(self.device)
		self.vae_decoder = self.vae_decoder.to(self.device)
		self.vae_optim = torch.optim.Adam(list(self.learner.model._encoder.parameters()) +\
									 list(self.vae_encoder.parameters()) +\
									 list(self.vae_decoder.parameters()), lr=cfg.lr)
		self.old_encoder = deepcopy(self.learner.model._encoder).requires_grad_(False)
		self.old_dynamics = deepcopy(self.learner.model._dynamics).requires_grad_(False)
		self.old_vae_encoder = None
		self.old_vae_decoder = None
		self.load_model()

	def reviewer_reward(self, obs, next_obs, action):
		"""Compute reviewer reward."""
		if self.cfg.modality == 'pixel':
			z = self.aug(obs)
			next_z = self.aug(next_obs)
		z = self.old_encoder(obs)
		next_z = self.old_encoder(next_obs)
		pred = self.old_dynamics(torch.cat([z, action], dim=-1))
		if self.cfg.env == 'minigrid' and self.cfg.modality == 'state':
			pred = torch.tanh(pred)
		with torch.no_grad():
			learner_pred = self.learner.model._dynamics(torch.cat([z, action], dim=-1))
			if self.cfg.env == 'minigrid' and self.cfg.modality == 'state':
				learner_pred = torch.tanh(learner_pred)
		reward = -torch.log(torch.mean(h.mse(pred, next_z), dim=-1, keepdim=True))
		reward = torch.sigmoid((reward - self.sigmoid_threshold) * 4)
		cost = 0
		cost = -torch.log(torch.mean(h.mse(learner_pred, next_z), dim=-1, keepdim=True))
		final_reward = reward - torch.sigmoid((cost - self.sigmoid_threshold)) * self.cfg.cost_coef
		return final_reward, reward, - torch.sigmoid((cost - self.sigmoid_threshold))
	
	def generate_obs_action(self, num):
		"""Generate observations and actions from the old VAE model."""
		assert self.old_vae_decoder is not None and self.old_vae_encoder is not None
		zs = torch.randn(num, self.cfg.vae_enc_dim, device=self.device)
		obs_action = self.old_vae_decoder(zs)
		return obs_action
	
	def update_vae(self, obs_action):
		"""Update VAE model."""
		obs_action = obs_action.to(self.device)
		obs = obs_action[:, :self.cfg.obs_shape[0]]
		action = obs_action[:, self.cfg.obs_shape[0]:]
		self.vae_encoder.train()
		self.vae_decoder.train()
		self.vae_encoder.zero_grad(set_to_none=True)
		self.vae_decoder.zero_grad(set_to_none=True)
		if self.cfg.modality == 'pixel':
			obs = self.aug(obs)
		with torch.no_grad():
			obs = self.learner.model._encoder(obs)
		obs_action = torch.cat([obs, action], dim=-1)
		embedding = self.vae_encoder(obs_action)
		mu = embedding[:, :self.cfg.vae_enc_dim]
		logvar = embedding[:, self.cfg.vae_enc_dim:]
		z = h.reparameterize(mu, logvar)
		recon = self.vae_decoder(z)
		recon_loss = h.mse(recon, obs_action)
		kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
		loss = self.cfg.recon_coef * recon_loss + self.cfg.kl_coef * kl_loss
		loss = loss.mean()
		loss.backward()
		grad_norm = torch.nn.utils.clip_grad_norm_(list(self.vae_encoder.parameters()) +\
		 list(self.vae_decoder.parameters()), self.cfg.grad_clip_norm, error_if_nonfinite=False)
		self.vae_optim.step()
		self.vae_encoder.eval()
		self.vae_decoder.eval()
		if self.reviewer is not None:
			self.reviewer.model._encoder = self.learner.model._encoder
		return {'vae_recon_loss': float(recon_loss.mean().item()),
				'vae_kl_loss': float(kl_loss.mean().item()),
				'vae_loss': float(loss.mean().item()),
				'vae_grad_norm': float(grad_norm)}
	
	def update(self, learner_replay_buffer, reviewer_replay_buffer, step, 
			update_learner=True, update_reviewer=False):
		"""Main update function. Corresponds to one iteration of the TOLD model learning."""
		learner_logs = {}
		reviewer_logs = {}
		learner_obs, learner_next_obses, learner_action, learner_reward, learner_idxs, learner_weights = None, None, None, None, None, None

		reviewer_batch_size = int(self.cfg.batch_size * 2 * self.cfg.reviewer_ratio)
		learner_batch_size = int(self.cfg.batch_size * 2 * (1 - self.cfg.reviewer_ratio))
		
		if update_learner:
			learner_obs, learner_next_obses, learner_action, learner_reward, learner_idxs,\
				  learner_weights = learner_replay_buffer.sample(reviewer_batch_size)
			learner_logs = self.learner.update(learner_replay_buffer, learner_obs, 
									  learner_next_obses, learner_action, learner_reward, 
									  learner_idxs, learner_weights, step)
			mean_score = learner_logs['learner_mean_prediction_score']
			if mean_score != 0:
				self.new_sigmoid_threshold = (self.new_sigmoid_threshold * 0.9 + mean_score * 0.1)

		if update_reviewer:
			reviewer_obs, reviewer_next_obses, reviewer_action, reviewer_reward,\
				  reviewer_idxs, reviewer_weights = reviewer_replay_buffer.sample(learner_batch_size)
			if learner_obs is not None:
				obs = torch.cat([learner_obs.unsqueeze(0), learner_next_obses[:-1]], dim=0)
				reward_from_learner, _, _ = self.reviewer_reward(obs, learner_next_obses, learner_action)
				reviewer_obs = torch.cat([reviewer_obs, learner_obs], dim=0)
				reviewer_next_obses = torch.cat([reviewer_next_obses, learner_next_obses], dim=1)
				reviewer_action = torch.cat([reviewer_action, learner_action], dim=1)
				reviewer_reward = torch.cat([reviewer_reward, reward_from_learner], dim=1)
				reviewer_weights = torch.cat([reviewer_weights, learner_weights], dim=0)
			reviewer_logs = self.reviewer.update(reviewer_replay_buffer, reviewer_obs, reviewer_next_obses, 
						 reviewer_action, reviewer_reward, reviewer_idxs, reviewer_weights, step, key='reviewer')
			mean_score = reviewer_logs['reviewer_mean_prediction_score']
			self.learner.model._dynamics = self.reviewer.model._dynamics
			self.learner.model._encoder = self.reviewer.model._encoder
			if mean_score != 0:
				self.new_sigmoid_threshold = (self.new_sigmoid_threshold * 0.9 + mean_score * 0.1)

		if self.cfg.update_vae:
			if self.cfg.use_reviewer and self.cfg.ckpt is not None:
				vae_obs_action = torch.cat([reviewer_obs, reviewer_action[0]], -1)
			else:
				vae_obs_action = torch.cat([learner_obs, learner_action[0]], -1)
			if self.old_vae_decoder is not None and self.old_vae_encoder is not None:
				simulated_obs_action = self.generate_obs_action(self.cfg.batch_size)
				vae_obs_action = torch.cat([vae_obs_action, simulated_obs_action], dim=0)
			learner_logs.update(self.update_vae(vae_obs_action))

		if step % self.cfg.vae_data_update_freq == 0 and\
			  self.cfg.use_vae and self.old_vae_encoder is not None and\
				  self.old_vae_decoder is not None:
			obs_action = self.generate_obs_action(self.cfg.batch_size)
			next_obs = self.old_dynamics(obs_action)
			obs = obs_action[:, :self.cfg.latent_dim]
			action = obs_action[:, self.cfg.latent_dim:]
			if self.cfg.use_reviewer and self.cfg.ckpt is not None:
				self.reviewer.update_wm(obs, next_obs, action)
				self.learner.model._dynamics = self.reviewer.model._dynamics
				self.learner.model._encoder = self.reviewer.model._encoder
			else:
				self.learner.update_wm(obs, next_obs, action)
		return {**learner_logs, **reviewer_logs}
	
	def state_dict(self):
		"""Retrieve state dict of DRAGO model, including slow-moving target network."""
		if self.reviewer is not None:
			return {'learner': self.learner.state_dict(),
					'reviewer': self.reviewer.state_dict(),
					'old_dynamics': self.old_dynamics.state_dict(),
					'old_encoder': self.old_encoder.state_dict(),
					'vae_encoder': self.vae_encoder.state_dict(),
					'vae_decoder': self.vae_decoder.state_dict(),
					'sigmoid_threshold': self.new_sigmoid_threshold}
		else:
			return {'learner': self.learner.state_dict(),
					'old_dynamics': self.old_dynamics.state_dict(),
					'old_encoder': self.old_encoder.state_dict(),
					'vae_encoder': self.vae_encoder.state_dict(),
					'vae_decoder': self.vae_decoder.state_dict(),
					'sigmoid_threshold': self.new_sigmoid_threshold}

	def save(self, fp):
		"""Save state dict of DRAGO model to filepath."""
		torch.save(self.state_dict(), fp)
	
	def load_model(self):
		"""Load model from checkpoint."""
			
		if self.cfg.ckpt is not None:
			d = torch.load(self.cfg.ckpt)
			if self.cfg.load_policy:
				self.learner.load(d['learner'])
				if self.cfg.use_reviewer:
					self.reviewer = TDMPC(self.cfg)
					if d.get('reviewer') is not None:
						self.reviewer.load(d['reviewer'])
					else:
						self.reviewer.load(d['learner'])
				print('Loading complete model.')

			else:
				self.learner.model._dynamics.load_state_dict(d['learner']['dynamics'])
				self.learner.model_target._dynamics.load_state_dict(d['learner']['dynamics_target'])
				self.learner.model._encoder.load_state_dict(d['learner']['encoder'])
				self.learner.model_target._encoder.load_state_dict(d['learner']['encoder_target'])
				if self.cfg.use_reviewer:
					# unify the world model of learner and reviewer
					self.reviewer = TDMPC(self.cfg)
					self.reviewer.model._dynamics.load_state_dict(d['learner']['dynamics'])
					self.reviewer.model_target._dynamics.load_state_dict(d['learner']['dynamics_target'])
					self.reviewer.model._encoder.load_state_dict(d['learner']['encoder'])
					self.reviewer.model_target._encoder.load_state_dict(d['learner']['encoder_target'])
				print('Loading pretrained world model only.')
			
			# load the old world model from previous wm
			self.old_dynamics = deepcopy(self.learner.model._dynamics).requires_grad_(False)
			self.old_encoder = deepcopy(self.learner.model._encoder).requires_grad_(False)
			
			# load vae
			if 'vae_encoder' in d.keys() and 'vae_decoder' in d.keys() and\
				 (self.cfg.use_vae or self.cfg.update_vae):
				self.vae_encoder.load_state_dict(d['vae_encoder'])
				self.vae_decoder.load_state_dict(d['vae_decoder'])
				self.old_vae_encoder = deepcopy(self.vae_encoder).requires_grad_(False)
				self.old_vae_decoder = deepcopy(self.vae_decoder).requires_grad_(False)
			if 'sigmoid_threshold' in d.keys():
				self.sigmoid_threshold = d['sigmoid_threshold']
				self.new_sigmoid_threshold = d['sigmoid_threshold']
				print('loaded sigmoid threshold:', self.sigmoid_threshold)
			else:
				print('use default sigmoid threshold:', self.sigmoid_threshold)
		else:
			print('Training from scratch.')