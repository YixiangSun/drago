import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal
from torch.autograd import Variable
import torch
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy


__REDUCE__ = lambda b: 'mean' if b else 'none'


def sample_gumbel(shape, eps=1e-10, use_cuda=False):
    """Sample from Gumbel(0, 1)"""
    if use_cuda:
        tens_type = torch.cuda.FloatTensor
    else:
        tens_type = torch.FloatTensor
    U = Variable(tens_type(*shape).uniform_(), requires_grad=False)
    return -torch.log(-torch.log(U + eps) + eps)


# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax_sample(logits, temperature, dim=-1):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )
    y = logits + gumbels
    #y = logits + sample_gumbel(logits.shape, use_cuda=logits.is_cuda)

    return F.softmax(y / temperature, dim=dim)


# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax(logits, temperature=1.0, hard=True, deterministic=False, dim=-1, return_prob=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probabilitiy distribution that sums to 1 across classes
    """

    if not deterministic:
        y = gumbel_softmax_sample(logits, temperature, dim=dim)
    else:
        y = F.softmax(logits, dim=dim)
    if hard:
        y_hard = onehot_from_logits(y, dim=dim)
        y_out = (y_hard - y).detach() + y

        if not return_prob:
            return y_out
        else:
            return y_out, y
    else:
        return y

def onehot_from_logits(logits, eps=0.0, dim=-1):
    """
    Given batch of logits, return one-hot sample using epsilon greedy strategy
    (based on given epsilon)
    """
    # get best (according to current policy) actions in one-hot form
    argmax_acs = (logits == logits.max(dim, keepdim=True)[0]).float()
    #print(logits[0],"a")
    #print(len(argmax_acs),argmax_acs[0])
    if eps == 0.0:
        return argmax_acs
    # get random actions in one-hot form
    rand_acs = Variable(torch.eye(logits.shape[1])[[np.random.choice(
        range(logits.shape[1]), size=logits.shape[0])]], requires_grad=False).cuda()

    # chooses between best and random actions using epsilon greedy
    return torch.stack([argmax_acs[i] if r > eps else rand_acs[i] for i, r in
                        enumerate(torch.rand(logits.shape[0]).cuda())])


def l1(pred, target, reduce=False):
	"""Computes the L1-loss between predictions and targets."""
	return F.l1_loss(pred, target, reduction=__REDUCE__(reduce))


def mse(pred, target, reduce=False):
	"""Computes the MSE loss between predictions and targets."""
	return F.mse_loss(pred, target, reduction=__REDUCE__(reduce))

def cross_entropy(pred, target, reduce=False):
	"""Computes the cross-entropy loss between predictions and targets."""
	return F.cross_entropy(pred, target, reduction=__REDUCE__(reduce))


def _get_out_shape(in_shape, layers):
	"""Utility function. Returns the output shape of a network for a given input shape."""
	x = torch.randn(*in_shape).unsqueeze(0)
	return (nn.Sequential(*layers) if isinstance(layers, list) else layers)(x).squeeze(0).shape


def orthogonal_init(m):
	"""Orthogonal layer initialization."""
	if isinstance(m, nn.Linear):
		nn.init.orthogonal_(m.weight.data)
		if m.bias is not None:
			nn.init.zeros_(m.bias)
	elif isinstance(m, nn.Conv2d):
		gain = nn.init.calculate_gain('relu')
		nn.init.orthogonal_(m.weight.data, gain)
		if m.bias is not None:
			nn.init.zeros_(m.bias)


def ema(m, m_target, tau):
	"""Update slow-moving average of online network (target network) at rate tau."""
	with torch.no_grad():
		for p, p_target in zip(m.parameters(), m_target.parameters()):
			p_target.data.lerp_(p.data, tau)

def vae(cfg, act_fn=nn.ELU()):
	# TODO: use CRelu activation function according to cfg in initialization
	if act_fn == 'crelu':
		act_fn = CRelu()
	encoder = nn.Sequential(nn.Linear(cfg.latent_dim + cfg.action_dim, cfg.enc_dim), act_fn,
							nn.Linear(cfg.enc_dim, cfg.enc_dim), act_fn,
							nn.Linear(cfg.enc_dim, cfg.vae_enc_dim * 2))
	decoder = nn.Sequential(nn.Linear(cfg.vae_enc_dim, cfg.enc_dim), act_fn,
                            nn.Linear(cfg.enc_dim, cfg.enc_dim), act_fn,
                            nn.Linear(cfg.enc_dim, cfg.latent_dim + cfg.action_dim))
	return encoder, decoder

def reparameterize(mu, logvar):
	std = torch.exp(0.5*logvar)
	eps = torch.randn_like(std)
	return mu + eps*std

def set_requires_grad(net, value):
	"""Enable/disable gradients for a given (sub)network."""
	for param in net.parameters():
		param.requires_grad_(value)


class TruncatedNormal(pyd.Normal):
	"""Utility class implementing the truncated normal distribution."""
	def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
		super().__init__(loc, scale, validate_args=False)
		self.low = low
		self.high = high
		self.eps = eps

	def _clamp(self, x):
		clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
		x = x - x.detach() + clamped_x.detach()
		return x

	def sample(self, clip=None, sample_shape=torch.Size()):
		shape = self._extended_shape(sample_shape)
		eps = _standard_normal(shape,
							   dtype=self.loc.dtype,
							   device=self.loc.device)
		eps *= self.scale
		if clip is not None:
			eps = torch.clamp(eps, -clip, clip)
		x = self.loc + eps
		return self._clamp(x)


class NormalizeImg(nn.Module):
	"""Normalizes pixel observations to [0,1) range."""
	def __init__(self):
		super().__init__()

	def forward(self, x):
		return x.div(255.)


class Flatten(nn.Module):
	"""Flattens its input to a (batched) vector."""
	def __init__(self):
		super().__init__()
		
	def forward(self, x):
		return x.view(x.size(0), -1)


def enc(cfg):
	"""Returns a TOLD encoder."""
	if not cfg.use_encoder:
		return nn.Identity()
	if cfg.modality == 'pixels':
		C = int(3*cfg.frame_stack)
		layers = [NormalizeImg(),
				  nn.Conv2d(C, cfg.num_channels, 7, stride=2), nn.ReLU(),
				  nn.Conv2d(cfg.num_channels, cfg.num_channels, 5, stride=2), nn.ReLU(),
				  nn.Conv2d(cfg.num_channels, cfg.num_channels, 3, stride=2), nn.ReLU(),
				  nn.Conv2d(cfg.num_channels, cfg.num_channels, 3, stride=2), nn.ReLU()]
		out_shape = _get_out_shape((C, cfg.img_size, cfg.img_size), layers)
		layers.extend([Flatten(), nn.Linear(np.prod(out_shape), cfg.latent_dim)])
	else:
		layers = [nn.Linear(cfg.obs_shape[0], cfg.enc_dim), nn.ELU(),
				  nn.Linear(cfg.enc_dim, cfg.latent_dim)]
	return nn.Sequential(*layers)

class CRelu(nn.Module):
    def forward(self, input):
        return torch.cat([F.relu(input), F.relu(-input)], -1)
	


def mlp(in_dim, mlp_dim, out_dim, act_fn=nn.ELU()):
	"""Returns an MLP."""
	if isinstance(mlp_dim, int):
		mlp_dim = [mlp_dim, mlp_dim]
	if act_fn == 'crelu':
		act_fn = CRelu()
		return nn.Sequential(
            nn.Linear(in_dim, mlp_dim[0]),
            act_fn,
            nn.Linear(mlp_dim[0]*2, mlp_dim[1]),
            act_fn,
            nn.Linear(mlp_dim[1]*2, out_dim))
	return nn.Sequential(
		nn.Linear(in_dim, mlp_dim[0]), act_fn,
		nn.Linear(mlp_dim[0], mlp_dim[1]), act_fn,
		nn.Linear(mlp_dim[1], out_dim))

def q(cfg, act_fn=nn.ELU()):
	"""Returns a Q-function that uses Layer Normalization."""
	return nn.Sequential(nn.Linear(cfg.latent_dim+cfg.action_dim, cfg.mlp_dim), nn.LayerNorm(cfg.mlp_dim), nn.Tanh(),
						 nn.Linear(cfg.mlp_dim, cfg.mlp_dim), nn.ELU(),
						 nn.Linear(cfg.mlp_dim, 1))


class RandomShiftsAug(nn.Module):
	"""
	Random shift image augmentation.
	Adapted from https://github.com/facebookresearch/drqv2
	"""
	def __init__(self, cfg):
		super().__init__()
		self.pad = int(cfg.img_size/21) if cfg.modality == 'pixels' else None

	def forward(self, x):
		if not self.pad:
			return x
		n, c, h, w = x.size()
		assert h == w
		padding = tuple([self.pad] * 4)
		x = F.pad(x, padding, 'replicate')
		eps = 1.0 / (h + 2 * self.pad)
		arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
		arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
		base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
		base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
		shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
		shift *= 2.0 / (h + 2 * self.pad)
		grid = base_grid + shift
		return F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)


class Episode(object):
	"""Storage object for a single episode."""
	def __init__(self, cfg, init_obs):
		self.cfg = cfg
		self.device = torch.device(cfg.device)
		dtype = torch.float32 if cfg.modality == 'state' else torch.uint8
		self.obs = torch.empty((cfg.episode_length+1, *init_obs.shape), dtype=dtype, device=self.device)
		self.obs[0] = torch.tensor(init_obs, dtype=dtype, device=self.device)
		self.action = torch.empty((cfg.episode_length, cfg.action_dim), dtype=torch.float32, device=self.device)
		self.reward = torch.empty((cfg.episode_length,), dtype=torch.float32, device=self.device)
		self.cumulative_reward = 0
		self.truncated = False
		self.terminated = False
		self._idx = 0
	
	def __len__(self):
		return self._idx

	@property
	def first(self):
		return len(self) == 0
	
	def __add__(self, transition):
		self.add(*transition)
		return self

	def add(self, obs, action, reward, terminated, truncated):
		self.obs[self._idx+1] = torch.tensor(obs, dtype=self.obs.dtype, device=self.obs.device)
		self.action[self._idx] = action
		self.reward[self._idx] = reward
		self.cumulative_reward += reward
		self.truncated = truncated
		self.terminated = terminated
		self._idx += 1


class ReplayBuffer():
	"""
	Storage and sampling functionality for training TD-MPC / TOLD.
	The replay buffer is stored in GPU memory when training from state.
	Uses prioritized experience replay by default."""
	def __init__(self, cfg):
		self.cfg = cfg
		self.device = torch.device(cfg.device)
		self.capacity = min(cfg.train_steps, cfg.max_buffer_size)
		dtype = torch.float32 if cfg.modality == 'state' else torch.uint8
		obs_shape = cfg.obs_shape if cfg.modality == 'state' else (3, *cfg.obs_shape[-2:])
		self._obs = torch.empty((self.capacity+1, *obs_shape), dtype=dtype, device=self.device)
		self._last_obs = torch.empty((self.capacity//cfg.episode_length, *cfg.obs_shape), dtype=dtype, device=self.device)
		self._action = torch.empty((self.capacity, cfg.action_dim), dtype=torch.float32, device=self.device)
		self._reward = torch.empty((self.capacity,), dtype=torch.float32, device=self.device)
		self._priorities = torch.ones((self.capacity,), dtype=torch.float32, device=self.device)
		self._eps = 1e-6
		self._full = False
		self.idx = 0

	def __add__(self, episode: Episode):
		self.add(episode)
		return self

	def add(self, episode: Episode):
		self._obs[self.idx:self.idx+self.cfg.episode_length] = episode.obs[:-1] if self.cfg.modality == 'state' else episode.obs[:-1, -3:]
		self._last_obs[self.idx//self.cfg.episode_length] = episode.obs[-1]
		self._action[self.idx:self.idx+self.cfg.episode_length] = episode.action
		self._reward[self.idx:self.idx+self.cfg.episode_length] = episode.reward
		if self._full:
			max_priority = self._priorities.max().to(self.device).item()
		else:
			max_priority = 1. if self.idx == 0 else self._priorities[:self.idx].max().to(self.device).item()
		mask = torch.arange(self.cfg.episode_length) >= self.cfg.episode_length-self.cfg.horizon
		new_priorities = torch.full((self.cfg.episode_length,), max_priority, device=self.device)
		new_priorities[mask] = 0
		self._priorities[self.idx:self.idx+self.cfg.episode_length] = new_priorities
		self.idx = (self.idx + self.cfg.episode_length) % self.capacity
		self._full = self._full or self.idx == 0

	def update_priorities(self, idxs, priorities):
		self._priorities[idxs] = priorities.squeeze(1).to(self.device) + self._eps

	def _get_obs(self, arr, idxs, batch_size):
		if self.cfg.modality == 'state':
			return arr[idxs]
		obs = torch.empty((batch_size, 3*self.cfg.frame_stack, *arr.shape[-2:]), dtype=arr.dtype, device=torch.device('cuda'))
		obs[:, -3:] = arr[idxs].cuda()
		_idxs = idxs.clone()
		mask = torch.ones_like(_idxs, dtype=torch.bool)
		for i in range(1, self.cfg.frame_stack):
			mask[_idxs % self.cfg.episode_length == 0] = False
			_idxs[mask] -= 1
			obs[:, -(i+1)*3:-i*3] = arr[_idxs].cuda()
		return obs.float()

	def sample(self, batch_size):
		probs = (self._priorities if self._full else self._priorities[:self.idx]) ** self.cfg.per_alpha
		probs /= probs.sum()
		total = len(probs)
		idxs = torch.from_numpy(np.random.choice(total, batch_size, p=probs.cpu().numpy(), replace=not self._full)).to(self.device)
		weights = (total * probs[idxs]) ** (-self.cfg.per_beta)
		weights /= weights.max()

		obs = self._get_obs(self._obs, idxs, batch_size)
		next_obs_shape = self._last_obs.shape[1:] if self.cfg.modality == 'state' else (3*self.cfg.frame_stack, *self._last_obs.shape[-2:])
		next_obs = torch.empty((self.cfg.horizon+1, batch_size, *next_obs_shape), dtype=obs.dtype, device=obs.device)
		action = torch.empty((self.cfg.horizon+1, batch_size, *self._action.shape[1:]), dtype=torch.float32, device=self.device)
		reward = torch.empty((self.cfg.horizon+1, batch_size), dtype=torch.float32, device=self.device)
		for t in range(self.cfg.horizon+1):
			_idxs = idxs + t
			next_obs[t] = self._get_obs(self._obs, _idxs+1, batch_size)
			action[t] = self._action[_idxs]
			reward[t] = self._reward[_idxs]

		mask = (_idxs+1) % self.cfg.episode_length == 0
		next_obs[-1, mask] = self._last_obs[_idxs[mask]//self.cfg.episode_length].cuda().float()
		if not action.is_cuda:
			action, reward, idxs, weights = \
				action.cuda(), reward.cuda(), idxs.cuda(), weights.cuda()

		return obs, next_obs, action, reward.unsqueeze(2), idxs, weights

class ContPseudoCounts:
    """
    PseudoCount class that estimates pseudo-counts for continuous states using the coin flip algorithm.
    This implementation uses a neural network classifier to distinguish between observed data and samples from a prior,
    updating the model incrementally and computing pseudo-counts based on the change in prediction probabilities.
    """ 
    def __init__(self, input_dim, hidden_dim=128, device='cuda', learning_rate=1e-3):
        self.device = device
        self.input_dim = input_dim

        # Neural network classifier
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.BCELoss()

        # Store previous model parameters for pseudo-count computation
        self.prev_model = None

    def update(self, x):
        """
        Update the density model with new observations x.

        Args:
            x (torch.Tensor): Observations of shape (batch_size, input_dim).
        """
        x = x.to(self.device)
        batch_size = x.size(0)

        # Store the previous model before updating
        self.prev_model = deepcopy(self.model)

        # Generate negative samples from a prior distribution (e.g., standard normal)
        x_neg = torch.randn_like(x)

        # Create labels: 1 for observed data, 0 for negative samples
        inputs = torch.cat([x, x_neg], dim=0)
        labels = torch.cat([torch.ones(batch_size, 1), torch.zeros(batch_size, 1)], dim=0).to(self.device)

        # Shuffle the inputs and labels
        perm = torch.randperm(2 * batch_size)
        inputs = inputs[perm]
        labels = labels[perm]

        # Forward pass
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, labels)

        # Backward pass and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    @torch.no_grad()
    def count(self, x):
        """
        Compute the pseudo-count N(x) for observation x.

        Args:
            x (torch.Tensor): Observations of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Pseudo-counts for each observation in x.
        """
        x = x.to(self.device)

        # Get the probability before and after the update
        if self.prev_model is None:
            # If no previous model, assume prior probability is 0.5
            p_prev = torch.full((x.size(0), 1), 0.5, device=self.device)
        else:
            p_prev = self.prev_model(x)
            p_prev = torch.clamp(p_prev, min=1e-6, max=1 - 1e-6)

        p = self.model(x)
        p = torch.clamp(p, min=1e-6, max=1 - 1e-6)

        # Compute pseudo-count N(x) using the change in probabilities
        # N(x) = (1 - p_prev) / (p - p_prev)
        numerator = (1 - p_prev)
        denominator = (p - p_prev) + 1e-8  # Add epsilon to avoid division by zero
        N = numerator / denominator
        N = torch.clamp(N, min=0.0)  # Ensure counts are non-negative

        return N

    @torch.no_grad()
    def get_intrinsic_rewards(self, x):
        """
        Compute the intrinsic reward based on the pseudo-count.

        Args:
            x (torch.Tensor): Observations of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Intrinsic rewards for each observation in x.
        """
        N = self.count(x)
        intrinsic_reward = 1.0 / torch.sqrt(N + 1e-8)
        return intrinsic_reward


class PseudoCounts:
    def __init__(self, cfg):
        self.size = cfg.size
        self.states = torch.zeros(cfg.size, cfg.size, 4)

    def update(self, state):

        x = int((state[0].cpu()[0] + 1) * self.size / 2)
        y = int((state[0].cpu()[1] + 1) * self.size / 2)
        z = int((state[0].cpu()[2] + 1) * 2)
        self.states[x, y, z] += 1
    
    def get_intrinsic_rewards(self, states):
		# Assuming states is a batch of states with shape (batch_size, 3)
		
		# Calculate x, y, z for the batch
        x = ((states[:, 0] + 1) * self.size / 2).int()
        y = ((states[:, 1] + 1) * self.size / 2).int()
        z = ((states[:, 2] + 1) * 2).int()
		
		# Create a mask for the batch
        batch_size = states.shape[0]
        mask = torch.zeros(batch_size, self.size, self.size, 4)
		
		# Using advanced indexing to set the mask values
        mask[torch.arange(batch_size), x, y, z] = 1
		
		# Broadcasting self.states to match the shape of the mask
        states_expanded = self.states.unsqueeze(0).expand(batch_size, -1, -1, -1)
		
		# Compute intrinsic reward
        intrinsic_reward = 1 / torch.sqrt((states_expanded * mask).sum(dim=(1, 2, 3)) + 0.99999) - 1 / 99999
		
        return intrinsic_reward.unsqueeze(1).cuda()
	
	


def linear_schedule(schdl, step):
	"""
	Outputs values following a linear decay schedule.
	Adapted from https://github.com/facebookresearch/drqv2
	"""
	try:
		return float(schdl)
	except ValueError:
		match = re.match(r'linear\((.+),(.+),(.+)\)', schdl)
		if match:
			init, final, duration = [float(g) for g in match.groups()]
			mix = np.clip(step / duration, 0.0, 1.0)
			return (1.0 - mix) * init + mix * final
	raise NotImplementedError(schdl)