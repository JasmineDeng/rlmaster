from rlmaster.core.base_environment import *
import numpy as np
from overrides import overrides
from rllab.mujoco_py import MjModel, MjViewer

class HalfCheetahSimulator(BaseSimulator):
	def __init__(self, **kwargs):
		super(HalfCheetahSimulator, self).__init__(**kwargs)

		self._imSz = 512
		self._im = np.zeros((self._imSz, self._imSz, 3), dtype=np.uint8)

		self.model = MjModel('../rlmaster/envs/mujoco_envs/xmls/half_cheetah.xml')
		self.viewer = None
		self._pos = {}
		self._pos['torso'] = np.zeros((3,))
		self._range_min = -1
		self._range_max = 1

		self.body_comvel = 0
		self.action = np.zeros((1, 2))

		self.frame_skip = 1

	@overrides
	def step(self, ctrl, loop=False):
		self.model.data.ctrl = ctrl + np.random.normal(size=ctrl.shape)
		for i in range(self.frame_skip):
		  self.model.step()
		self.model.forward()
		ind = self.model.body_names.index('torso')
		self._pos['torso'] = self.model.body_pos[ind]
		self.body_comvel = self.model.body_comvels[ind]
		self.action = ctrl

	@overrides 
	def get_image(self):
		data, width, height = self.viewer.get_image()
		self._im = np.fromstring(data, dtype='uint8').reshape(height, width, 3)[::-1,:,:]
		print(self._im)
		return self._im.copy()

	@overrides
	def _setup_renderer(self):
		self.viewer = MjViewer(visible=True, init_width=self._imSz, init_height=self._imSz)
		self.viewer.start()
		self.viewer.set_model(self.model)

	@overrides
	def render(self):
  		self.viewer.loop_once()

class CheetahIm(BaseObservation):

	def get_body_com(self, body_name):
		idx = self.simulator.model.body_names.index(body_name)
		return self.simulator.model.data.com_subtree[idx]

	@overrides
	def ndim(self):
		dim = {}
		dim['im'] = (1, 20)
		return dim

	@overrides
	def observation(self):
		obs = {}
		current_obs = np.concatenate([
			self.simulator.model.data.qpos.flatten()[1:],
			self.simulator.model.data.qvel.flat,
			self.get_body_com('torso').flat,
		])
		obs['im'] = current_obs
		return obs

class RewardCheetah(BaseRewarder):

	@property
	def action(self):
		return self.prms['sim'].action if hasattr(self.prms['sim'], 'action') else np.zeros((1,2))

	@property
	def body_comvel(self):
		return self.prms['sim'].body_comvel if hasattr(self.prms['sim'], 'body_comvel') else 0

	@overrides
	def get(self):
		ctrl_cost = 1e-1 * 0.5 * np.sum(np.square(self.action))
		run_cost = -1 * self.body_comvel[0]
		return -(ctrl_cost + run_cost)

class ContinuousCheetahAction(BaseContinuousAction):
	@overrides
	def action_dim(self):
		return 6

	def minval(self):
		return -1.25

	def maxval(self):
		return 1.1

	@overrides
	def process(self, action):
		return action

# TODO(jasmine): fix this so it initializes (is it called at the start of each epoch? find out)
class InitCheetah(BaseInitializer):

	@overrides
	def sample_env_init(self):
		pass

def get_environment(max_episode_length=100, initPrms={}, obsPrms={}, actPrms={}):
	sim = HalfCheetahSimulator()
	initObj = InitCheetah(sim, initPrms)
	obsObj = CheetahIm(sim, obsPrms)
	rewPrms = { 'sim': sim }
	rewObj = RewardCheetah(sim, rewPrms)
	actObj = ContinuousCheetahAction(actPrms)
	env = BaseEnvironment(sim, initObj, obsObj, rewObj, actObj, 
		params={'max_episode_length':max_episode_length})
	return env
