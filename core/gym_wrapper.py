import numpy as np
import rlmaster.core.base_environment as base_environment
from gym import spaces
from rllab.spaces.box import Box
from rllab.core.serializable import Serializable
from rllab.misc.overrides import overrides
from rllab.misc import logger

class GymWrapper(object):
  def __init__(self, env):
    self._env = env
    if isinstance(self.env.action_processor, 
                  base_environment.BaseDiscreteAction):
      self.action_space = spaces.Discrete(self.env.num_actions())
    else:
      assert isinstance(self.env.action_processor,
                        base_environment.BaseContinuousAction)
      self.action_space = Box(
                      low  = env.action_processor.minval(), 
                      high = env.action_processor.maxval(), 
                      shape=(env.action_processor.action_dim()))
    obsNdim = self.env.observation_ndim()   
    obsKeys = list(obsNdim.keys())
    assert len(obsKeys) == 1, 'gym only supports one observation type'
    self._obsKey = obsKeys[0]
    self.observation_space = Box(low=0, high=255, shape=obsNdim[obsKeys[0]])

  @property
  def frameskip(self):
    raise NotImplementedError

  @property
  def ale(self):
    raise NotImplementedError

  @property
  def env(self):
    return self._env

  @property
  def _n_actions(self):
    return self.env.action_dim()

  def _observation(self):
    """
    Gym environment only supports one kind of observation
    """
    return self.env.observation()[self._obsKey]


  def _reset(self):
    self.env.reset()
    return self._observation()


  def reset(self):
    return self._reset()


  def _step(self, action):
    assert type(action) == np.ndarray, 'action must be a nd-array'
    self.env.step(action)
    obs    = self._observation()
    reward = self.env.reward()
    # done   = self.env.simulator.done # test?
    done   = False
    return obs, reward, done, dict(reward=reward)


  def step(self, action):
    return self._step(action)


  def viewer_setup(self):
    self.env.setup_renderer() 


  def render(self):
    return self._render()


  def _render(self):
    return self.env.render() 
