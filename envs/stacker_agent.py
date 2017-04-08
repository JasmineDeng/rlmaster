from rlmaster.core.base_environment import *
import numpy as np
from overrides import overrides
from pyhelper_fns import vis_utils

def contains(obj1, obj2):
  x1, y1, _ = obj1.pos.tolist()
  x2, y2, _ = obj2.pos.tolist()
  s1, s2 = obj1.size, obj2.size
  if x2 + s2 <= x1 or x1 + s1 <= x2:
    return False
  if y2 + s2 <= y1 or y1 + s1 <= y2:
    return False
  return True

class StackedBox():

  def __init__(self, pos, size=2):
    # position is center of the box
    self.pos = pos.reshape((3,))
    self.size = size

class SimpleStackerSimulator(BaseSimulator):
  def __init__(self, **kwargs):
    super(SimpleStackerSimulator, self).__init__(**kwargs)
    self._imSz = 32
    self.width = self._imSz
    self.height = self._imSz
    self.move_block = StackedBox(np.zeros(3))
    self.other_block = StackedBox(np.zeros(3))

    self._im = np.zeros((self._imSz, self._imSz, 3), dtype=np.uint8)
    self._range_min = self.move_block.size
    self._range_max = self._imSz - self.move_block.size

  @overrides
  def step(self, pos):
    #####@pulkitag: You should make use of BaseDiscreteAction and BaseContinuousAction
    #### to input the actions. 
    pos = pos.reshape((2,))
    pos = np.append(pos, np.zeros(1))
    
    new_pos = np.clip(self.move_block.pos + pos, self._range_min, self._range_max)
    new_pos[2] = 0
    self.move_block.pos = new_pos
    if contains(self.move_block, self.other_block):
      self.move_block.pos[2] = self.other_block.size

  def _plot_object(self, coords, color='r'):
    x, y = coords
    mnx, mxx  = int(x), int(min(self._imSz, x + self.move_block.size))
    mny, mxy  = int(y), int(min(self._imSz, y + self.move_block.size))
    if color == 'r':
      self._im[mny:mxy, mnx:mxx, 0] = 255
    elif color == 'g':
      self._im[mny:mxy, mnx:mxx, 1] = 255
    else:
      self._im[mny:mxy, mnx:mxx, 2] = 255

  @overrides
  def get_image(self):
    imSz = self._imSz
    rng = np.linspace(self._range_min, self._range_max, imSz)
    x_1, y_1 = self.other_block.pos[0], self.other_block.pos[1]
    x_2, y_2 = self.move_block.pos[0], self.move_block.pos[1]
    self._im = np.zeros((imSz, imSz, 3), dtype=np.uint8)
    self._plot_object((x_1, y_1), 'r')
    self._plot_object((x_2, y_2), 'g')
    return self._im.copy()

  @overrides 
  def _setup_renderer(self):
    self._canvas = vis_utils.MyAnimation(None, height=self._imSz, width=self._imSz)

  @overrides
  def render(self):
    self._canvas._display(self.get_image())

class StackerIm(BaseObservation):

  @overrides
  def ndim(self):
    dim = {}
    dim['im'] = (self.simulator._imSz, self.simulator._imSz, 3)
    return dim

  @overrides
  def observation(self):
    obs = {}
    obs['im'] = self.simulator.get_image()
    return obs

class StateIm(BaseObservation):

  @overrides
  def ndim(self):
    dim = {}
    dim['im'] = (6, 1)
    return dim

  def scale(self, obs):
    norm = np.linalg.norm(obs)
    obs = np.copy(obs)
    obs = obs / 16 - 1
    return obs

  @overrides
  def observation(self):
    obs = {}
    m_pos = self.simulator.move_block.pos
    o_pos = self.simulator.other_block.pos
    new_im = np.append(self.scale(m_pos), self.scale(o_pos))
    obs['im'] = new_im
    return obs


####@pulkitag: This is good. 
class RewardStacker(BaseRewarder):
  
  @property
  def block_height(self):
    if hasattr(self.prms['sim'], 'other_block') and hasattr(self.prms['sim'], 'move_block'):
      return 1 if self.prms['sim'].move_block.pos[2] == self.prms['sim'].other_block.size else 0
    return 0

  @overrides
  def get(self):
    return self.block_height

class ContinuousStackerAction(BaseContinuousAction):
  @overrides
  def action_dim(self):
    return 2

  @overrides
  def process(self, action):
    return action

  def minval(self):
    return -1

  def maxval(self):
    return 1

class InitStacker(BaseInitializer):
  @overrides
  def sample_env_init(self):
    sim = self.simulator['sim']
    size = sim.move_block.size
    sim.move_block.pos = np.random.randint(sim._range_min, sim._range_max, size=3)
    sim.move_block.pos[2] = 0
    sim.other_block.pos = np.random.randint(sim._range_min, sim._range_max, size=3)
    sim.other_block.pos[2] = 0
    if contains(sim.move_block, sim.other_block):
      sim.move_block.pos[2] = size

def get_environment(obsType='StateIm', max_episode_length=100, initPrms={}, obsPrms={}, rewPrms={}, actPrms={}):
  sim = SimpleStackerSimulator()
  rewPrms = { 'sim': sim }
  initPrms = { 'sim': sim }
  initObj = InitStacker(initPrms)
  obsObj = globals()[obsType](sim, obsPrms)
  rewObj = RewardStacker(sim, rewPrms)
  actObj = ContinuousStackerAction(actPrms)
  env = BaseEnvironment(sim, initObj, obsObj, rewObj, actObj,
    params={'max_episode_length':max_episode_length})
  return env
