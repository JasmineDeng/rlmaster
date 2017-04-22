from rlmaster.core.base_environment import *
import numpy as np
from overrides import overrides
from pyhelper_fns import vis_utils

def contains(obj1, obj2):
  if not obj1 or not obj2:
    return False
  x1, y1 = obj1.pos[0], obj1.pos[1]
  x2, y2 = obj2.pos[0], obj2.pos[1]
  s1, s2 = obj1.size, obj2.size
  x1, y1 = x1, y1
  x2, y2 = x2, y2
  if x2 + 2 * s2 <= x1 or x1 + 2 * s1 <= x2:
    return False
  if y2 + 2 * s2 <= y1 or y1 + 2 * s1 <= y2:
    return False
  return True

class StackedBox():
  def __init__(self, pos, size=2):
    self.pos = pos.reshape((3,))
    self.size = size

class ReacherArm():
  def __init__(self, pos, size=2):
    self.pos = pos.reshape((2,))
    self.size = size

    self.block = None

class SimpleReacherSimulator(BaseSimulator):
  def __init__(self, num_blocks=2, **kwargs):
    # should always have same size blocks (width = height = length = n)
    super(SimpleReacherSimulator, self).__init__(**kwargs)
    self._imSz = 32
    self.width = self._imSz
    self.height = self._imSz
    self.move_block = [StackedBox(np.zeros(3)) for i in range(num_blocks)]
    self.goal = StackedBox(np.zeros(3))
    self.reacher_arm = ReacherArm(np.zeros(2))

    self._im = np.zeros((self._imSz, self._imSz, 3), dtype=np.uint8)
    self._range_min = self.reacher_arm.size
    self._range_max = self._imSz - self.reacher_arm.size

  @property
  def done(self):
    for block in self.move_block:
      if not contains(block, self.goal):
        return False
    return True

  @overrides
  def step(self, action):
    action = action.reshape((2,))

    on_goal = contains(self.reacher_arm.block, self.goal)
    on_block = None
    for block in self.move_block:
      if contains(self.reacher_arm, block):
        on_block = block
        break

    if self.reacher_arm.block is None and on_block is not None and not contains(on_block, self.goal):
      self.reacher_arm.block = on_block

    if on_goal:
      self.reacher_arm.block.pos[2] = self.goal.size
      self.reacher_arm.block = None

    self.reacher_arm.pos = np.clip(self.reacher_arm.pos + action, self._range_min, self._range_max)

    if self.reacher_arm.block is not None:
      block = self.reacher_arm.block
      action = np.append(action, 0)
      block.pos = np.clip(block.pos + action, self._range_min, self._range_max)
      block.pos[2] = self._imSz
  
  def _plot_object(self, coords, box, color='r'):
    x, y = coords
    mnx, mxx  = int(max(x - box[0], 0)), int(min(self._imSz, x + box[0]))
    mny, mxy  = int(max(y - box[1], 0)), int(min(self._imSz, y + box[1]))
    if color == 'r':
      self._im[mny:mxy, mnx:mxx, 0] = 255
    elif color == 'g':
      self._im[mny:mxy, mnx:mxx, 1] = 255
    else:
      self._im[mny:mxy, mnx:mxx, 2] = 255

  @overrides
  def get_image(self):
    # other blocks in red, active block in green
    imSz = self._imSz
    self._im = np.zeros((imSz, imSz, 3), dtype=np.uint8)
    x, y = self.goal.pos[0], self.goal.pos[1]
    self._plot_object((x, y), (2, 2), 'r')
    for block in self.move_block:
      x, y = block.pos[0], block.pos[1]
      self._plot_object((x, y), (2, 2), 'g')
    x, y = self.reacher_arm.pos[0], self.reacher_arm.pos[1]
    self._plot_object((x, y), (2, 2), 'b')
    return self._im.copy()

  @overrides 
  def _setup_renderer(self):
    self._canvas = vis_utils.MyAnimation(None, height=self._imSz, width=self._imSz)

  @overrides
  def render(self):
    if not hasattr(self, '_canvas'):
      self._setup_renderer()
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
    num_blocks = len(self.simulator.move_block)
    dim['im'] = (num_blocks * 3 + 6, 1)
    return dim

  def scale(self, obs):
    obs = np.copy(obs)
    obs = obs / float(self.simulator._imSz) * 2 - 1
    return obs

  @overrides
  def observation(self):
    obs = {}

    m_block = self.simulator.move_block
    g_pos = self.simulator.goal.pos
    r_pos = self.simulator.reacher_arm.pos
    new_im = np.append(self.scale(g_pos), self.scale(r_pos))
    new_im = np.append(new_im, np.array([1 if self.simulator.reacher_arm.block is not None else 0]))
    for b in m_block:
      new_im = np.append(new_im, self.scale(b.pos))
    obs['im'] = new_im
    return obs

####@pulkitag: This is good. 
class RewardReacher(BaseRewarder):

  @property
  def contains_block(self):
    if hasattr(self.prms['sim'], 'move_block') and hasattr(self.prms['sim'], 'goal'):
      move_block = self.prms['sim'].move_block
      goal = self.prms['sim'].goal
      reward = 1 if all([contains(block, goal) for block in move_block]) else 0
      return reward
    return 0

  @overrides
  def get(self):
    return self.contains_block

# need additional actions: to pick up and to drop the block
class ContinuousReacherAction(BaseContinuousAction):
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

class InitReacher(BaseInitializer):
  @overrides
  def sample_env_init(self):
    sim = self.simulator['sim']
    sim.reacher_arm.pos = np.random.randint(sim._range_min, sim._range_max, size=2)
    sim.reacher_arm.block = None
    
    sim.goal.pos = np.random.randint(sim._range_min, sim._range_max, size=3)
    sim.goal.pos[2] = 0
    for block in sim.move_block:
      block.pos = np.random.randint(sim._range_min, sim._range_max, size=3)
      block.pos[2] = 0
      if contains(sim.goal, block):
        block.pos[2] = sim.goal.size

def get_environment(obsType='StateIm', max_episode_length=100, initPrms={}, obsPrms={}, rewPrms={}, actPrms={}):
  sim = SimpleReacherSimulator()
  initPrms = {'sim' : sim}
  initObj = InitReacher(initPrms)
  obsObj = globals()[obsType](sim, obsPrms)
  rewPrms = { 'sim': sim }
  rewObj = RewardReacher(sim, rewPrms)
  actObj = ContinuousReacherAction(actPrms)
  env = BaseEnvironment(sim, initObj, obsObj, rewObj, actObj,
    params={'max_episode_length':max_episode_length})
  return env
