from rlmaster.core.base_environment import *
import numpy as np
from overrides import overrides
from pyhelper_fns import vis_utils

def contains(obj1, obj2):
  x1, y1, _ = obj1.pos.tolist()
  x2, y2, _ = obj2.pos.tolist()
  s1, s2 = obj1.size, obj2.size
  if x2 + w2 <= x1 or x1 + w1 <= x2:
    return False
  if y2 + l2 <= y1 or y1 + l1 <= y2:
    return False
  return True

class StackedBox():

  def __init__(self, pos, size=3):
    assert len(pos) == 3, 'must enter an xyz position'

    self.pos = pos.reshape((3,))
    # height is the height in 3d, box in 2d is width x length
    self.size = size

class ReacherArm():

  def __init__(self, pos, size=4):
    # takes in xyz position, but z is implicitly infinity
    self.pos = pos.reshape((2,))
    self.size = size

    self.hold_block = False

class SimpleReacherSimulator(BaseSimulator):
  def __init__(self, **kwargs):
    # should always have same size blocks (width = height = length = n)
    super(SimpleStackerSimulator, self).__init__(**kwargs)
    self._imSz = 32
    self.width = self._imSz
    self.height = self._imSz
    self.move_block = StackedBox(np.zeros(3))
    self.other_block = StackedBox(np.zeros(3))

    self.reacher_arm = ReacherArm(np.zeros(3))

    self._im = np.zeros((self._imSz, self._imSz, 3), dtype=np.uint8)
    self._range_min = 0
    self._range_max = 32 - self.move_block.size

  @overrides
  def step(self, action):
    # should take in xyz position, but z position doesn't matter
    action = action.reshape((5,))
    pos = action[:3]
    grab_act = action[3] < 4
    drop_act = action[4] < 4
    print(grab_act, drop_act)
    
    if grab_act: 
      if contains(self.reacher_arm, self.move_block):
        self.reacher_arm.hold_block = True

    elif drop_act:
      if self.reacher_arm.hold_block:
        self.reacher_arm.hold_block = False
        self.move_block.pos[2] = 0
        if contains(self.other_block, self.move_block):
          self.move_block.pos[2] = self.other_block.size

    if self.reacher_arm.hold_block:
      delta = pos[0] - self.reacher_arm.pos[0], pos[1] - self.reacher_arm.pos[1]
      self.move_block.pos = np.array([self.move_block.pos[0] + delta[0], 
              self.move_block.pos[1] + delta[1], 10]) # to preserve relative position

    self.reacher_arm.pos = pos

  def _plot_object(self, coords, box, color='r'):
    x, y = coords
    width, length = box[0], box[1]
    mnx, mxx  = int(x), int(min(self._imSz, x + width))
    mny, mxy  = int(y), int(min(self._imSz, y + length))
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
    x, y = self.other_block.pos[0], self.other_block.pos[1]
    self._plot_object((x, y), (3, 3), 'r')
    x, y = self.move_block.pos[0], self.move_block.pos[1]
    self._plot_object((x, y), (3, 3), 'g')
    x, y = self.reacher_arm.pos[0], self.reacher_arm.pos[1]
    self._plot_object((x, y), (4, 4), 'b')
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
    dim['im'] = (9, 1)
    return dim

  @overrides
  def observation(self):
    obs = {}
    new_im = np.append(self.simulator.move_block.pos, self.simulator.other_block.pos)
    new_im = np.append(new_im, self.simulator.reacher_arm.pos[:2])
    new_im = np.append(new_im, np.array([1 if self.simulator.reacher_arm.hold_block else 0]))
    obs['im'] = new_im
    return obs

####@pulkitag: This is good. 
class RewardReacher(BaseRewarder):
  
  @property
  def contains_block(self):
    if hasattr(self.prms['sim'], 'reacher_arm') and hasattr(self.prms['sim'], 'move_block') and hasattr(self.prms['sim'], 'other_block'):
      move_block = self.prms['sim'].move_block0,  0,  0,
      other_block = self.prms['sim'].other_block
      reacher_arm = self.prms['sim'].reacher_arm
      return 1 if move_block.pos[2] == other_block.size and not reacher_arm.hold_block else 0
    return 0

  @overrides
  def get(self):
    return self.contains_block

# need additional actions: to pick up and to drop the block
class ContinuousReacherAction(BaseContinuousAction):
  @overrides
  def action_dim(self):
    return 5

  @overrides
  def process(self, action):
    return np.around(action, decimals=0)

  def minval(self):
    return 0

  def maxval(self):
    return 32

class InitReacher(BaseInitializer):
  @overrides
  def sample_env_init(self):
    sim = self.simulator['sim']
    sim.reacher_arm.pos = np.random.randint(0, 32 - sim.reacher_arm.size, size=3)
    sim.reacher_arm.hold_block = False
    sim.move_block.pos = np.random.randint(0, 32 - sim.move_block.size, size=3)
    sim.move_block.pos[2] = 0
    sim.other_block.pos = np.random.randint(0, 32 - sim.move_block.size, size=3)
    sim.other_block.pos[2] = 0
    if contains(sim.other_block, sim.move_block):
      sim.move_block.pos[2] = sim.other_block.size

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
