from rlmaster.core.base_environment import *
import numpy as np
from overrides import overrides
from rllab.mujoco_py import MjModel

from pyhelper_fns import vis_utils

def str2action(cmd):
  cmd = cmd.strip()
  if cmd == 'w':
    #up
    ctrl = [0, 0.1]
  elif cmd == 'a':
    #left
    ctrl = [-0.1, 0]
  elif cmd == 'd':
    #right
    ctrl = [0.1, 0]
  elif cmd == 's':
    #down
    ctrl = [0, -0.1]
  else:
    return None
  ctrl = np.array(ctrl).reshape((2,))
  return ctrl 
    

class DiscreteActionFour(BaseDiscreteAction):
  @overrides
  def num_actions(self):
    return 4

  def minval(self):
    return 0

  def maxval(self):
    return 3

  @overrides
  def process(self, action):
    assert len(action) == 1
    assert action[0] in [0, 1, 2, 3]
    if action[0] == 0:
      #up
      ctrl = [0, 0.1]
    elif action[0] == 1:
      #left
      ctrl = [-0.1, 0]
    elif action[0] == 2:
      #right
      ctrl = [0.1, 0]
    elif action[0] == 3:
      #down
      ctrl = [0, -0.1]
    else:
      raise Exception('Action %s not recognized' % action)
    ctrl = np.array(ctrl).reshape((2,))
    return ctrl 

class ContinuousAction(BaseContinuousAction):
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

class MoveTeleportSimulator(BaseSimulator):
  def __init__(self, **kwargs):
    super(MoveTeleportSimulator, self).__init__(**kwargs)
    self._pos = {}
    self._pos['manipulator'] = np.zeros((2,))
    self._pos['object']      = np.zeros((2,))
    self._pos['goal']        = np.zeros((2,))
    #Maximum and minimum locations of objects
    self._range_min = -1
    self._range_max = 1
    #Manipulate radius
    self._manipulate_radius = 0.2
    #Image size
    self._imSz = 32 
    self._im = np.zeros((self._imSz, self._imSz, 3), dtype=np.uint8)      

  def object_names(self):
    return self._pos.keys()

  def _dist(self, x, y):
    dist = x - y
    dist = np.sqrt(np.sum(dist * dist))
    return dist

  def dist_manipulator_object(self):
    return self._dist(self._pos['manipulator'], self._pos['object'])

  def dist_object_goal(self):
    return self._dist(self._pos['object'], self._pos['goal']) 
   
  def _clip(self, val):
    val = np.clip(val, self._range_min, self._range_max)
    return val

  @overrides
  def step(self, ctrl):
    self._pos['manipulator'] += ctrl.reshape((2,))
    self._pos['manipulator'] = self._clip(self._pos['manipulator']) 
    if self.dist_manipulator_object() < self._manipulate_radius:
      self._pos['object'] = self._pos['manipulator'].copy()  
 
  def _get_bin(self, rng, coords):
    try:
      x = np.where(rng <= coords[0])[0][-1]
      y = np.where(rng <= coords[1])[0][-1]       
    except:
      print (coords)
      raise Exception('Something is incorrect') 
    return x, y

  def _plot_object(self, coords, color='r'):
    x, y = coords
    mnx, mxx  = max(0, x - 2), min(self._imSz, x + 2)
    mny, mxy  = max(0, y - 2), min(self._imSz, y + 2)
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
    g_x, g_y = self._get_bin(rng, self._pos['goal'])
    m_x, m_y = self._get_bin(rng, self._pos['manipulator'])
    o_x, o_y = self._get_bin(rng, self._pos['object'])
    self._im = np.zeros((imSz, imSz, 3), dtype=np.uint8)      
    self._plot_object((o_x, o_y), 'r')
    self._plot_object((g_x, g_y), 'g')
    self._plot_object((m_x, m_y), 'b')
    return self._im.copy()

  @overrides 
  def _setup_renderer(self):
    self._canvas = vis_utils.MyAnimation(None, height=self._imSz, width=self._imSz)

  @overrides
  def render(self):
    self._canvas._display(self.get_image())

class InitFixed(BaseInitializer):
  @overrides
  def sample_env_init(self):
    self.simulator._pos['goal'] = np.array([0.5, 0.5])
    self.simulator._pos['object'] = np.array([-0.7, -0.5])
    self.simulator._pos['manipulator'] = np.array([-0.9, -0.6])     

 
class InitRandom(BaseInitializer):
  @overrides
  def sample_env_init(self):
    range_mag = self.simulator._range_max - self.simulator._range_min
    for k in self.simulator._pos.keys():
      self.simulator._pos[k] = range_mag * self.random.rand(2,) + \
                               self.simulator._range_min

class ObsState(BaseObservation):
  @overrides
  def ndim(self):
    dim = {}
    dim['feat'] = 6
    return dim

  @overrides
  def observation(self):
    obs = {}
    obs['feat'] = np.zeros((6,))
    for i, k in enumerate(self.simulator._pos.keys()):
      obs[2*i, 2*i + 2] = self.simulator._pos[k].copy()
    return obs
 
class ObsIm(BaseObservation):
  @overrides
  def ndim(self):
    dim = {}
    # dim['im'] = (self.simulator._imSz, self.simulator._imSz, 3)
    dim['im'] = (3072, 1)
    return dim

  @overrides
  def observation(self):
    obs = {}
    obs['im'] =  self.simulator.get_image().flatten()
    return obs

class RewardSimple(BaseRewarder):
  #The radius around the goal in which reward is provided to the agent.
  @property
  def radius(self):
    return self.prms['radius'] if hasattr(self.prms, 'radius') else 0.2

  @overrides 
  def get(self):
    if self.simulator.dist_object_goal() < self.radius:
      return 1
    else:
      return 0 

def get_environment(initName='InitRandom', obsName='ObsIm', rewName='RewardSimple',
                    actType='DiscreteActionFour', simName='MoveTeleportSimulator',
                    max_episode_length=100, initPrms={}, obsPrms={}, rewPrms={}, 
                    actPrms={}):
  sim     = globals()[simName]()
  initObj = globals()[initName](sim, initPrms)
  obsObj  = globals()[obsName](sim, obsPrms)
  rewPrms = { 'sim': sim }
  rewObj  = globals()[rewName](sim, rewPrms)
  actObj  = globals()[actType](actPrms)
  env     = BaseEnvironment(sim, initObj, obsObj, rewObj, actObj, 
              params={'max_episode_length':max_episode_length})
  return env
