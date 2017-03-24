from move_agent import MoveTeleportSimulator, ObsIm, BaseRewarder

class HalfCheetahSimulator(MoveTeleportSimulator):
  def __init__(self, **kwargs):
    super(HalfCheetahSimulator, self).__init__(**kwargs)
    self.model = MjModel('../rlmaster/envs/mujoco_envs/xmls/half_cheetah.xml')
    self._pos = {}
    self._pos['torso'] = np.zeros((3,))
    self._range_min = -1
    self._range_max = 1

    self._imSz = 32
    self._im = np.zeros((self._imSz, self._imSz, 3), dtype=np.uint8)
    self.body_comvel = 0
    self.action = np.zeros((1, 2))

  @overrides
  def step(self, ctrl):
    self.model.data.ctrl = ctrl
    for i in range(10):
      self.model.step()
    self.model.forward()
    ind = self.model.body_names.index('torso')
    self._pos['torso'] = self.model.body_pos[ind]
    self.body_comvel = self.model.body_comvels[ind]
    self.action = ctrl

  @overrides 
  def get_image(self):
    imSz = self._imSz
    rng = np.linspace(self._range_min, self._range_max, imSz)
    g_x, g_y = self._get_bin(rng, self._pos['torso'])
    self._im = np.zeros((imSz, imSz, 3), dtype=np.uint8)   
    self._plot_object((g_x, g_y), 'r')
    return self._im.copy()

class CheetahIm(ObsIm):
  @overrides
  def observation(self):
    obs = {}
    obs['im'] = self.simulator.get_image().flatten()
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
    return ctrl_cost + run_cost
  