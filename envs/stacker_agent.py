from move_agent import MoveTeleportSimulator, ObsIm, BaseRewarder

class StackedBox():

  def __init__(self, pos, width=3, height=3, length=3):
    assert len(pos) == 3, 'must enter an xyz position'

    self.pos = pos
    self.width = width 
    self.height = height
    self.length = length

  def contains(self, new_pos):
    x, y, z = self.pos
    new_x, new_y, new_z = new_pos
    return new_x >= x and new_x <= x + self.width \
      and new_y >= y and new_y <= y + self.height and new_z >= z and new_z <= z + self.length

    
### @pulkitag: This should inherit from BaseSimulator. Any reason for this? 
class SimpleStackerSimulator(MoveTeleportSimulator):
  def __init__(self, **kwargs):
    super(SimpleStackerSimulator, self).__init__(**kwargs)
    self.width = self._imSz
    self.height = self._imSz
    self.block1 = StackedBox(np.random.randint(0, 32 - 3, size=3))
    self.block1.pos[2] = 0
    self.block2 = StackedBox(np.random.randint(0, 32 - 3, size=3))
    self.block2.pos[2] = 0
    if self.block1.contains(self.block2.pos):
      self.block2.pos[2] = self.block1.height

    self._imSz = 32
    self._im = np.zeros((self._imSz, self._imSz, 3), dtype=np.uint8)
    self._range_min = 0
    self._range_max = 32 - self.block1.width

  @overrides
  def step(self, pos):
    #####@pulkitag: You should make use of BaseDiscreteAction and BaseContinuousAction
    #### to input the actions. 
    assert len(pos) == 2, 'step takes in an xy position'

    x, y = min(pos[0], 32 - 3), min(pos[1], 32 - 3)
    self.block2.pos = np.array([x, y, 0])
    if self.block1.contains(self.block2.pos):
      self.block2.pos[2] = self.block1.height

  @overrides
  def _plot_object(self, coords, color='r'):
    x, y = coords
    mnx, mxx  = x, min(self._imSz, x + self.block1.width)
    mny, mxy  = y, min(self._imSz, y + self.block1.height)
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
    x_1, y_1 = self.block1.pos[0], self.block1.pos[1]
    x_2, y_2 = self.block2.pos[0], self.block2.pos[1]
    self._im = np.zeros((imSz, imSz, 3), dtype=np.uint8)
    self._plot_object((x_1, y_1), 'r')
    self._plot_object((x_2, y_2), 'g')
    return self._im.copy()

####@pulkitag: Again, why are you inherting from ObsIm? 
class StackerIm(ObsIm): 
  @overrides
  def observation(self):
    obs = {}
    obs['im'] = self.simulator.get_image().flatten()
    return obs

####@pulkitag: This is good. 
class RewardStacker(BaseRewarder):
  
  @property
  def block_height(self):
    return self.prms['sim'].block2.pos[2] if hasattr(self.prms['sim'], 'block2') else 0

  @overrides
  def get(self):
    return self.block_height
  
###@pulkitag: See get_environment in move_agent - you should have a similar function
### so that it easy to instatiate a basic version of the environment. 
