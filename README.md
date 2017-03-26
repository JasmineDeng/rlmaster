# rlmaster

Environments and RL algorithms. 

## Environment
An environment is defined by:

- Simulator: an object of `type BaseSimulator` that simulates the effect of agent's actions in the environment and provides
             observations.

- Initializer: an object of type `BaseInitializer` that defines how the epsiode should be initialized.

- Observer: an object of `type BaseObserver` that choses the desired representation from the simulator. It is useful when 
            it is required to train an agent in the same environment but with different observations (for eg: training with
            pixels or with state features)
            
- Rewarder: an object of `type BaseRewarder` that describes the reward function. It is useful when the agent needs to be trained
            on the same environment but with different reward functions. 
            
- ActionProcessor: an object of `type BaseAction` that defines functions that process the action. Generally simulator works
                   by applying continous forces/torque, however the desired action space might be discrete or continuous. 
                   `BaseAction` objects allow the user to easily switch between different action spaces. 

### Interactive Mode

To run environments in interactive mode, a function mapping the keyboard inputs into commands supplied to the agent must be defined. This function is typically called, `str2action` in `rlmaster` repository. The environment can be run in interactive mode in the following way:

```
from envs.mujoco_envs import move_single_env
env = move_single_env.get_environment(actType='ContinuousAction', imSz=480)
env.interactive(move_single_env.str2action)
```
You can use the commands, `w, s, d, a` to move the agent and `q` to quit the interactive mode.

### Cheetah in Interactive Mode

No interface has been developed yet, but after creating an instance of SimpleStackerAgent in from `stacker_agent`, calling `_setup_renderer` and `render` will open a display window. You can step by entering an xy position that will move the second block there instantly. Calling `render` again will show the updated positions of the blocks. They are initialized at position (0,0) if not called with the initializer.

##Environment in openAI gym format

```
from envs import move_agent
from core import gym_wrapper
env = move_agent.get_environment()
gymEnv = gym_wwapper.GymWrapper(env)
```
                               



