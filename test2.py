import os
if not os.path.exists('bullet3'):
  os.system('git clone https://github.com/bulletphysics/bullet3.git')

import gym
import pybullet_envs

def main():
  environment = gym.make("KukaDiverseObjectGrasping-v0", maxSteps=1000, 
                         isDiscrete=False, renders=True, removeHeightHack=True, isTest=True)

  done = False

  environment.reset()
  while (not done):

    action = environment.action_space.sample()

    obs, reward, done, _= environment.step(action)
    

    if done:
            print("Episode finished")
            break
    
  environment.close()


if __name__ == "__main__":
  main()