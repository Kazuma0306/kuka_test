import gym
import torch
from sac import SAC
import numpy as np
from replaybuffer import ReplayBuffer
from replaybuffer import ReplayBuffer2
import kuka_env

import pybullet_envs


episodes_num = 30000
batch_size = 256
steps_limit = 20
eval_interval=10**4,
num_eval_episodes=3


def main():
    if torch.cuda.is_available():
        print("Device Count: {}".format(torch.cuda.device(0)))
        print("Device Name (first): {}".format(torch.cuda.get_device_name(0)))
    else:
        device = torch.device("cpu")

    #test or not
    test = False


    if test == False:
        env = kuka_env.get_test_env()
    else:
        env = kuka_env.get_train_env()


    #make replay buffer object
    replay_buffer = ReplayBuffer2(capacity=30000, observation_shape=([3, 48, 48]),
                                   action_dim=3)

    model = SAC(rb = replay_buffer, device = device)

    if test == False:
        model.load('models/SAC_kuka')



#***********************train loop***************************


    for i in range(episodes_num):
        print("--------Episode %d--------" % i)

        reward_per_episode = 0
        state = env.reset()


        state = torch.as_tensor(state, device=device)
        state = state.transpose(1, 2).transpose(0, 1).unsqueeze(0)

        for steps in range(steps_limit):

            action, _ = model.explore(state)

            next_state, reward, done, _ = env.step(action) #更新


            if test == False:
                next_state = torch.as_tensor(next_state, device=device)
                next_state = next_state.transpose(1, 2).transpose(0, 1).unsqueeze(0)
                # For replay buffer. (s_t, a_t, s_t+1, r)
                model.add_experience(action, state, next_state, reward, done)

                #networkの更新
                if model.is_update(steps):
                    model.update()

                if ((i+1)%20 == 0):
                    model.save()


            reward_per_episode += reward
            state = next_state


            if (done or steps == steps_limit -1):
                print("Steps count: %d" % steps)
                print("Total reward: %d" % reward_per_episode)

                break


if __name__ == '__main__':
    main()


            