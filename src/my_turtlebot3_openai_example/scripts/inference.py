#!/usr/bin/env python3

import os
import gym
import numpy
import math
import random
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from functools import reduce

import time
from gym import wrappers
# ROS packages required
import rospy
import rospkg
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, inputs, outputs):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(inputs, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.head = nn.Linear(64, outputs)

    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.head(x)


def select_action(state, eps_threshold):
    sample = random.random()
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).argmax().item()
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


if __name__ == '__main__':
    rospy.init_node('turtlebot3_world_deepq_inference', anonymous=True, log_level=rospy.INFO)

    task_and_robot_environment_name = rospy.get_param('/turtlebot3/task_and_robot_environment_name')
    env = StartOpenAI_ROS_Environment(task_and_robot_environment_name)
    rospy.loginfo("Gym environment done")

    # Set the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('my_turtlebot3_openai_example')
    outdir = os.path.join(pkg_path, 'training_results')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get number of actions from gym action space
    n_actions = env.action_space.n
    n_observations = 120
    eps_threshold = 0.9

    # initialize networks with input and output sizes
    policy_net = DQN(n_observations, n_actions).to(device)
    policy_net.load_state_dict(torch.load(os.path.join(outdir, '3000_rl_deepQ_model.pth'), weights_only=True, map_location=device))
    policy_net.eval()
    
    observation = env.reset()
    state = torch.tensor(observation, device=device, dtype=torch.float)
    
    try:
        while True:
            action = policy_net(state).argmax().item()
            action = select_action(state, eps_threshold)
            observation, reward, done, info = env.step(action)
            
            if done:
                env.reset()
                state = torch.tensor(observation, device=device, dtype=torch.float)
    except KeyboardInterrupt:
        env.close()
