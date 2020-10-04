# !/usr/bin/env python
from .soccerpy.learning_agent import LearningAgent as baseAgent
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch
import random
import os
import errno
from sklearn.metrics import classification_report

# The striker agent


# methods from actionHandler are
# CATCH = "catch"(rel_direction)
# CHANGE_VIEW = "change_view"
# DASH = "dash"(power)
# KICK = "kick"(power, rel_direction)
# MOVE = "move"(x,y) only pregame
# SAY = "say"(you_can_try_cursing)
# SENSE_BODY = "sense_body"
# TURN = "turn"(rel_degrees in 360)
# TURN_NECK = "turn_neck"(rel_direction)

# potentially useful from aima
# learning.py
# mdp
#
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.in_norm = nn.BatchNorm1d(9)
        self.fc1 = nn.Linear(9, 500)
        self.fc1_norm = nn.BatchNorm1d(500)
        self.fc2 = nn.Linear(500, 500)
        self.fc2_norm = nn.BatchNorm1d(500)
        self.fc3 = nn.Linear(500, 5)

    def forward(self, x):
        x = x.view(-1, 9)
        x = self.in_norm(x)
        x = F.relu(self.fc1(x))
        x = self.fc1_norm(x)
        x = F.relu(self.fc2(x))
        x = self.fc2_norm(x)
        return F.log_softmax(self.fc3(x), dim=1)

class Agent(baseAgent):
    """
    The extended Agent class with specific heuritics
    """
    def __init__(self, model_path='models/krislet_mlp.pt', load_model=False,
                 dataset_dir='data', load_dataset=True,
                 report_name='krislet_mlp', *pargs, **kwargs):
        model = Net()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-4)
        self.criterion = nn.CrossEntropyLoss()

        super().__init__(model, model_path, load_model,
                         dataset_dir, load_dataset, report_name, *pargs, **kwargs)

        # Print the device we are using.
        print('device = {}\n'.format(self.device))

    def train(self, epochs=10):
        self.model.train()
        print('training policy..')

        for epoch in range(epochs):
            data = self.get_data()

            target = self.get_target()

            self.optimizer.zero_grad()

            output = self.model(data)

            loss = self.criterion(output, target)

            loss.backward()

            self.optimizer.step()

        print('Iteration = {}\n'.format(self.iteration))
        print('Saving model parameters...\n')
        self.save_model()

    def init_hidden(self, batch_size):
        pass

    def switch_to_train(self):
        # Requires the model to have been trained first.
        pass
