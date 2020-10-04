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
        self.input_dim = 9
        self.hidden_dim = 128
        self.n_layers = 1
        self.hidden = None
        self.device = None

        self.in_norm = nn.BatchNorm1d(9)
        self.lstm_norm = nn.BatchNorm1d(self.hidden_dim)
        self.lstm = nn.LSTM(self.input_dim,
                            self.hidden_dim,
                            self.n_layers,
                            batch_first=True)
        self.out = nn.Linear(self.hidden_dim, 5)

    def forward(self, x, init_hidden=False):
        x = x.view(-1, 9)

        if self.hidden is None or init_hidden:
            self.hidden = self.init_hidden(1)

        x = self.in_norm(x)

        # Input is always a single batch of a variable sequence.
        x = x.unsqueeze(0)

        x, self.hidden = self.lstm(x, self.hidden)

        self.hidden = (self.hidden[0].detach(),
                       self.hidden[1].detach())

        x = x.contiguous().view(-1, self.hidden_dim)

        x = self.lstm_norm(F.relu(x))
        x = self.out(x)
        return F.log_softmax(x, dim=1)

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        self.hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device),
                       weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device))

    def switch_to_train(self):
        # Requires the model to have been trained first.
        pass

class Agent(baseAgent):
    """
    The extended Agent class with specific heuritics
    """
    def __init__(self, model_path='models/krislet_lstm.pt', load_model=False,
                 dataset_dir='data', load_dataset=True,
                 report_name='krislet_lstm', *pargs, **kwargs):
        model = Net()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-4)
        self.criterion = nn.CrossEntropyLoss()

        super().__init__(model, model_path, load_model,
                         dataset_dir, load_dataset, report_name, *pargs, **kwargs)

        # Print the device we are using.
        print('device = {}\n'.format(self.device))
        self.model.device = self.device

    def train(self, epochs=10):
        # Save the last hidden state to use when back in evaluation mode.
        hidden = self.model.hidden

        self.model.train()
        print('training policy..')

        for epoch in range(epochs):
            data = self.get_data().view(-1, 9)
            target = self.get_target()
            output = self.model(data, init_hidden=True)

            loss = self.criterion(output, target)

            # loss.backward(retain_graph=epoch != (epochs - 1))
            loss.backward()

            # Clip is 5.
            nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.optimizer.step()

        print('Iteration = {}\n'.format(self.iteration))
        print('Saving model parameters...\n')
        self.save_model()

        self.model.hidden = hidden
