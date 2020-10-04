# --------------------------------------------------------------------------------
#
# Every imitation learning/behavioral cloning agent's base class.
#
# --------------------------------------------------------------------------------
import numpy as np
import torch
import random
import os
import errno
from sklearn.metrics import classification_report

# --------------------------------------------------------------------------------
from .agent import Agent

# --------------------------------------------------------------------------------
class LearningAgent(Agent):
    def __init__(self, model, model_path=None, load_model=False, dataset_dir=None,
                 load_dataset=True, report_name=None, passthrough=False,
                 clone=False, cloning_epochs=100, save_traj=True, t=10):
        # model: pytorch model to use.
        #
        # model_path: pytorch model to load and/or save.
        #
        # dataset_dir: the dataset directory to use for loading and storing the
        #              dataset collected.
        #
        # load_dataset: set to True if you want to load the stored dataset,
        #               otherwise set to False (default is True).
        #
        # report_name: the name of the classification report of the model.
        #
        # passthrough: set this flag to False if you want the learning agent
        #              to take actions. If this flag is set to True, the agent
        #              is selecting the supervisor's action.
        #
        # clone: set this flag to True if you want to perform behavioral
        #        cloning as opposed to imitation learning. The dataset stored
        #        will be used (i.e. dataset_dir has to contain data).
        #
        # cloning_epochs: The number of training epochs in cloning mode.
        #
        # save_traj: set this flag to True to save the trajectory to the stored
        #            dataset.
        #
        # t: the time before a t-step trajectory is completed and the agent
        #    needs to update its policy.
        super().__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() \
            else torch.device('cpu')

        # This is to ensure that we are using the best available option.
        self.model = model.to(self.device)

        if load_model and model_path:
            self.model.load_state_dict(torch.load(model_path))

        self.dataset_dir = dataset_dir
        self.model_path = model_path

        if report_name:
            self.report_name = report_name + '.txt'
        else:
            self.report_name = 'report.txt'

        self.iteration = 0
        self.passthrough = passthrough
        self.env_stack = list()
        self.act_stack = list()
        self.clone = clone
        self.save_traj = save_traj
        self.t = t

        self.y_true = list()
        self.y_pred = list()

        if load_dataset:
            self.load_dataset()

        if self.clone and self.env_stack:
            self.train(cloning_epochs)
        elif self.clone and not self.env_stack:
            raise Exception('No data in dataset_dir for the model to clone.')

    def train(self, epochs):
        raise Exception('train(self, epochs) method not implemented by the agent.')

    def get_data(self):
        return torch.stack(self.env_stack)

    def get_target(self):
        return torch.stack(self.act_stack).squeeze(1)

    def take_action(self):
        self.model.eval()
        data = torch.from_numpy(self.ctx.env).float().to(self.device)

        output = self.model(data)
        selected_action = torch.argmax(output, dim=1).item()

        print('supervisor_action = {}'.format(self.ctx.act))
        print('selected_action = {}'.format(selected_action))

        self.y_pred.append(selected_action)
        self.y_true.append(self.ctx.act)

        self._take_action(selected_action)

    def _take_action(self, selected_action):
        # Transform action to string representation.
        selected_action = self.ctx.parent.reverse_transform_action(selected_action)

        # The agent object reports that it took the selected action.
        self.ctx.parent.stats.log_act(selected_action)

        if selected_action == self.ctx.parent.TURN_POS:
            self.ctx.parent.wm.ah.turn(40)

        elif selected_action == self.ctx.parent.TURN_NEG:
            self.ctx.parent.wm.ah.turn(-40)

        elif selected_action == self.ctx.parent.TURN_BALL:
            if self.ctx.parent.wm.ball and self.ctx.parent.wm.ball.direction:
                self.ctx.parent.wm.ah.turn(self.ctx.parent.wm.ball.direction)

        elif selected_action == self.ctx.parent.DASH:
            if self.ctx.parent.wm.ball and self.ctx.parent.wm.ball.distance:
                self.ctx.parent.wm.ah.dash(10 * self.ctx.parent.wm.ball.distance)
            else:
                self.ctx.parent.wm.ah.dash(10)

        elif selected_action == self.ctx.parent.KICK_TO_GOAL:
            if self.ctx.goal and self.ctx.goal.direction is not None:
                self.ctx.parent.wm.ah.kick(100, self.ctx.goal.direction)
            else:
                print('self.ctx.goal = {}'.format(self.ctx.goal))
                if self.ctx.goal:
                    print('self.ctx.goal.direction = {}'.format(self.ctx.goal.direction))

        else:
            raise Exception('invalid action key')

    def _append_to_dataset(self, env, act):
        self.env_stack.append(torch.from_numpy(env).float().to(self.device))
        self.act_stack.append(torch.Tensor([act]).long().to(self.device).view(-1))

    def think(self):
        """
        Performs a single step of thinking for our agent.  Gets called on every
        iteration of our think loop.
        """
        if self.ctx.parent.wm.is_kick_off_us() or \
           self.ctx.parent.wm.is_playon():
            # The main decision loop
            return self.decisionLoop()

    def decisionLoop(self):
        self.iteration += 1
        self._append_to_dataset(self.ctx.env, self.ctx.act)

        # If the supervisor is supervising take your own
        # action and not the supervisor's action.
        if not self.passthrough:
            self.take_action()

        else:
            self._take_action(self.ctx.act)

        if self.iteration % self.t == 0 and not self.clone \
           and not self.passthrough:
            self.train()

    def load_dataset(self):
        # If the dataset directory is None, do nothing
        if not self.dataset_dir:
            return

        env_file_name = 'data.csv'
        act_file_name = 'actions.csv'
        env_path = os.path.join(self.dataset_dir, env_file_name)
        act_path = os.path.join(self.dataset_dir, act_file_name)

        # If the dataset directory does not exist, do nothing
        if not os.path.exists(os.path.dirname(env_path)):
            return

        file_env = open(env_path, 'r')
        file_act = open(act_path, 'r')

        env_stack = np.loadtxt(file_env)
        act_stack = np.loadtxt(file_act)

        file_env.close()
        file_act.close()

        for env, act in zip(env_stack, act_stack):
            self.env_stack.append(torch.from_numpy(env).float().to(self.device))
            self.act_stack.append(torch.Tensor([int(act)]).long().to(self.device).view(-1))

    def save_dataset(self):
        env_file_name = 'data.csv'
        act_file_name = 'actions.csv'
        env_path = os.path.join(self.dataset_dir, env_file_name)
        act_path = os.path.join(self.dataset_dir, act_file_name)
        # Create directories in the model path if they do not exist.
        if not os.path.exists(os.path.dirname(env_path)):
            try:
                os.makedirs(self.dataset_dir)
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        file_env = open(env_path, 'wb')
        file_act = open(act_path, 'wb')

        for env, act in zip(self.env_stack, self.act_stack):
            np.savetxt(file_env, env.cpu().numpy().reshape(1, -1))
            np.savetxt(file_act, act.cpu().numpy())
        file_env.close()
        file_act.close()

    def save_model(self):
        # Do not save the model if model path is not defined.
        if not self.model_path:
            return

        # Create directories in the model path if they do not exist.
        if not os.path.exists(os.path.dirname(self.model_path)):
            try:
                os.makedirs(os.path.dirname(self.model_path))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        torch.save(self.model.state_dict(), self.model_path)

    def report_results(self):
        report_dir = 'report'
        report_path = os.path.join(report_dir, self.report_name)

        # Create directories in the model path if they do not exist.
        if not os.path.exists(os.path.dirname(report_path)):
            try:
                os.makedirs(report_dir)
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        report_file = open(report_path, 'w')

        # Get the name of each class.
        classes = sorted(self.ctx.parent.action_dict.keys(),
                         key=self.ctx.parent.action_dict.get)

        print(classes)
        size = len(set(self.y_true))

        report_file.write(classification_report(self.y_true,
                                                self.y_pred,
                                                target_names=classes[:size]))
        report_file.close()
