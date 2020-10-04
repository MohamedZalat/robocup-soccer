#!/usr/bin/env python

# The striker agent

import random
from .soccerpy.supervisor import Supervisor
from .soccerpy.world_model import WorldModel

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

class KrisletSupervisor(Supervisor):
    """
    The extended Agent class with specific heuritics
    """
    TURN_POS = 'turn+40'
    TURN_BALL = 'turn_ball'
    DASH = 'dash'
    KICK_TO_GOAL = 'kick_to_goal'
    TURN_NEG = 'turn-40'

    def _think(self):
        """
        Performs a single step of thinking for our agent.  Gets called on every
        iteration of our think loop.
        """

        # DEBUG:  tells us if a thread dies
        if not self._think_thread.is_alive() or not self._msg_thread.is_alive():
            raise Exception("A thread died.")

        # take places on the field by uniform number
        if not self.in_kick_off_formation:
            print('the side is {}'.format(self.wm.side))

            # used to flip x coords for other side
            side_mod = 1
            if self.wm.side == WorldModel.SIDE_R:
                side_mod = -1

            if self.wm.uniform_number == 9:
                self.wm.teleport_to_point((-5 * side_mod, 30))
            elif self.wm.uniform_number == 2:
                self.wm.teleport_to_point((-40 * side_mod, 15))
            elif self.wm.uniform_number == 3:
                self.wm.teleport_to_point((-40 * side_mod, 00))
            elif self.wm.uniform_number == 4:
                self.wm.teleport_to_point((-40 * side_mod, -15))
            elif self.wm.uniform_number == 5:
                self.wm.teleport_to_point((-5 * side_mod, -30))
            elif self.wm.uniform_number == 6:
                self.wm.teleport_to_point((-20 * side_mod, 20))
            elif self.wm.uniform_number == 7:
                self.wm.teleport_to_point((-20 * side_mod, 0))
            elif self.wm.uniform_number == 8:
                self.wm.teleport_to_point((-20 * side_mod, -20))
            elif self.wm.uniform_number == 1:
                self.wm.teleport_to_point((-10 * side_mod, 0))
            elif self.wm.uniform_number == 10:
                self.wm.teleport_to_point((-10 * side_mod, 20))
            elif self.wm.uniform_number == 11:
                self.wm.teleport_to_point((-10 * side_mod, -20))

            self.in_kick_off_formation = True

            return

        if (not self.wm.is_before_kick_off() and self.wm.play_mode != self.wm.PlayModes.TIME_OVER) \
                or self.wm.is_kick_off_us() or self.wm.is_playon():
            # The main decision loop
            return self.decisionLoop()

        if self.wm.play_mode == self.wm.PlayModes.TIME_OVER and not self.saved:
            if self._agent.save_traj:
                self._agent.save_dataset()

            # Report metrics of the classifier if it is not being trained during execution.
            if self._agent.clone:
                self._agent.report_results()

            # Report the stats of visited states and actions taken.
            self.stats.save()

            self.saved = True

    def decisionLoop(self):
        if not self.wm.ball:
            action = self.TURN_POS
            self.stats.log_env('ball not in vision')

        elif self.wm.ball.distance and self.wm.ball.distance > 1.0:
            if self.wm.ball.direction != 0:
                action = self.TURN_BALL
                self.stats.log_env('ball in vision')
            else:
                action = self.DASH
                self.stats.log_env('ball aligned')

        else:
            if not self.goal:
                action = self.TURN_POS
                self.stats.log_env('ball kickable and can NOT see goal')
            else:
                action = self.KICK_TO_GOAL
                self._agent.ctx.goal = self.goal
                self.stats.log_env('ball kickable and can see goal')

        self._agent.ctx.act = self.transform_action(action)
