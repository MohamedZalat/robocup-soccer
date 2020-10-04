#!/usr/bin/env python

import threading
import time
import random
import sys
import multiprocessing as mp
import sys
import os

# import agent types (positions)
from aigent.soccerpy.agent import Agent as A0
# strikers
from aigent.agent_mlp import Agent as A1
# defenders
from aigent.krislet_supervisor import KrisletSupervisor
from aigent.statebased_supervisor import StateBasedSupervisor

# set team
TEAM_NAME = 'Expert'
NUM_PLAYERS = 1

# return type of agent: midfield, striker etc.
def agent_type(position):
    return {
    }.get(position, StateBasedSupervisor)

# spawn an agent of team_name, with position
def spawn_agent(team_name, position):
    """
    Used to run an agent in a seperate physical process.
    """
    # return type of agent by position, construct
    a = agent_type(position)(A1(load_dataset=False, model_path=None,
                                passthrough=True, clone=False,
                                save_traj=False),
                             model_name='statebased_expert')
    a.connect("localhost", 6000, team_name)
    a.play()

    # we wait until we're killed
    while True:
        # we sleep for a good while since we can only exit if terminated.
        time.sleep(1)


if __name__ == "__main__":

    # spawn all agents as seperate processes for maximum processing efficiency
    agentthreads = []
    for position in range(1, NUM_PLAYERS + 1):
        print("  Spawning agent %d..." % position)

        at = mp.Process(target=spawn_agent, args=(TEAM_NAME, position))
        at.daemon = True
        at.start()

        agentthreads.append(at)

    print("Spawned %d agents." % len(agentthreads))
    print()
    print("Playing soccer...")

    # wait until killed to terminate agent processes
    try:
        while True:
            time.sleep(0.05)
            sys.stdout.flush()
    except KeyboardInterrupt:
        print()
        print("Killing agent threads...")

        # terminate all agent processes
        count = 0
        for at in agentthreads:
            print("  Terminating agent %d..." % count)
            at.terminate()
            count += 1
        print("Killed %d agent threads." % (count - 1))

        print()
        print("Exiting.")
        sys.exit()
