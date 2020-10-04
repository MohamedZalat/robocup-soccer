#!/usr/bin/env python
from functools import reduce
import os
import errno
import csv
import matplotlib.pyplot as pyplot
from .filelock.filelock import FileLock
import math

class Stats(object):
    MODEL = 'model'
    COND_FMT = '{}|{}'
    CHI_SCORE_KEY_FMT = '{}: {} and {}'
    MONTE_CARLO_DIST_KEY_FMT = '{} and {}'
    TRAJ_FILE_NAME_FMT = '{}_trajectory.csv'

    def __init__(self, model_name):
        self.model_name = model_name
        self.env_stats = dict()
        self.act_stats = dict()
        self.env_cond_stats = dict()
        self.act_cond_stats = dict()
        self.act_cond_env_stats = dict()
        self.prev_env = None
        self.prev_act = None
        self.trajectory = list()

        # Add an extra field to identify the model used.
        self.env_stats[Stats.MODEL] = model_name
        self.act_stats[Stats.MODEL] = model_name
        self.env_cond_stats[Stats.MODEL] = model_name
        self.act_cond_stats[Stats.MODEL] = model_name
        self.act_cond_env_stats[Stats.MODEL] = model_name

    def log_env(self, env):
        # The supervisor object uses this method to report the
        # environment state visited, as the supervisor knows the
        # pertinent states to record.
        self.increment_frequency(self.env_stats, env)

        # Update the conditional frequency if there is a previous
        # environment state.
        if self.prev_env:
            self._increment_cond_freq(self.env_cond_stats,
                                      self.prev_env,
                                      env)

        self.prev_env = env

    def log_act(self, act):
        # The agent object uses this method to report the action
        # taken. The supervisor never needs to use this method as
        # it is not the one taking actions.
        self.increment_frequency(self.act_stats, act)

        # P(A|E)
        self._increment_cond_freq(self.act_cond_env_stats,
                                  self.prev_env,
                                  act)

        # Update the conditional frequency if there is a previous
        # environment state.
        if self.prev_act:
            # P(A|A)
            self._increment_cond_freq(self.act_cond_stats,
                                      self.prev_act,
                                      act)

        self.prev_act = act

        # Add the state action pair to the trajectory.
        self._append_to_trajectory(self.prev_env, self.prev_act)

    def _increment_cond_freq(self, cond_stats, prev_key, key):
        cond_key = Stats.COND_FMT.format(key, prev_key)
        self.increment_frequency(cond_stats, cond_key)

    def increment_frequency(self, stats, key):
        try:
            stats[key] += 1
        except KeyError:
            stats[key] = 1

    def save(self):
        self._save(self.env_stats, 'environment')
        self._save(self.act_stats, 'action')
        self._save(self.env_cond_stats, 'environment_cond')
        self._save(self.act_cond_stats, 'action_cond')
        self._save(self.act_cond_env_stats, 'action_cond_environment')
        self._save_trajectory()

    def _append_to_trajectory(self, state, action):
        self.trajectory.append({'state': state,
                                'action': action})

    def _save_trajectory(self):
        traj_file_name = Stats.TRAJ_FILE_NAME_FMT.format(self.model_name)

        report_dir = 'report'

        traj_path = os.path.join(report_dir, traj_file_name)

        # Create directories in the model path if they do not exist.
        if not os.path.exists(os.path.dirname(traj_path)):
            try:
                os.makedirs(report_dir)
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        # Write the stats to the stats file.
        with open(traj_path, 'w', newline='') as traj_file:
            field_names = ['state', 'action']

            traj_writer = csv.DictWriter(traj_file,
                                         fieldnames=field_names)
            traj_writer.writeheader()

            for pair in self.trajectory:
                traj_writer.writerow(pair)

    def _save(self, stats, _type):
        stats_file_name = '{}_stats.csv'.format(_type)

        report_dir = 'report'

        stats_path = os.path.join(report_dir, stats_file_name)

        # Create directories in the model path if they do not exist.
        if not os.path.exists(os.path.dirname(stats_path)):
            try:
                os.makedirs(report_dir)
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        # Helper method to write to the file the new entry
        def write_to_file(entries):
            # Add the stats entry to the list of entries.
            entries.append(stats)

            # Write the stats to the stats file.
            with open(stats_path, 'w', newline='') as stats_file:
                field_names = reduce(lambda x, y: x.union(y.keys()), entries, set())
                field_names = sorted(list(field_names))

                stats_writer = csv.DictWriter(stats_file,
                                              fieldnames=field_names)
                stats_writer.writeheader()

                for entry in entries:
                    stats_writer.writerow(entry)

        # Check if the file exists to not overwrite previous entries.
        entries = list()
        if os.path.isfile(stats_path):
            with FileLock(stats_path):
                entries = Stats._read(stats_path)

                # Filter the list to have no records of the current model
                entries = [entry for entry in entries
                           if entry[Stats.MODEL] != self.model_name]

                write_to_file(entries)
        else:
            with FileLock(stats_path):
                write_to_file(entries)

    @staticmethod
    def _read(stats_path):
        # Returns the entries of the stats file.
        old_entries = list()
        with open(stats_path) as stats_file:
            stats_reader = csv.DictReader(stats_file)

            for stats in stats_reader:
                old_entries.append(stats)

        return old_entries

    @staticmethod
    def read(stats_path):
        with FileLock(stats_path):
            return Stats._read(stats_path)

    @staticmethod
    def get_chi_score(entries):
        # Assumes entries is a 2 item dictionary.
        def get_frequency(entry, key):
            # Returns frequency if a value exists,
            # otherwise return 0 for the key.
            try:
                return int(entry[key])
            except BaseException:
                return 0

        keys = set()
        model_totals = dict()
        key_totals = dict()
        for entry in entries:
            for key in entry:
                # Add keys to use for chi test.
                if key != Stats.MODEL:
                    keys.add(key)
                    frequency = get_frequency(entry, key)

                    # Keep track of the total frequency of each model.
                    try:
                        model_totals[entry[Stats.MODEL]] += frequency
                    except KeyError:
                        model_totals[entry[Stats.MODEL]] = frequency

                    try:
                        key_totals[key] += frequency
                    except KeyError:
                        key_totals[key] = frequency

        grand_total = sum(list(model_totals.values()))

        chi_score = 0
        for key in keys:
            for entry in entries:
                # expected = model_total * key_total / grand_total
                expected = model_totals[entry[Stats.MODEL]] * key_totals[key] / float(grand_total)

                if expected == 0:
                    continue

                observed = get_frequency(entry, key)

                chi_score += ((observed - expected) ** 2.0) / expected

        return chi_score

    @staticmethod
    def report_chi_score(entries, model_name, expert_name, _type):
        # Requires 2 entries in the entries list:
        # 1) The Agent
        # 2) The Expert
        scores_file_name = 'chi_scores.csv'

        report_dir = 'report'

        scores_path = os.path.join(report_dir, scores_file_name)

        # Create directories in the model path if they do not exist.
        if not os.path.exists(os.path.dirname(scores_path)):
            try:
                os.makedirs(report_dir)
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        chi_score = Stats.get_chi_score(entries)
        score_name = Stats.CHI_SCORE_KEY_FMT.format(_type,
                                                    model_name,
                                                    expert_name)
        new_entry = {
            Stats.MODEL: score_name,
            'chi_score': chi_score,
        }

        # Check if the file exists to not overwrite previous entries.
        curr_entries = list()
        if os.path.isfile(scores_path):
            curr_entries = Stats._read(scores_path)

            # Filter the list to have no records of the current model
            curr_entries = [entry for entry in curr_entries
                            if entry[Stats.MODEL] != score_name]

        curr_entries.append(new_entry)

        # Write the stats to the stats file.
        with open(scores_path, 'w', newline='') as scores_file:
            field_names = sorted(list(new_entry.keys()))

            scores_writer = csv.DictWriter(scores_file, fieldnames=field_names)
            scores_writer.writeheader()

            for entry in curr_entries:
                scores_writer.writerow(entry)

        return chi_score

    @staticmethod
    def get_monte_carlo_distance(agent_traj, expert_traj):
        state_space = set([pair['state'] for pair in agent_traj])
        state_space = state_space.union(set([pair['state'] for pair in expert_traj]))

        action_space = set([pair['action'] for pair in agent_traj])
        action_space = action_space.union(set([pair['action'] for pair in expert_traj]))

        running_log_sum = 0
        for agent_pair in agent_traj:
            running_sum = 0

            for expert_pair in expert_traj:
                if (agent_pair['action'] == expert_pair['action']
                        and agent_pair['state'] == expert_pair['state']):
                    running_sum += 1

            term = (running_sum + 1) / float(len(expert_traj) + len(state_space) * len(action_space))

            log_result = math.log(term)

            running_log_sum += log_result

        return (-1 * running_log_sum) / len(agent_traj)

    @staticmethod
    def report_monte_carlo_distance(agent_traj, expert_traj,
                                    agent_name, expert_name):
        scores_file_name = 'monte_carlo_dist.csv'

        report_dir = 'report'

        scores_path = os.path.join(report_dir, scores_file_name)

        # Create directories in the model path if they do not exist.
        if not os.path.exists(os.path.dirname(scores_path)):
            try:
                os.makedirs(report_dir)
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        monte_carlo_dist = Stats.get_monte_carlo_distance(agent_traj, expert_traj)
        score_name = Stats.MONTE_CARLO_DIST_KEY_FMT.format(agent_name,
                                                           expert_name)
        new_entry = {
            Stats.MODEL: score_name,
            'monte_carlo_dist': monte_carlo_dist,
        }

        # Check if the file exists to not overwrite previous entries.
        curr_entries = list()
        if os.path.isfile(scores_path):
            curr_entries = Stats._read(scores_path)

            # Filter the list to have no records of the current model
            curr_entries = [entry for entry in curr_entries
                            if entry[Stats.MODEL] != score_name]

        curr_entries.append(new_entry)

        # Write the stats to the stats file.
        with open(scores_path, 'w', newline='') as scores_file:
            field_names = sorted(list(new_entry.keys()))

            scores_writer = csv.DictWriter(scores_file, fieldnames=field_names)
            scores_writer.writeheader()

            for entry in curr_entries:
                scores_writer.writerow(entry)

        return monte_carlo_dist

    @staticmethod
    def report_pie_charts(entries, _type):
        PIE_CHART_FILE_FMT = '{}_{}_piechart.png'
        PIE_CHART_TITLE_FMT = '{} {} stats'

        report_dir = 'report'

        def get_frequency(entry, key):
            # Returns frequency if a value exists,
            # otherwise return 0 for the key.
            try:
                return int(entry[key])
            except BaseException:
                return 0

        for entry in entries:
            model_name = entry.pop(Stats.MODEL)

            labels = [field for field in entry]

            fig, axl = pyplot.subplots(figsize=(48, 24))

            axl.pie([get_frequency(entry, field) for field in entry],
                    explode=[0.1 for field in entry],
                    autopct='%1.2f%%', pctdistance=1.1, textprops={'fontsize': 24})
            pyplot.title(PIE_CHART_TITLE_FMT.format(model_name, _type),
                         fontsize=24)
            axl.legend(labels, loc='upper right', fontsize=24)
            pyplot.tight_layout()

            pie_chart_path = os.path.join(report_dir,
                                          PIE_CHART_FILE_FMT.format(model_name, _type))
            pyplot.savefig(pie_chart_path,
                           bbox_inches='tight')
            pyplot.clf()
