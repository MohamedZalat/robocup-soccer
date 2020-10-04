#!/usr/bin/env python
# --------------------------------------------------------------------------------
#
# Use this script to calculate the chi score
#
# --------------------------------------------------------------------------------
import argparse
import os

from aigent.soccerpy.stats import Stats

# --------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--type', help='set to "environment" if reporting'
                        ' environment stats, otherwise set to "action"',
                        required=True)
    parser.add_argument('-f', '--file', help='the file that contains the'
                        ' frequency records of the environment states/actions',
                        required=True)

    parser.add_argument('-r', '--report', help='set to "chi" to report chi scores'
                        '\nset to "pie" to report pie charts\n'
                        'set to "monte" to report monte carlo distance',
                        required=True)

    parser.add_argument('-a', '--agent', help='set to the agent\'s model name,'
                        ' must give it a value if using a chi report or '
                        'a monte carlo distance report')
    parser.add_argument('-e', '--expert', help='set to the agent\'s model name,'
                        ' must give it a value if using a chi report or '
                        'a monte carlo distance report')

    args = parser.parse_args()
    _type = args.type
    stats_path = args.file

    if args.report == 'chi':
        if not args.agent or not args.expert:
            print('Need to include agent and expert model name in chi report,'
                  ' under options --agent and --expert')
            return
        agent_model = args.agent
        expert_model = args.expert
        entries = Stats.read(stats_path)

        report_chi(entries, agent_model, expert_model, _type)

    elif args.report == 'pie':
        entries = Stats.read(stats_path)
        report_pie_charts(entries, _type)

    elif args.report == 'monte':
        if not args.agent or not args.expert:
            print('Need to include agent and expert model name in monte report,'
                  ' under options --agent and --expert')
            return
        agent_model = args.agent
        expert_model = args.expert

        report_monte_carlo_distance(agent_model, expert_model)

# --------------------------------------------------------------------------------
def report_chi(entries, agent_model, expert_model, _type):
    print(('Calculating the chi score for model {} and'
           ' expert {}').format(agent_model, expert_model))
    entries = [entry for entry in entries if entry[Stats.MODEL] in [agent_model,
                                                                    expert_model]]

    chi_score = Stats.report_chi_score(entries, agent_model, expert_model, _type)

    print('chi_score = {}'.format(chi_score))

# --------------------------------------------------------------------------------
def report_pie_charts(entries, _type):
    Stats.report_pie_charts(entries, _type)
    print('reported pie charts in the report file')

# --------------------------------------------------------------------------------
def report_monte_carlo_distance(agent_model, expert_model):
    report_dir = 'report'
    agent_traj = Stats.read(os.path.join(report_dir,
                                         Stats.TRAJ_FILE_NAME_FMT
                                         .format(agent_model)))
    expert_traj = Stats.read(os.path.join(report_dir,
                                          Stats.TRAJ_FILE_NAME_FMT
                                          .format(expert_model)))
    monte_dist = Stats.report_monte_carlo_distance(agent_traj, expert_traj,
                                                   agent_model, expert_model)
    print('monte_carlo_dist = {}'.format(monte_dist))

# --------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
