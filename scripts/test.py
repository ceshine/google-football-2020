# Set up the Environment.
import typer
import numpy as np
from kaggle_environments import make, utils


def main(runs: int):
    # submission = utils.read_file("rf_sub.py")
    env = make("football", configuration={
        "save_video": True,
        "scenario_name": "11_vs_11_hard_stochastic",
        # "scenario_name": "11_vs_11_easy_stochastic",
        # "scenario_name": "11_vs_11_stochastic",
        "running_in_notebook": False,
        "dump_full_episodes": True, "logdir": "../cache/runs/", "render": True
    }, debug=True)

    rewards = []
    for _ in range(runs):
        # output = env.run(["main.py", "do_nothing"])[-1]
        # output = env.run(["main.py", "run_right"])[-1]
        output = env.run(["main.py", "builtin_ai"])[-1]
        # output = env.run(["main.py", "main.py"])[-1]
        print('Left player: reward = %s, status = %s, info = %s' %
              (output[0]['reward'], output[0]['status'], output[0]['info']))
        rewards.append(output[0]['reward'])
    # print('Right player: reward = %s, status = %s, info = %s' %
    #       (output[1]['reward'], output[1]['status'], output[1]['info']))
    # env.render(mode="human", width=800, height=600)
    print(rewards)
    print(np.mean(rewards))


if __name__ == "__main__":
    typer.run(main)
