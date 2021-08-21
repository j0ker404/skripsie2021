from alphaslime.evaluate.eval_agents import EvaluateGame
from alphaslime.agents.baseline import BaselineAgent

if __name__ == '__main__':
    agent_right = BaselineAgent()
    agent_left = BaselineAgent()
    base_dir = './'
    RENDER = True
    env_id = "SlimeVolley-v0"
    # env_id = "SlimeVolleyPixel-v0"
    eval_game = EvaluateGame(agent_right=agent_right, agent_left=agent_left, base_dir_path=base_dir, render=RENDER, env_id=env_id)

    # run N episodes
    N = 1
    print('Start running episodes..')
    for n in range(N):
        print('Evaulating episode {}'.format(n))
        eval_game.evaluate_episode()