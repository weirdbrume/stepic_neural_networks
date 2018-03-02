# from HW_3.cars import *
from cars.world import SimpleCarWorld
from cars.agent import SimpleCarAgent
from cars.physics import SimplePhysics
from cars.track import generate_map
import numpy as np
import random


def train_agent(agent_name, maps, steps_on_map):
    for map in maps:
        seed = map
        np.random.seed(seed)
        random.seed(seed)
        m = generate_map(8, 5, 3, 3)
        w = SimpleCarWorld(1, m, SimplePhysics, SimpleCarAgent, timedelta=0.2)
        agent = SimpleCarAgent.from_file(agent_name)
        w.set_agents([agent])
        w.run(steps_on_map)


def exam_agent(agent_name, maps, steps_on_map):
    rewards = []
    for map in maps:
        seed = map
        np.random.seed(seed)
        random.seed(seed)
        m = generate_map(8, 5, 3, 3)
        w = SimpleCarWorld(1, m, SimplePhysics, SimpleCarAgent, timedelta=0.2)
        agent = SimpleCarAgent.from_file(agent_name)
        reward = w.evaluate_agent(agent, steps_on_map)
        print('оценка на карте {0} равна {1}'.format(map, reward))
        rewards.append(reward)
    return rewards


# код по умолчанию - расскоментить для использования
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--steps", type=int)
parser.add_argument("-f", "--filename", type=str)
parser.add_argument("-e", "--evaluate", type=bool)
parser.add_argument("--seed", type=int)
args = parser.parse_args()

print(args.steps, args.seed, args.filename, args.evaluate)

steps = args.steps
seed = args.seed if args.seed else 23
np.random.seed(seed)
random.seed(seed)
m = generate_map(8, 5, 3, 3)

if args.filename:
    agent = SimpleCarAgent.from_file(args.filename)
    w = SimpleCarWorld(1, m, SimplePhysics, SimpleCarAgent, timedelta=0.2)
    if args.evaluate:
        print(w.evaluate_agent(agent, steps))
    else:
        w.set_agents([agent])
        w.run(steps)
else:
    SimpleCarWorld(1, m, SimplePhysics, SimpleCarAgent, timedelta=0.2).run(steps)

# мой код - расскоментить для использования
"""steps = 800  # здесь и далее в комментарии указаны значения по умолчанию 800
# норм мапы :) [5, 9, 15, 18, 21, 22]
maps = [3, 13, 23]  # [3, 13, 23]
filename = 'network_config_agent_0_layers_11_6_3_1.txt'
training_sessions = 10

for i in range(training_sessions):
    train_agent(filename, maps, steps)

    agent_scores = exam_agent(filename, maps, steps)
    mean_agent_score = sum(agent_scores) / len(agent_scores)
    print('средняя оценка за один проход по всем картам', mean_agent_score)

    f = open('scores.txt', 'a')
    f.write(str(mean_agent_score) + '\n')
    f.close()
"""