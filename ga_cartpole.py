import gym
import numpy as np
import torch
import torch.nn as nn
from typing import List
import random

POP_SIZE = 100
CROSS_RATE = 0.8
MUTATION_RATE = 0.01
MUTATION_FACTOR = 0.001
N_GENERATIONS = 20
MAX_GLOBAL_STEPS = 500000

torch.set_grad_enabled(False)

env = gym.make('CartPole-v1')
env = gym.wrappers.RecordEpisodeStatistics(env)

env.seed(1)
env.action_space.seed(1)
env.observation_space.seed(1)
obs = env.reset()

in_dim = len(obs)
out_dim = env.action_space.n
global_step = 0

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Pytorch will use device {device}")


def get_params(net):
    params = []
    for layer in net:
        if hasattr(layer, 'weight'):
            params.append(layer.weight)
        if hasattr(layer, 'bias'):
            params.append(layer.bias)
    return params

def set_params(net, params):
    i = 0
    for layerid, layer in enumerate(net):
        if hasattr(layer, 'weight'):
            net[layerid].weight = params[i]
            i += 1
        if hasattr(layer, 'bias'):
            net[layerid].bias = params[i]
            i += 1
    return net

def fitness(solution, net):
    global global_step
    net = set_params(net, solution)
    ob = env.reset()
    done = False
    while not done:
        ob = torch.tensor(ob).float().unsqueeze(0).to(device)
        q_vals = net(ob)
        act = torch.argmax(q_vals.cpu()).item()
        ob_next, _, done, info = env.step(act)
        global_step+=1
        ob = ob_next
        if 'episode' in info.keys():
            print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
            return info['episode']['r']

def select(pop, fitnesses):
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True, p=fitnesses/fitnesses.sum())
    return [pop[i] for i in idx]

def crossover(parent1, pop):
    if np.random.rand() < CROSS_RATE:
        i = np.random.randint(0, POP_SIZE, size=1)[0]
        parent2 = pop[i]
        child = []
        for p1l, p2l in zip(parent1, parent2):
            split = np.random.randint(0, len(p1l), size=1)[0]
            new_param = nn.parameter.Parameter(torch.cat([p1l[:split], p2l[split:]]))
            child.append(new_param)

        return child
    else:
        return parent1
    
def mutate(child):
    for i in range(len(child)):
        for j in range(len(child[i])):
            child[i][j] += torch.randn(child[i][j].shape).to(device)*MUTATION_FACTOR
    return child

def init_pop(net):
    base = get_params(net)
    shapes = [param.shape for param in base]
    print(shapes)
    pop = []
    for _ in range(POP_SIZE):
        entity = []
        for shape in shapes:
            try:
                rand_tensor = nn.init.kaiming_uniform_(torch.empty(shape)).to(device)
            except ValueError:
                rand_tensor = nn.init.uniform_(torch.empty(shape), -0.5, 0.5).to(device)
            entity.append((torch.nn.parameter.Parameter(rand_tensor)))
        pop.append(entity)
    return pop


if __name__ == "__main__":
    net = nn.Sequential(nn.Linear(in_dim, 64),
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, out_dim)).to(device)
    pop = init_pop(net)

    for i in range(N_GENERATIONS):
        fitnesses = np.array([fitness(entity, net) for entity in pop])
        
        fittest = np.argmax(fitnesses)
        agent = set_params(net, pop[fittest])
        avg_fitness = fitnesses.sum()/len(fitnesses)
        
        print(f"Generation {i}: Average Fitness is {avg_fitness} | Max Fitness is {fitnesses.max()}")
        pop = select(pop, fitnesses)
        pop2 = list(pop)
        for i in range(len(pop)):
            child = crossover(pop[i], pop2)
            child = mutate(child)
            pop[i] = child
        if global_step>MAX_GLOBAL_STEPS:
            break
