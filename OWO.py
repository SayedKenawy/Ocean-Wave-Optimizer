"""
Ocean-Wave-Optimizer(OWO)
Inspired by the behavior of Tardigrades
Author: Ahmed Mohamed Zaki & El-Sayed M. El-kenawy
"""

import time
import numpy
import random

class solution:
    def __init__(self):
        self.startTime = None
        self.endTime = None
        self.executionTime = None
        self.convergence = None
        self.optimizer = None
        self.objfname = None

def OWO(objf, lb, ub, dim, SearchAgents_no, Max_iter):
    """
    Ocean Wave Optimizer (OWO) inspired by dominant wave dynamics in ocean wave groups.
    Parameters:
    - objf: Objective function to optimize (fitness function)
    - lb: Lower bounds (list or scalar)
    - ub: Upper bounds (list or scalar)
    - dim: Number of dimensions
    - SearchAgents_no: Number of waves (population size)
    - Max_iter: Maximum number of iterations
    """
    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim
    # Initialize wave positions randomly within bounds
    Positions = numpy.zeros((SearchAgents_no, dim))
    for i in range(dim):
        Positions[:, i] = numpy.random.uniform(0, 1, SearchAgents_no) * (ub[i] - lb[i]) + lb[i]
    
    Convergence_curve = numpy.zeros(Max_iter)
    s = solution()
    print('OWO is optimizing "' + objf.__name__ + '"')
    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    dominant_wave = numpy.zeros(dim)
    dominant_fitness = float("inf")
    dominant_index = 0

    prev_fit = float("inf")
    prev_fit2 = float("inf")

    for t in range(Max_iter):
        for i in range(SearchAgents_no):
            # Enforce boundaries on each dimension
            for j in range(dim):
                Positions[i, j] = numpy.clip(Positions[i, j], lb[j], ub[j])
            # Calculate fitness
            fitness = objf(Positions[i, :])
            # Update dominant wave if better solution found
            if fitness < dominant_fitness:
                dominant_fitness = fitness
                dominant_wave = Positions[i, :].copy()
                dominant_index = i

        # Dynamically calculate exploration and exploitation counts
        a = 2
        k = a - a * t**2 / Max_iter**2
        exploration_percent = 70
        min_exploration = exploration_percent - (100 - exploration_percent)
        exploration_count = int(SearchAgents_no * (exploration_percent - min_exploration * t / Max_iter) / 100)
        exploitation_count = SearchAgents_no - exploration_count

        # Reset exploration/exploitation if stagnating
        if (prev_fit == prev_fit2) and (dominant_fitness == prev_fit):
            exploration_count = int(SearchAgents_no * exploration_percent / 100)
            exploitation_count = SearchAgents_no - exploration_count
            k = a

        # Exploration phase: random mutations (interference/new waves)
        for w in range(exploration_count):
            if dominant_index == w:
                continue
            if w % 2 == 0:
                for d in range(dim):
                    rn = random.uniform(0, 1)
                    if 0.5 >= rn:
                        Positions[w, d] = random.uniform(lb[d], ub[d])
            else:
                r1 = random.uniform(0, 2)
                r2 = random.uniform(0, 1)
                for d in range(dim):
                    dist = Positions[w, d] * r1 - r1
                    Positions[w, d] += dist * (2 * r2 - 1)

        # Exploitation phase: waves follow the dominant wave
        for i in range(exploitation_count):
            w = exploration_count + i
            if dominant_index == w:
                continue
            if w % 2 == 0:
                r3 = random.uniform(0, 2)
                for d in range(dim):
                    dist = r3 * (dominant_wave[d] - Positions[w, d])
                    Positions[w, d] += dist
            else:
                r4 = random.uniform(0, 1)
                r5 = random.uniform(0, 1)
                for d in range(dim):
                    dist = dominant_wave[d] * (k - r4)
                    Positions[w, d] = dominant_wave[d] + dist * (2 * r5 - 1)

        prev_fit2 = prev_fit
        prev_fit = dominant_fitness
        numpy.random.shuffle(Positions)

        Convergence_curve[t] = dominant_fitness
        if t % 1 == 0:
            print(["At iteration " + str(t) + " the best fitness is " + str(dominant_fitness)])

    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = Convergence_curve
    s.optimizer = "OWO"
    s.objfname = objf.__name__

    return s
