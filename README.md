# Ocean Wave Optimizer (OWO)

A Python implementation of the Ocean Wave Optimizer (OWO), a nature-inspired metaheuristic optimization algorithm based on the dynamics of dominant waves in ocean wave groups.

## Overview

The Ocean Wave Optimizer mimics the physical behavior of ocean waves to solve continuous optimization problems. The algorithm balances exploration (wave interference and random perturbations) with exploitation (waves following the dominant wave pattern) to efficiently search the solution space.

## Features

- **Dynamic exploration-exploitation balance**: Automatically adjusts the ratio based on iteration progress
- **Stagnation detection**: Detects when the algorithm is stuck and resets exploration parameters
- **Dominant wave tracking**: Maintains the best solution (dominant wave) found during optimization
- **Flexible boundary handling**: Supports different bounds for each dimension
- **Convergence tracking**: Records fitness values across iterations for analysis
- **Execution time monitoring**: Tracks start time, end time, and total execution duration

## Installation

### Requirements

```bash
pip install numpy
```

### Clone Repository

```bash
git clone https://github.com/SayedKenawy/ocean-wave-optimizer.git
cd ocean-wave-optimizer
```

## Usage

```python
from OWO import OWO
import numpy as np

# Define your objective function (minimization)
def sphere_function(x):
    return np.sum(x**2)

# Set parameters
dim = 10                    # Problem dimensions
SearchAgents_no = 30        # Population size (number of waves)
Max_iter = 500              # Maximum iterations
lb = -100                   # Lower bound
ub = 100                    # Upper bound

# Run optimizer
solution = OWO(sphere_function, lb, ub, dim, SearchAgents_no, Max_iter)

# Access results
print(f"Best fitness: {solution.convergence[-1]}")
print(f"Execution time: {solution.executionTime:.4f} seconds")
print(f"Convergence curve: {solution.convergence}")
```

## Parameters

| Parameter         | Type          | Description                         |
| ----------------- | ------------- | ----------------------------------- |
| `objf`            | function      | Objective function to minimize      |
| `lb`              | float or list | Lower bound(s) for search space     |
| `ub`              | float or list | Upper bound(s) for search space     |
| `dim`             | int           | Number of dimensions in the problem |
| `SearchAgents_no` | int           | Population size (number of waves)   |
| `Max_iter`        | int           | Maximum number of iterations        |

## Mathematical Equations

### 1. Initialization Equation

The initialization model
[
X_i = lb + rand(0,1) \times (ub - lb)
]
defines the **initial spatial distribution of waves** (search agents) within the feasible search domain. Each component of the position vector is sampled uniformly between the lower bound (lb) and upper bound (ub), ensuring unbiased coverage of the search space at the start of the optimization process. This mechanism is standard in stochastic optimization and prevents premature bias toward any region of the solution space.

Such random initialization promotes **global exploration**, a fundamental requirement for avoiding early convergence to local optima in nonlinear or multimodal problems .

---

### 2. Fitness Evaluation and Dominant Wave Selection

The fitness computation
[
f_i = objf(X_i)
]
maps each wave’s position to a scalar performance value using the objective function. The dominant (or leading) wave is then identified as
[
X_{best} = \arg\min_i f_i
]
for minimization problems.

This step establishes a **leader–follower dynamic**, where the best-performing wave represents the most promising solution discovered so far. Similar leader-based mechanisms are widely used in swarm intelligence algorithms to guide population movement toward optimal regions .

---

### 3. Exploration Phase (Global Search)

The exploration update rule
[
X_i^{new} = X_i + \alpha \cdot (rand(0,1) - 0.5) \times (ub - lb)
]
introduces controlled random perturbations around the current wave position. The scaling factor (\alpha) regulates step size, while the term ((rand - 0.5)) ensures symmetric movement in positive and negative directions.

An alternative exploration strategy is full random reinitialization:
[
X_i^{new} = lb + rand(0,1) \times (ub - lb)
]
which helps the algorithm **escape stagnation** by injecting diversity when the population becomes overly concentrated. This mechanism is commonly employed in evolutionary computation to counteract loss of diversity .

---

### 4. Exploitation Phase (Local Search)

During exploitation, waves are attracted toward the dominant wave using:
[
X_i^{new} = X_{best} + k \cdot rand(0,1) \times (X_{best} - X_i)
]
This equation reduces the distance between subordinate waves and the best solution, intensifying the search locally around high-quality regions. The stochastic multiplier preserves variability while maintaining directional bias.

The nonlinear reflection variant
[
X_i^{new} = X_{best} + (k - rand(0,1)) \cdot X_{best}
]
models oscillatory wave behavior and allows fine-grained adjustments near the optimum, enhancing convergence precision. Such nonlinear local search mechanisms are known to improve exploitation efficiency in swarm-based optimizers .

---

### 5. Adaptive Control Parameter (k)

The parameter update rule
[
k = a - a \left( \frac{t^2}{T^2} \right)
]
implements a **nonlinear decay schedule**, where (k) decreases quadratically as iterations progress. Early iterations favor exploration (large (k)), while later iterations emphasize exploitation (small (k)).

Nonlinear parameter control has been shown to outperform linear schedules by providing smoother transitions between global and local search phases, particularly in complex optimization landscapes .

---

### 6. Convergence Monitoring

Finally, convergence is tracked using:
[
f_{best}^{(t)} = \min_i f_i^{(t)}
]
which records the best fitness value at each iteration. This metric is used to assess algorithmic progress and determine termination when improvements fall below a predefined tolerance or when the maximum iteration count is reached.

Monitoring best-so-far fitness is a standard convergence criterion in metaheuristic optimization and provides insight into both stability and performance trends .




## Applications

OWO can be applied to various optimization problems including:

- Hyperparameter tuning for machine learning models
- Feature selection
- Engineering design optimization
- Resource allocation problems
- Function approximation
- Neural network training

## Performance Tips

- Start with `SearchAgents_no = 30-50` for most problems
- Increase iterations for complex, high-dimensional problems
- Use appropriate bounds based on problem domain knowledge
- Monitor convergence curves to assess performance

## License

MIT License - feel free to use and modify for your research and applications.
