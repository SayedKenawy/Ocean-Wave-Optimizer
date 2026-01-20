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

### Initialization

The initialization equation defines how each search agent (wave) is uniformly distributed within the bounded search space. This guarantees unbiased sampling and adequate coverage at the start of the optimization process.

```math
X_i = lb + rand(0,1) \times (ub - lb), \quad i = 1, 2, \ldots, N
```



### Fitness Evaluation

Each wave’s position is evaluated using the objective function, transforming a multidimensional solution into a scalar fitness value.

```math
f_i = objf(X_i)
```

The dominant wave represents the best solution found so far and guides the population movement.

```math
X_{best} = \arg\min_i f_i
```



### Exploration Phase

Global exploration is achieved by introducing stochastic perturbations that allow waves to explore unexplored regions of the search space.

```math
X_i^{new} = X_i + \alpha \cdot (rand(0,1) - 0.5) \times (ub - lb)
```

To prevent stagnation, complete random redistribution may also be applied.

```math
X_i^{new} = lb + rand(0,1) \times (ub - lb)
```


### Exploitation Phase

Local exploitation pulls waves toward the dominant solution, refining candidate solutions around promising regions.

```math
X_i^{new} = X_{best} + k \cdot rand(0,1) \times (X_{best} - X_i)
```

A nonlinear reflection strategy enhances fine-grained local adjustments.

```math
X_i^{new} = X_{best} + (k - rand(0,1)) \cdot X_{best}
```



### Adaptive Control Parameter

The parameter controlling exploration and exploitation decays nonlinearly to ensure smooth convergence.

```math
k = a - a \left( \frac{t^2}{T^2} \right)
```



### Convergence Criterion

The best fitness value at each iteration is monitored to evaluate convergence behavior.

```math
f_{best}^{(t)} = \min_i f_i^{(t)}
```


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
