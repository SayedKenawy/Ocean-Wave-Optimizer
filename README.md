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

## Algorithm Mechanics

### Exploration Phase (70% initially, decreasing)

- Generates random wave patterns through interference
- Creates new waves in random positions
- Maintains diversity in the population

### Exploitation Phase (30% initially, increasing)

- Waves move toward the dominant wave
- Refines solutions around the best-found position
- Converges to optimal solution

### Stagnation Recovery

When fitness remains unchanged for consecutive iterations, the algorithm:

- Resets exploration percentage to 70%
- Increases parameter `k` to expand search
- Prevents premature convergence

## Solution Object

The algorithm returns a `solution` object with the following attributes:

- `convergence`: Array of best fitness values per iteration
- `optimizer`: Name of the optimizer ("OWO")
- `objfname`: Name of the objective function
- `startTime`: Timestamp when optimization started
- `endTime`: Timestamp when optimization ended
- `executionTime`: Total execution time in seconds

## Example Benchmark Functions

```python
# Sphere function
def sphere(x):
    return np.sum(x**2)

# Rastrigin function
def rastrigin(x):
    return 10*len(x) + np.sum(x**2 - 10*np.cos(2*np.pi*x))

# Rosenbrock function
def rosenbrock(x):
    return np.sum(100*(x[1:]-x[:-1]**2)**2 + (1-x[:-1])**2)
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
