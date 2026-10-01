import numpy as np
from scipy.optimize import minimize

from src.utility import utility


def centralized_optimum(a, c, resource):
    """
    Find the resource allocation that maximizes
    total social welfare.
    """

    n_agents = len(a)

    def objective(x):
        total_utility = sum(
            utility(x[i], a[i], c[i])
            for i in range(n_agents)
        )

        return -total_utility

    constraints = {
        "type": "ineq",
        "fun": lambda x: resource - np.sum(x)
    }

    bounds = [
        (0, resource)
        for _ in range(n_agents)
    ]

    initial_guess = np.full(
        n_agents,
        resource / n_agents
    )

    result = minimize(
        objective,
        initial_guess,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    if not result.success:
        raise RuntimeError(result.message)

    return result.x


if __name__ == "__main__":
    a = np.array([5.0, 4.0, 6.0, 3.0, 5.0])

    c = np.array([0.10, 0.20, 0.15, 0.10, 0.20])

    resource = 100

    allocation = centralized_optimum(
        a=a,
        c=c,
        resource=resource
    )

    print("Centralized allocation:")
    print(allocation)

    print("\nTotal allocated:")
    print(np.sum(allocation))

    total_welfare = sum(
        utility(allocation[i], a[i], c[i])
        for i in range(len(a))
    )

    print("\nTotal social welfare:")
    print(total_welfare)

