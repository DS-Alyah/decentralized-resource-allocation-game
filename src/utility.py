import numpy as np


def utility(x, a, c):
    """
    Calculate the utility of an agent.

    Parameters
    ----------
    x : float
        Amount of resource allocated to the agent.

    a : float
        Agent's value parameter.

    c : float
        Agent's cost parameter.

    Returns
    -------
    float
        Utility of the agent.
    """

    return a * np.log(1 + x) - c * x




if __name__ == "__main__":
    result = utility(
        x=20,
        a=5.0,
        c=0.10
    )

    print("Utility:", result)