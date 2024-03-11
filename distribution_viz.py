import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm, skewnorm, gamma, expon, beta, norm, poisson


def generate_distribution(distribution_type, params, size=1000):
    """
    Generates and plots a specified distribution given parameters and size.

    Args:
        distribution_type (str): Type of distribution to generate. Options: 'log-normal', 'skew-normal', 'gamma', 'exponential', 'beta'.
        params (dict): Parameters needed to generate the distribution. The keys should match the expected parameters for the scipy.stats distribution function.
        size (int): The number of random variables to generate.

    Returns:
        np.ndarray: An array of generated random variables.
    """
    if distribution_type == "log-normal":
        sigma = params.get("sigma", 1)
        data = lognorm.rvs(s=sigma, size=size)
    elif distribution_type == "poisson":
        mu = params.get("mu", 3)
        data = poisson.rvs(mu=mu, size=size)
    elif distribution_type == "skew-normal":
        alpha = params.get("alpha", 0)
        loc = params.get("loc", 0)
        scale = params.get("scale", 1)
        data = skewnorm.rvs(a=alpha, loc=loc, scale=scale, size=size)
    elif distribution_type == "gamma":
        a = params.get("a", 1)  # shape
        data = gamma.rvs(a=a, size=size)
    elif distribution_type == "exponential":
        scale = params.get("scale", 1)
        data = expon.rvs(scale=scale, size=size)
    elif distribution_type == "beta":
        a = params.get("a", 0.5)
        b = params.get("b", 0.5)
        data = beta.rvs(a=a, b=b, size=size)
    elif distribution_type == "normal":
        mu = params.get("mu", 5)
        sigma = params.get("sigma", 2)
        data = norm.rvs(loc=mu, scale=sigma, size=size)
    else:
        raise ValueError(
            "Unsupported distribution type. Choose from 'log-normal', 'skew-normal', 'gamma', 'exponential', 'beta'."
        )

    # Plotting
    plt.hist(data, bins=30, density=True, alpha=0.6)
    plt.title(f"{distribution_type.capitalize()} Distribution")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()

    return data


# Example usage:
data = generate_distribution("normal", {"mu": 4, "sigma": 2}, size=1000)
