import numpy as np
from scipy.stats import norm

def simulate_normal_mix(n, p1, m1, m2, sd1, sd2):
    """Simulate data from a mixture of two normal distributions."""
    # Simulate component labels
    labels = np.random.choice([0, 1], size=n, p=[p1, 1-p1])
    # Simulate data from the normal distributions
    x = np.where(labels == 0, 
                 np.random.normal(m1, sd1, size=n), 
                 np.random.normal(m2, sd2, size=n))    
    return x

def fit_normal_mix(x, p1, m1, m2, sd1, sd2, niter=100):
    """
    Fit a univariate mixture of two normal distributions using the EM algorithm.

    Parameters:
    x (numpy.ndarray): 1-D array of data points.
    p1 (float): Initial guess for the probability of the first normal component.
    m1 (float): Initial guess for the mean of the first normal component.
    m2 (float): Initial guess for the mean of the second normal component.
    sd1 (float): Initial guess for the standard deviation of the first normal component.
    sd2 (float): Initial guess for the standard deviation of the second normal component.
    niter (int): Number of iterations for the EM algorithm (default is 100).

    Returns:
    tuple: Optimal values of (p1, m1, m2, sd1, sd2) after fitting.
    """
    n = len(x)    
    for i in range(niter):
        # E-step: compute the responsibilities
        r1 = p1 * norm.pdf(x, m1, sd1)
        r2 = (1 - p1) * norm.pdf(x, m2, sd2)
        w1 = r1 / (r1 + r2)
        w2 = r2 / (r1 + r2)
        # M-step: update the parameters
        p1 = np.sum(w1) / n
        m1 = np.sum(w1 * x) / np.sum(w1)
        m2 = np.sum(w2 * x) / np.sum(w2)
        sd1 = np.sqrt(np.sum(w1 * (x - m1)**2) / np.sum(w1))
        sd2 = np.sqrt(np.sum(w2 * (x - m2)**2) / np.sum(w2))
    return p1, m1, m2, sd1, sd2