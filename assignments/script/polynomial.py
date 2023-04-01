import numpy as np 


def evaluate_function_on_random_noise(N: int=15, mu: float=0, sigma: float=0, x_min: float=-1, x_max: float=3) -> tuple[np.array, np.array]:
    """Evaluate function y = x^2 - 3x + 1 on N points uniformly sampled from interval [x_min, x_max],
        with random Gaussian noise N(mu, sigma^2)

    Parameters
    N: sample size 
    mu: mean 
    sigma: standard deviation 
    x_min: lower bound of interval 
    x_max: upper bound of interval

    Returns
    x: (N, ) 
    y: (N, )
    """
    x = np.linspace(x_min, x_max, N) 
    noise = np.random.normal(mu, sigma, N)
    y = x**2 - 3*x + 1 + noise
    
    return x, y 


def fit_polynomial(x: np.array, y: np.array, degree: int=1, lbda: float = 0) -> tuple[np.array, np.array, np.array]:
    """Polynomial regression

    Parameters
    ----------
    x: (N, )
    y: (N, )
    lbda: lambda regularization level

    Returns
    coeffs: estimated coefficients (degree+1)
    haty: predicted y (N, )
    MSE: mean squared error, a scalar
    """
    X = np.zeros((x.shape[0], degree+1))
    for i in range(degree+1):
        X[:,i] = x**i 
    # Moore-Penrose Pseudoinverse
    coeffs = np.linalg.inv(lbda*np.identity(degree+1) + X.T@X)@X.T@y 

    haty = coeffs.T@X.T
    mse = np.mean((haty - y)**2)
    return coeffs, haty, mse