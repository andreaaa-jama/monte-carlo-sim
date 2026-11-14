import numpy as np

def generate_z_matrix(n_paths, n_steps, seed=None):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_paths, n_steps))


def simulate_gbm(S0, mu, sigma, T, n_steps, n_paths, seed=None):
    dT = T/ n_steps
    Z = generate_z_matrix(n_paths, n_steps, seed=seed)

    log_increments = ( mu - (0.5 * sigma**2)) * dT + sigma * np.sqrt(dT) * Z
    log_paths = np.cumsum(log_increments, axis=1)
    paths = S0 * np.exp(log_paths)

    start = np.full((n_paths, 1), S0)
    return np.hstack((start, paths))



    

    

    