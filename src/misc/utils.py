import numpy as np

def clip(input, min_bound, max_bound, out_type = int):
    assert (min_bound < max_bound).all(), "lower and upper volume boundary mismatching"
    return np.minimum(max_bound, np.maximum(input, min_bound))