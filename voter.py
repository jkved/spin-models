import numpy as np


def initialize_state(grid_size, rng):
    """Initialize voter model with an aligned spin grid.

    Input:
        grid_size:
            Length of the side of a square lattice.
        rng:
            numpy Random Generator (or compatible) object.

    Output:
        Two dimensional array of [grid_size x grid_size] size filled with
        all values either 1 or -1.
    """
    if rng.random() < 0.5:
        return -np.ones((grid_size, grid_size))
    return np.ones((grid_size, grid_size))


def _get_neighbor(state, pos, rng):
    """Get a state of random neighbor from four cardinal neighbors."""
    grid_size = state.shape[0]

    neigh = pos.copy()
    # random cardinal direction
    axis = [0, 1][rng.random() < 0.5]
    dir = [-1, 1][rng.random() < 0.5]
    # pick wraping around the border
    neigh[axis] = (neigh[axis] + dir) % grid_size

    return state[neigh[0], neigh[1]]


def update_state_voter(state, prob, rng):
    """Update state according to noisy voter model.

    Input:
        state:
            Square [N x N] array encoding the state of the Ising model.
        prob:
            Probability that selected agent will act randomly instead
            of immitating.
        rng:
            numpy Random Generator (or compatible) object.

    Output:
        Updated state of the model (square [N x N] array) after N**2 MC
        steps.
    """
    grid_size = state.shape[0]
    n_steps = grid_size**2

    for _ in range(n_steps):
        pos = rng.integers(0, grid_size, size=2)

        if rng.random() < prob:
            state[pos[0], pos[1]] = [-1, 1][rng.random() < 0.5]
        else:
            state[pos[0], pos[1]] = _get_neighbor(state, pos, rng)

    return state
