import numpy as np


def initialize_state(grid_size, rng):
    """Initialize Ising model with an aligned spin grid.

    Input:
        grid_size:
            Length of the side of a square lattice.
        rng:
            numpy Random Generator (or compatible) object.

    Output:
        Two dimensional array of [grid_size x grid_size] size filled with
        all values either 1 or -1.
    """
    #if rng.random() < 0.5:
     #   return -np.ones((grid_size, grid_size))
    return np.ones((grid_size, grid_size))


def _get_neighborhood(state, pos):
    """Get total spin of a four neighboring spins."""
    grid_size = state.shape[0]

    # boundaries continue on opposite side
    return (
        state[(pos[0] + 1) % grid_size, pos[1]]
        + state[(pos[0] - 1) % grid_size, pos[1]]
        + state[pos[0], (pos[1] + 1) % grid_size]
        + state[pos[0], (pos[1] - 1) % grid_size]
    )


def update_state_ising(state, beta, rng):
    """Update state according to Glauber Ising model.

    Input:
        state:
            Square [N x N] array encoding the state of the Ising model.
        beta:
            Inverse temperature (1/kT).
        rng:
            numpy Random Generator (or compatible) object.

    Output:
        Updated state of the model ([N x N] square array) after N**2 MC
        steps.
    """
    exponent = np.exp(-np.arange(1, 3) * 4 * beta)

    grid_size = state.shape[0]
    n_steps = grid_size**2

    for _ in range(n_steps):
        pos = rng.integers(0, grid_size, size=2)

        dE = 2 * state[pos[0], pos[1]] * _get_neighborhood(state, pos)

        if (dE <= 0) or (rng.random() < exponent[int(dE / 4 - 1)]):
            state[pos[0], pos[1]] = -state[pos[0], pos[1]]

    return state
