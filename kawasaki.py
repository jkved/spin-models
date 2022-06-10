import numpy as np
import matplotlib.pyplot as plt

def initialize_kawasaki(grid_size, m, rng=np.random.default_rng()):
    '''
    Initialize matrix for Ising modelling in Kawasaki. Creates a random distribution depending on conserved-order parameter (transfors m -> ro, as known in literature). 
    To achieve this, matrix cannot be done randomly but specific amount of spins need to be distributed.
    Input:
        grid_size:
            Dimension of matrix. Variable grid_size**2 denotes total number of lattice points. Each one contains spin with values +1 or -1.
        m:
            Conserved-order parameter, intensive magnetization. Main factor in determining relative number of up/down spins in matrix.
        rng:
            Random number generator object
    
    Output:
        state:
            Matrix of grid_size dimensions. Contains spins in proportion to conserved-order parameter value.
            
    '''
    a = np.arange(0, grid_size**2).reshape(grid_size, grid_size)
    a[a < grid_size*grid_size*0.5*(m+1)] = 1
    a[a >= grid_size*grid_size*0.5*(m+1)] = -1
    a = a.ravel()
    rng.shuffle(a)
    a = a.reshape(grid_size,grid_size)
    
    return a

def get_lists(state):
    '''
    Obtain two lists from matrix of Ising model. One is comprised of coordinates for up spins and another with coordinates for down spins.
    Input:
        state:
            Matrix of grid_size**2 lattice count. Each lattice contains up (+1) or down (-1) spin values.
    Output:
        up:
            List of coordinates in Ising matrix for up (+1) spins.
        down:
            List of coordinates in Ising matrix for down (-1) spins.
    '''
    up = np.ndarray((state.shape[0]**2, 2)).astype(int)
    up.fill(-1)
    
    down = np.ndarray((state.shape[0]**2, 2)).astype(int)
    down.fill(-1)
    
    i_u = 0
    i_d = 0
    
    
    for i, j in np.ndindex(state.shape):
        if state[i, j] == 1:
            up[i_u, :] = i, j
            i_u +=1
        else:
            down[i_d, :] = i, j
            i_d +=1
    
    up = up[~np.all(up == -1, axis=1)]
    down = down[~np.all(down == -1, axis=1)]
    
    return up, down

def get_neighborhood(state, pos):
    '''
    Gets sum of spins of closest neighbours. Closest meaning bordering on the side lattices.
    Input:
        state:
            Matrix of grid_size**2 lattice count. Each lattice contains up (+1) or down (-1) spin values.
        pos:
            Coordinates of position for the selected lattice.
    Output:
        sum:
            Summation of neigbours spin values.
    '''
    grid_size = state.shape[0]

    # boundaries continue on opposite side
    return (
        state[(pos[0] + 1) % grid_size, pos[1]]
        + state[(pos[0] - 1) % grid_size, pos[1]]
        + state[pos[0], (pos[1] + 1) % grid_size]
        + state[pos[0], (pos[1] - 1) % grid_size]
    )

def coord_swap(up, down, p1, p2):
    
    up[p1, 0], down[p2, 0] = coord_swap_2(up[p1, 0], down[p2, 0])
    up[p1, 1], down[p2, 1] = coord_swap_2(up[p1, 1], down[p2, 1])
    
    
    return up, down

def coord_swap_2(x1, y1):
    '''
    Function is used because a,b=b,a form doesnt work for arrays of dim>1 (like up and down from get_lists(). See discussion comments on: https://stackoverflow.com/questions/22847410/swap-two-values-in-a-numpy-array
    '''
    
    temp = x1
    x1 = y1
    y1 = temp
    
    return x1, y1

def update_state_kawasaki(up, down, state, beta, rng):
    """Update state according to Glauber Ising model.

    Input:
        up:
            List of coordinates in Ising matrix for up (+1) spins.
        down:
            List of coordinates in Ising matrix for down (-1) spins.
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
    #instead of looking for certain possible exp values, let's calculate exponents e^1 to e^16
    #exponent = np.exp(-np.arange(2, 5) * 4 * beta)
    exponent = np.exp(-np.arange(1, 17) * beta) 
    
    grid_size = state.shape[0]
    n_steps = grid_size**2

    for _ in range(n_steps):
        p1 = rng.integers(0, up.shape[0])
        p2 = rng.integers(0, down.shape[0])
        
        '''
        Different methods are applied than in Ising Metropolis for switching neighbouring or far away spins.
        
        For neighbouring - we calculate difference in energy of changed vs current state spins. If the change is suitable, we get to keep the changed lattice. Otherwise we need to reverse the change.
        
        For far-away spins we choose equation from Newman which relies only on current state spins. This way we only calculate the current state energy.
        
        For simplier code implementation we choose to calculate it the neighbouring way.
        
        '''
            
        dEu = state[up[p1, 0], up[p1, 1]] * get_neighborhood(state, up[p1,:]) + state[down[p2, 0], down[p2, 1]] * get_neighborhood(state, down[p2,:])
        
        state[up[p1, 0], up[p1, 1]] = -state[up[p1, 0], up[p1, 1]]
        state[down[p2, 0], down[p2, 1]] = -state[down[p2, 0], down[p2, 1]]

        dEv = state[up[p1, 0], up[p1, 1]] * get_neighborhood(state, up[p1,:]) + state[down[p2, 0], down[p2, 1]] * get_neighborhood(state, down[p2,:])

        if ((dEu - dEv) <= 0 ) or (rng.random() < exponent[int(dEu - dEv - 1)]):
            #spins are kept, cause we made the change earlier. we do update the lists though
            up, down = coord_swap(up, down, p1, p2)
        else:
            state[up[p1, 0], up[p1, 1]] = -state[up[p1, 0], up[p1, 1]]
            state[down[p2, 0], down[p2, 1]] = -state[down[p2, 0], down[p2, 1]]
            
    return state                                               