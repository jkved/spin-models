import numpy as np

def initialize_state(N, rng=np.random.default_rng()):
    return 2*np.around(rng.random((N,N)))-1

def stepMC(lattice, kT, rng=np.random.default_rng()):
    '''
    Performs action of applying cluster area to lattice. Applied area is flipped.
    Input:
        lattice:
            Lattice of Ising model. Contains u/d spins, size of grid_size.
        kT:
            beta parameter, temperature at which simulations take place. Can also be changed with J.
        rng:
            random generator object
    Output:
        Lattice with flipped cluster.
    '''
    grid_size = lattice.shape[0]
    state = np.ones((grid_size, grid_size))
    
    state = update(lattice,
                   state, 
                   np.array((np.around((grid_size-1)*rng.random()), np.around((grid_size-1)*rng.random()))).astype(int),
                   (1 - np.exp(-2*kT)),
                   rng)
    
    return lattice*state
    
def update(lattice, state, pos, P, rng = np.random.default_rng()):
    '''
    Recursive function which selects cluster with certain probability P and returns same dim as lattice variable "state". This variable marks the area of cluster.
    Input:
        lattice, kT, rng:
            same as in stepMC
        state:
            position of cluster on lattice.
        P:
            probability to include spin to the cluster
        pos:
            seed spin
    Output:
        mapped area of cluster.
    '''
    state[pos[0],pos[1]] *= -1
    
    for i in pos_neighbourhood(state, pos):
        
        if((lattice[pos[0], pos[1]] == lattice[i[0], i[1]])
           and (rng.random() < P) 
           and (state[i[0], i[1]] == 1)):
            
            state = update(lattice, state, np.array((i[0], i[1])), P, rng)
            
    return state
    
def pos_neighbourhood(__state, pos):
    '''
    Gets coords of closest neighbours. Closest meaning bordering on the side lattices.
    Input:
        state:
            Matrix of grid_size**2 cell count.
        pos:
            Coordinates of position for the selected cell.
    Output:
            array of 4 rows with x and y coordinates for neighbour
    '''
    grid_size = __state.shape[0]
    
    return np.array((((pos[0] + 1) % grid_size, pos[1]),
                     ((pos[0] - 1) % grid_size, pos[1]),
                     (pos[0], (pos[1] + 1) % grid_size),
                     (pos[0], (pos[1] - 1) % grid_size)))

def calcE(state, H = 0, mu = 1):
    '''
    Function for calculating overall energy of spin matrix.
    Input:
        state:
            lattice of spins
        H, mu:
            external magnetic fields parameters
    Output:
        E:
            total energy
    '''
    
    E = 0
    
    for a, k in enumerate(state):
        for b, l in enumerate(k):
            
            N1 = len(state[:, 0])
            N2 = len(state[0, :])
            
            s = state[a, b]
            n = state[(a+1)%N1, b] + state[(a-1)%N1, b] + state[a, (b+1)%N2] + state[a, (b-1)%N2]
            
            E += (n*s)/2 + mu*H*s
    
    return E     