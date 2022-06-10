import numpy as np

def init_pol(grid_size, m):
    
    a = np.arange(0, grid_size**2).reshape(grid_size, grid_size)
    a[a < grid_size*grid_size*0.5*(m+1)] = 1
    a[a >= grid_size*grid_size*0.5*(m+1)] = -1
    
    return a

def init_random(grid_size, m, rng=np.random.default_rng()):
    
    a = np.arange(0, grid_size**2).reshape(grid_size, grid_size)
    a[a < grid_size*grid_size*0.5*(m+1)] = 1
    a[a >= grid_size*grid_size*0.5*(m+1)] = -1
    a = a.ravel()
    rng.shuffle(a)
    a = a.reshape(grid_size,grid_size)
    
    return a

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

def count_links(state):
    
    count = 0
    
    for i in range(state.shape[0]):
        for j in range(state.shape[1]):
            for n in pos_neighbourhood(state, np.array((i, j))):
                if (state[i, j] == state[n[0], n[1]]):
                    count += 1
    return count/2

def index_links(state, rng):
    
    link = count_links(state) 

    pol = count_links(init_pol(state.shape[0], np.mean(state)))

    ran = count_links(init_random(state.shape[0], np.mean(state), rng))
    
    if (link - ran) >= 0:
        return (link-ran)/(pol-ran)
    else:
        return (link-ran)/(link)
            
