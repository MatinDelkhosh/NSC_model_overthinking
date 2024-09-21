import numpy as np
import random

def neighbor(cell, direction, msize):
    neigh = ((cell + direction + msize - 1) % msize) + 1
    return neigh

def neighbors(cell, msize, wrap=True):
    dirs = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    Ns = [cell + dirs[a] for a in range(4)]
    as_ = list(range(4))
    
    if wrap:  # States outside arena are wrapped to the other side
        Ns = [((N + msize - 1) % msize) + 1 for N in Ns]
    else:  # States outside arena are not considered neighbors
        inds = [i for i in range(len(Ns)) if (np.min(Ns[i]) > 0.5) and (np.max(Ns[i]) < msize + 0.5)]
        Ns, as_ = [Ns[i] for i in inds], [as_[i] for i in inds]
    
    return Ns, as_

def walk(maz, nxtcell, msize, visited=None, wrap=True):
    if visited is None:
        visited = []

    dir_map = {1: 2, 2: 1, 3: 4, 4: 3}
    visited.append((nxtcell[0] - 1) * msize + nxtcell[1])  # Add to list of visited cells

    neighs, as_ = neighbors(nxtcell, msize, wrap=wrap)  # Get list of neighbors

    for nnum in random.sample(range(len(neighs)), len(neighs)):  # Randomly shuffle neighbors
        neigh, a = neighs[nnum], as_[nnum]  # Corresponding state and action
        ind = (neigh[0] - 1) * msize + neigh[1]  # Convert from coordinates to index

        if ind not in visited:  # Check that we haven't been there
            maz[nxtcell[0] - 1, nxtcell[1] - 1, a] = 0.0  # Remove wall
            maz[neigh[0] - 1, neigh[1] - 1, dir_map[a]] = 0.0  # Remove reverse wall
            maz, visited = walk(maz, neigh, msize, visited, wrap=wrap)

    return maz, visited

def maze(msize, wrap=True):
    dirs = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    dir_map = {1: 2, 2: 1, 3: 4, 4: 3}
    
    maz = np.ones((msize, msize, 4), dtype=np.float32)  # Start with walls everywhere
    cell = np.random.randint(1, msize+1, 2)  # Random start point
    
    maz, visited = walk(maz, cell, msize, wrap=wrap)  # Walk through the maze
    
    # Remove additional walls to increase degeneracy
    if wrap:
        holes = int(3 * (msize - 3))  # 3 holes for Larena=4, 6 holes for Larena=5
    else:
        holes = int(4 * (msize - 3))  # 4 holes for Larena=4, 8 holes for Larena=5
        
        # Set permanent walls
        maz[msize - 1, :, 0] = 0.5
        maz[0, :, 1] = 0.5
        maz[:, msize - 1, 2] = 0.5
        maz[:, 0, 3] = 0.5
    
    for _ in range(holes):
        walls = np.argwhere(maz == 1)
        wall = random.choice(walls)
        cell, a = wall[:2], wall[2]

        neigh = neighbor([cell[0] + 1, cell[1] + 1], dirs[a], msize)
        maz[cell[0], cell[1], a] = 0.0  # Remove wall
        maz[neigh[0] - 1, neigh[1] - 1, dir_map[a]] = 0.0  # Remove reverse wall
    
    maz[maz == 0.5] = 1.0  # Reinstate permanent walls

    # Reshape to a 2D array
    maz = np.reshape(np.transpose(maz, (1, 0, 2)), (msize**2, 4))

    return maz.astype(np.float32)