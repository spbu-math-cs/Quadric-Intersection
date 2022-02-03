from scipy.optimize import fsolve
from sklearn.neighbors import NearestNeighbors


class GridSolver:
    
    def __init__(self, equations, grid, tolerance=1e-6):
        self.tolerance = tolerance
        self.grid = grid
        self.equations = equations
        self.solutions = set()
        
    def _add(self, solution):
        for sol in self.solutions:
            if (abs(sol[0] - solution[0]) + abs(sol[1] - solution[1])) < self.tolerance:
                return
        self.solutions.add(solution)
        
    def find(self):
        self.solutions = set()
        for p in self.grid:
            sol, d, eir, msg = fsolve(self.equations, p, full_output=True)
            if eir==1:
                self._add((sol[0], sol[1]))
        return self.solutions

    
def get_points_line(points, k=10):

    # eps = 1e-1

    blocks = []
    N = points.shape[0]
    neigh = NearestNeighbors(n_neighbors=k, metric='euclidean')
    neigh.fit(points)
    nearest = neigh.kneighbors(points, return_distance=False)

    first = 0
    second = nearest[first][1]
    next_point = [first, second]
    used = [second]
    while (len(used) < N):
        current = next_point[-1]
        new_points = [p for p in nearest[current] if p not in used]
        if len(new_points) > 0:
            next_point.append(new_points[0])
            used.append(new_points[0])
        else:
            new_points = [p for p in nearest[first] if p not in used]
            if len(new_points) > 0:
                next_point = next_point[::-1]
                next_point.append(new_points[0])
                used.append(new_points[0])
            else:
#                 used.append(first)
                blocks.append(points[next_point])
                new_block = [i for i in range(N) if i not in used]
                first = new_block[0]
                second = nearest[first][1]
                next_point = [first, second]
                used.append(second)

    blocks.append(points[next_point])

    return blocks