import sys
import heapq
from collections import deque

# A Node keeps track of where we are in the maze, how we got here, and some extra info
class Node:
    __slots__ = ('state', 'parent', 'cost', 'depth', 'f')
    def __init__(self, state, parent=None, cost=0, depth=0, f=0):
        # state can be a (row, col) tuple for single-goal search
        # or a ((row, col), mask) pair for multi-goal search
        # parent points to the node we came from (so we can build the path)
        # cost is how many steps from the start so far
        # depth is how deep this is in the search tree
        # f is used by A* (cost plus heuristic)
        self.state = state
        self.parent = parent
        self.cost = cost
        self.depth = depth
        self.f = f

def load_maze(filepath):
    """
    Read a maze file where '%' are walls, 'P' is start, and '.' is the single goal.
    We return the maze as a 2D list, plus the start and goal coordinates.
    """
    maze = []
    start = None
    goal = None
    with open(filepath, 'r') as f:
        lines = [line.rstrip('\n') for line in f]

    # figure out how wide the maze is (some lines might be shorter)
    width = max(len(line) for line in lines)

    for y, line in enumerate(lines):
        # pad the line with spaces if it's shorter than the widest one
        row = list(line.ljust(width))
        for x, ch in enumerate(row):
            if ch == 'P':
                start = (y, x)
            elif ch == '.':
                goal = (y, x)
        maze.append(row)

    if start is None or goal is None:
        raise ValueError(f"{filepath} did not have both a 'P' start and a '.' goal.")
    return maze, start, goal

def load_multigoal_maze(filepath):
    """
    Read a maze file where '%' are walls, 'P' is start, and each '.' is a dot we need to eat.
    We return the maze, the start, and a list of all dot coordinates.
    """
    maze = []
    start = None
    goals = []
    with open(filepath, 'r') as f:
        lines = [line.rstrip('\n') for line in f]

    width = max(len(line) for line in lines)

    for y, line in enumerate(lines):
        row = list(line.ljust(width))
        for x, ch in enumerate(row):
            if ch == 'P':
                start = (y, x)
            elif ch == '.':
                goals.append((y, x))
        maze.append(row)

    if start is None:
        raise ValueError(f"{filepath} did not have a 'P' start.")
    return maze, start, goals

def get_neighbors(state, maze):
    """
    Given a position (row, col) in the maze, return all valid moves (up/down/left/right).
    Each move costs 1. We skip walls ('%').
    """
    y, x = state
    neighbors = []
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ny, nx = y + dy, x + dx
        if 0 <= ny < len(maze) and 0 <= nx < len(maze[0]) and maze[ny][nx] != '%':
            neighbors.append(((ny, nx), 1))
    return neighbors

def manhattan_heuristic(state, goal):
    """
    Simple Manhattan distance from our (row, col) to the goal's (row, col).
    Used in A* for the single-goal search.
    """
    return abs(state[0] - goal[0]) + abs(state[1] - goal[1])

def depth_first_search(maze, start, goal):
    """
    Do a DFS from start to goal. 
    We count how many nodes we expand, the maximum depth, and the largest fringe.
    When we find the goal, we build the path by following parent pointers.
    """
    # start with a stack containing the root node
    root = Node(state=start, parent=None, cost=0, depth=0)
    frontier = [root]
    explored = set()
    nodes_expanded = 0
    max_depth = 0
    max_fringe = 1

    while frontier:
        node = frontier.pop()  # take last inserted
        if node.state == goal:
            # we reached the goal, so reconstruct the path
            return {
                'path': reconstruct_path(node),
                'cost': node.cost,
                'nodes_expanded': nodes_expanded,
                'max_depth': max_depth,
                'max_fringe': max_fringe
            }
        if node.state in explored:
            continue
        explored.add(node.state)
        nodes_expanded += 1
        max_depth = max(max_depth, node.depth)

        # look at each neighbor and push onto stack if not already seen
        for (nbr_state, step_cost) in get_neighbors(node.state, maze):
            if nbr_state not in explored:
                child = Node(
                    state=nbr_state,
                    parent=node,
                    cost=node.cost + step_cost,
                    depth=node.depth + 1
                )
                frontier.append(child)
                max_fringe = max(max_fringe, len(frontier))

    # if we empty the stack without finding the goal, no solution
    return None

def breadth_first_search(maze, start, goal):
    """
    Do a BFS from start to goal. 
    BFS is guaranteed to find the shortest path (in terms of steps) since all steps cost 1.
    We measure nodes expanded, max depth, and max fringe size.
    """
    root = Node(state=start, parent=None, cost=0, depth=0)
    frontier = deque([root])
    explored = set()
    nodes_expanded = 0
    max_depth = 0
    max_fringe = 1

    while frontier:
        node = frontier.popleft()  # take oldest inserted
        if node.state == goal:
            return {
                'path': reconstruct_path(node),
                'cost': node.cost,
                'nodes_expanded': nodes_expanded,
                'max_depth': max_depth,
                'max_fringe': max_fringe
            }
        if node.state in explored:
            continue
        explored.add(node.state)
        nodes_expanded += 1
        max_depth = max(max_depth, node.depth)

        for (nbr_state, step_cost) in get_neighbors(node.state, maze):
            if nbr_state not in explored:
                child = Node(
                    state=nbr_state,
                    parent=node,
                    cost=node.cost + step_cost,
                    depth=node.depth + 1
                )
                frontier.append(child)
                max_fringe = max(max_fringe, len(frontier))

    return None

def a_star_search(maze, start, goal):
    """
    A* search for the single-goal maze. 
    We keep a min-heap (priority queue) sorted by f = g + h. 
    best_cost maps a state to the best g we've seen so far, so we can skip worse paths.
    """
    frontier = []
    counter = 0  # tie-breaker so that nodes with the same f are ordered by insertion
    h0 = manhattan_heuristic(start, goal)
    root = Node(state=start, parent=None, cost=0, depth=0, f=h0)
    heapq.heappush(frontier, (root.f, counter, root))
    counter += 1

    best_cost = {start: 0}
    nodes_expanded = 0
    max_depth = 0
    max_fringe = 1

    while frontier:
        max_fringe = max(max_fringe, len(frontier))
        f_curr, _, node = heapq.heappop(frontier)
        g_curr = node.cost

        # if we already found a cheaper way to this state, skip it
        if g_curr > best_cost.get(node.state, float('inf')):
            continue

        if node.state == goal:
            return {
                'path': reconstruct_path(node),
                'cost': node.cost,
                'nodes_expanded': nodes_expanded,
                'max_depth': max_depth,
                'max_fringe': max_fringe
            }

        nodes_expanded += 1
        max_depth = max(max_depth, node.depth)

        for (nbr_state, step_cost) in get_neighbors(node.state, maze):
            g2 = g_curr + step_cost
            # if this path to neighbor is better than any we've seen, record and push
            if g2 < best_cost.get(nbr_state, float('inf')):
                best_cost[nbr_state] = g2
                h2 = manhattan_heuristic(nbr_state, goal)
                child = Node(
                    state=nbr_state,
                    parent=node,
                    cost=g2,
                    depth=node.depth + 1,
                    f=g2 + h2
                )
                heapq.heappush(frontier, (child.f, counter, child))
                counter += 1

    # no way to reach goal
    return None

def reconstruct_path(node):
    """
    Follow parent pointers from the node back to the start, then reverse the list
    so we get a path from start to goal.
    """
    path = []
    while node:
        path.append(node.state)
        node = node.parent
    path.reverse()
    return path

def visualize_solution(maze, path, start, goal):
    """
    Make a copy of the maze and put '.' on every square in the path (skip the start and goal).
    Then print that new maze.
    """
    import copy
    grid = copy.deepcopy(maze)
    for (y, x) in path:
        if (y, x) not in [start, goal]:
            grid[y][x] = '.'
    for row in grid:
        print(''.join(row))

def get_multi_neighbors(state, maze, goal_indices):
    """
    For multi-goal search, our state = (position, mask).
    position is (row, col), mask is a bitmask showing which dots we've eaten.
    If we move onto a dot, we flip that bit on.
    """
    (pos, mask) = state
    y, x = pos
    neighbors = []
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ny, nx = y + dy, x + dx
        if 0 <= ny < len(maze) and 0 <= nx < len(maze[0]) and maze[ny][nx] != '%':
            new_mask = mask
            if (ny, nx) in goal_indices:
                new_mask |= (1 << goal_indices[(ny, nx)])
            neighbors.append(((ny, nx), new_mask))
    return neighbors

def reconstruct_multi_path(node):
    """
    Like reconstruct_path, but since node.state = (pos, mask),
    we only grab pos when building the path.
    """
    path = []
    while node:
        path.append(node.state[0])
        node = node.parent
    path.reverse()
    return path

def h_max_distance(state, goals, goal_indices):
    """
    Heuristic for multi-goal: look at all dots we haven't eaten and 
    return the max Manhattan distance from our current pos to any of those dots.
    That is a lower bound on how many steps we still need.
    """
    pos, mask = state
    distances = []
    for dot in goals:
        idx = goal_indices[dot]
        if not (mask & (1 << idx)):  # dot not eaten yet
            distances.append(abs(pos[0] - dot[0]) + abs(pos[1] - dot[1]))
    return max(distances) if distances else 0

def compute_mst_cost(points):
    """
    Given a list of (row, col) points, build a Minimum Spanning Tree (MST)
    where edge weights are Manhattan distances. Return total MST length.
    We do a simple Prim's algorithm in O(n^2) time for n points.
    """
    import math
    if not points:
        return 0
    N = len(points)
    visited = set()
    dist_to_tree = [math.inf] * N
    dist_to_tree[0] = 0  # pick the first point to start Prim's
    total = 0

    for _ in range(N):
        # pick the unvisited point with the smallest distance to what's already in MST
        u = min((d, i) for i, d in enumerate(dist_to_tree) if i not in visited)[1]
        total += dist_to_tree[u]
        visited.add(u)
        # update distances for all other points not yet in MST
        for v in range(N):
            if v not in visited:
                md = abs(points[u][0] - points[v][0]) + abs(points[u][1] - points[v][1])
                if md < dist_to_tree[v]:
                    dist_to_tree[v] = md
    return total

def h_mst(state, goals, goal_indices):
    """
    A stronger heuristic for multi-goal: 
    1) Find the distance from our current position to the closest dot.
    2) Compute an MST over all the dots we haven't eaten yet.
    The sum of these is a lower bound on how many steps we still need.
    """
    pos, mask = state
    remaining = []
    for dot in goals:
        idx = goal_indices[dot]
        if not (mask & (1 << idx)):
            remaining.append(dot)
    if not remaining:
        return 0

    # dist from current position to the nearest remaining dot
    d0 = min(abs(pos[0] - d[0]) + abs(pos[1] - d[1]) for d in remaining)
    # now build MST over all remaining dots
    mst_cost = compute_mst_cost(remaining)
    return d0 + mst_cost

def a_star_multi(maze, start, goals, heuristic_fn, node_limit=200_000):
    """
    A* search for the multi-goal problem (eat all dots).
    We keep state = (pos, mask). The mask is a bitmask that shows which dots are already eaten.
    heuristic_fn is one of the above heuristics (h_max_distance or h_mst).
    If we expand more than node_limit nodes, we bail out to avoid really long runs.
    """
    goal_indices = {g: i for i, g in enumerate(goals)}
    N = len(goals)
    all_dots_mask = (1 << N) - 1

    init_state = (start, 0)  # no dots eaten at the very beginning
    h0 = heuristic_fn(init_state, goals, goal_indices)
    root = Node(state=init_state, parent=None, cost=0, depth=0, f=h0)

    frontier = []
    counter = 0
    heapq.heappush(frontier, (root.f, counter, root))
    counter += 1

    best_cost = {init_state: 0}  # map state -> best g we’ve seen
    nodes_expanded = 0
    max_depth = 0
    max_fringe = 1

    while frontier:
        if nodes_expanded > node_limit:
            print(">> Expanded beyond node_limit, giving up on this heuristic")
            return None

        max_fringe = max(max_fringe, len(frontier))
        f_curr, _, node = heapq.heappop(frontier)
        state = node.state
        g_curr = node.cost

        # if this path to state is worse than one we already found, skip
        if g_curr > best_cost.get(state, float('inf')):
            continue

        pos, mask = state
        # check if we've eaten every dot
        if mask == all_dots_mask:
            return {
                'path': reconstruct_multi_path(node),
                'cost': node.cost,
                'nodes_expanded': nodes_expanded,
                'max_depth': max_depth,
                'max_fringe': max_fringe
            }

        nodes_expanded += 1
        max_depth = max(max_depth, node.depth)

        for (nbr_pos, nbr_mask) in get_multi_neighbors(state, maze, goal_indices):
            new_state = (nbr_pos, nbr_mask)
            g2 = g_curr + 1
            if g2 < best_cost.get(new_state, float('inf')):
                best_cost[new_state] = g2
                h2 = heuristic_fn(new_state, goals, goal_indices)
                child = Node(
                    state=new_state,
                    parent=node,
                    cost=g2,
                    depth=node.depth + 1,
                    f=g2 + h2
                )
                heapq.heappush(frontier, (child.f, counter, child))
                counter += 1

    # no solution if we empty the heap
    return None

def run_part1():
    """
    Run single-goal searches (BFS and A*) on four mazes and print results.
    We show path cost, nodes expanded, max depth, max fringe, and an ASCII map of the path.
    """
    maze_files = ["smallMaze.lay", "mediumMaze.lay", "bigMaze.lay", "openMaze.lay"]
    algorithms = [
        ("BFS", breadth_first_search),
        ("A*", a_star_search)
    ]
    # If you want DFS instead of BFS, replace breadth_first_search with depth_first_search

    for alg_name, func in algorithms:
        print(f"\n--- {alg_name} on Part 1 Mazes ---")
        for fpath in maze_files:
            maze, start, goal = load_maze(fpath)
            result = func(maze, start, goal)
            if not result:
                print(f"{fpath}: no solution found")
            else:
                print(f"{fpath:12} | cost={result['cost']:3} | #exp={result['nodes_expanded']:6} "
                      f"| maxD={result['max_depth']:3} | maxF={result['max_fringe']:5}")
                print("Here’s the maze with the path marked by dots:")
                visualize_solution(maze, result['path'], start, goal)
                print()

def run_part2():
    """
    Run A* on the multi-goal mazes (tinySearch, smallSearch, trickySearch) using different heuristics.
    For each heuristic we show cost, nodes expanded, and an ASCII overlay of the path.
    """
    search_files = ["tinySearch.lay", "smallSearch.lay", "trickySearch.lay"]
    heuristics = [
        ("max-dist", h_max_distance),
        ("mst    ", h_mst),
        # You can add ("farthest-pair", h_farthest_pair) if you write that heuristic
    ]

    for fpath in search_files:
        print(f"\n=== Now solving: {fpath} ===")
        maze, start, goals = load_multigoal_maze(fpath)
        for hname, hfunc in heuristics:
            print(f" Using heuristic: {hname}")
            result = a_star_multi(maze, start, goals, hfunc, node_limit=200_000)
            if result is None:
                print("  Didn’t find a solution within the node limit.")
            else:
                print(f"  cost={result['cost']:3} | #exp={result['nodes_expanded']:6} "
                      f"| maxF={result['max_fringe']:6}")
                print("  Path as a list of coordinates:", result['path'])
                print("  Maze with path marked by dots:")
                sol_maze = [row.copy() for row in maze]
                for (y, x) in result['path']:
                    # don’t overwrite the start or the actual dot positions
                    if (y, x) not in goals and (y, x) != start:
                        sol_maze[y][x] = '.'
                for row in sol_maze:
                    print(''.join(row))
                print()

if __name__ == "__main__":
    print("===== Part 1 =====")
    run_part1()
    print("\n===== Part 2 =====")
    run_part2()
