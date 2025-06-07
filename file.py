import numpy as np
import heapq
import math
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- Configuration Constants ---
GRID_SIZE = 100
START_NODE = (5, 5)
END_NODE = (95, 95)
VIDEO_FILENAME = "pathfinding_simulation.mp4"

# --- Visualization Constants ---
FRAME_COUNT = 20
ANIMATION_INTERVAL = 400 # Slower for better viewing

# --- Color Definitions ---
COLOR_START = 'lime'
COLOR_END = 'red'
COLOR_PATH = 'gold'
COLOR_VISITED = 'lightblue'
COLOR_RRT_TREE = '#8A2BE2'
COLOR_APF_WAYPOINTS = 'cyan'
COLOR_APF_CURRENT_WAYPOINT = 'red'
COLOR_APF_CURRENT_POS = 'black'
COLOR_RRT_RANDOM_SAMPLE = 'blue'
COLOR_RRT_NEW_NODE = 'green'

# --- Grid Generation that Guarantees a Solvable Path ---
def generate_grid(size, num_obstacles, max_obstacle_size):
    """
    Creates a grid by adding obstacles to an empty space, then guarantees
    a path by carving a tunnel if needed.
    """
    grid = np.zeros((size, size))

    for _ in range(num_obstacles):
        w = random.randint(max_obstacle_size // 2, max_obstacle_size)
        h = random.randint(max_obstacle_size // 2, max_obstacle_size)
        x = random.randint(0, size - w - 1)
        y = random.randint(0, size - h - 1)
        grid[y:y+h, x:x+w] = 1

    grid[START_NODE] = 0
    grid[END_NODE] = 0

    def _carve_path(grid, start, end):
        def heuristic(a, b): return abs(a[0] - b[0]) + abs(a[1] - b[1])
        neighbors, came_from, gscore, close_set = [(0,1),(-1,0),(1,0),(0,-1)], {}, {start: 0}, set()
        fscore = {start: heuristic(start, end)}
        oheap = [(fscore[start], start)]
        while oheap:
            current = heapq.heappop(oheap)[1]
            if current == end:
                path = []
                while current in came_from: path.append(current); current = came_from[current]
                for pos in path:
                    for r_offset in range(-1, 2):
                        for c_offset in range(-1, 2):
                           ry, cx = pos[0] + r_offset, pos[1] + c_offset
                           if 0 <= ry < size and 0 <= cx < size: grid[ry, cx] = 0
                return True
            close_set.add(current)
            for i, j in neighbors:
                neighbor = current[0] + i, current[1] + j
                if not (0 <= neighbor[0] < size and 0 <= neighbor[1] < size): continue
                move_cost = 1 if grid[neighbor] == 0 else 50
                tentative_g_score = gscore[current] + move_cost
                if tentative_g_score < gscore.get(neighbor, float('inf')):
                    came_from[neighbor], gscore[neighbor] = current, tentative_g_score
                    fscore[neighbor] = tentative_g_score + heuristic(neighbor, end)
                    heapq.heappush(oheap, (fscore[neighbor], neighbor))
        return False
    
    _carve_path(grid, START_NODE, END_NODE)
    return grid

# --- A* Helper Function ---
def a_star_path_finder(grid, start, end):
    """
    A more robust A* helper that can pass through obstacles at a high
    cost. This guarantees it will always find a path on a solvable map.
    """
    def heuristic(a, b): return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    neighbors = [(0,1),(-1,0),(1,0),(0,-1)]
    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, end)}
    oheap = [(fscore[start], start)]

    while oheap:
        current = heapq.heappop(oheap)[1]

        if current == end:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            # Return a sparse list of waypoints from the full path
            return path[::7] + [end] if len(path) > 14 else path

        close_set.add(current)
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            if not (0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1]):
                continue
            
            # Key change: Treat obstacles as high-cost terrain instead of impassable walls
            move_cost = 1 if grid[neighbor] == 0 else 50 
            tentative_g_score = gscore[current] + move_cost

            if tentative_g_score < gscore.get(neighbor, float('inf')):
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, end)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))
    return [] # Should not be reached on a solvable map

def advanced_potential_field(grid, start, end):
    """
    A corrected 'Advanced' Potential Field with a robust helper and
    re-tuned forces for more reliable navigation.
    """
    waypoints = a_star_path_finder(grid, start, end)
    
    # This check ensures the animator doesn't crash on the rare occasion
    # the grid is truly impossible (which our generator should prevent).
    if not waypoints: 
        print("\n  [Adv. Potential Field]: A* guide planner failed to find a path.")
        yield {'waypoints': [], 'path': [start], 'waypoint_idx': None, 'pos': start}
        return

    print(f"\n  [Adv. Potential Field]: A* guide path found with {len(waypoints)} waypoints.")
    full_path, pos = [start], np.array(start, dtype=float)
    obstacle_pos = np.array(np.where(grid == 1)).T
    
    for waypoint_index, waypoint in enumerate(waypoints[1:]):
        current_waypoint_goal = np.array(waypoint, dtype=float)
        stuck_counter = 0
        
        for step in range(300): # Max steps per waypoint
            last_pos = pos.copy()
            if step % 8 == 0: 
                yield {'waypoints': waypoints, 'path': full_path, 'waypoint_idx': waypoint_index + 1, 'pos': pos}
            
            # --- Re-tuned Force Calculation ---
            vec_to_waypoint = current_waypoint_goal - pos
            attractive_force = vec_to_waypoint / (np.linalg.norm(vec_to_waypoint) + 1e-6)
            
            repulsive_force = np.zeros(2)
            for obs in obstacle_pos:
                diff = pos - obs
                dist = np.linalg.norm(diff)
                
                # A more effective influence radius
                if dist < 8.0: 
                    repulsive_force += (1.0/dist - 1.0/8.0) * (1.0 / dist**2) * (diff / (dist + 1e-6))
            
            # Using your requested weaker repulsion strength
            total_force = attractive_force + 0.3 * repulsive_force

            # --- Local Minima Escape ---
            if np.linalg.norm(pos - last_pos) < 0.1:
                stuck_counter += 1
            else:
                stuck_counter = 0
            
            if stuck_counter > 25:
                slide_force = np.array([-total_force[1], total_force[0]])
                total_force += slide_force * 0.8 
                stuck_counter = 0

            # Update position
            pos += total_force * 1.5
            pos[0] = np.clip(pos[0], 0, grid.shape[0] - 1)
            pos[1] = np.clip(pos[1], 0, grid.shape[1] - 1)
            
            full_path.append(tuple(pos))

            if np.linalg.norm(pos - current_waypoint_goal) < 1.5: 
                break
            
    # Yield the final state
    yield {'waypoints': waypoints, 'path': full_path, 'waypoint_idx': len(waypoints)-1, 'pos': pos}




def a_star(grid, start, end):
    """
    A faster A* implementation that uses 8-directional movement
    with weighted costs for more direct paths.
    """
    def heuristic(a, b):
        # Manhattan distance is a good heuristic for grid movement
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    # --- OPTIMIZATION: 8-directional movement ---
    # (move_y, move_x, cost)
    neighbors = [
        (0, 1, 1), (0, -1, 1), (1, 0, 1), (-1, 0, 1),  # Cardinal directions
        (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), # Diagonal directions
        (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2))
    ]

    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, end)}
    oheap = [(fscore[start], start)]
    
    yield_interval = int((grid.size * 0.1) / FRAME_COUNT) or 1
    path_found = False

    while oheap:
        current = heapq.heappop(oheap)[1]

        if current == end:
            path_found = True
            break
        
        close_set.add(current)

        if len(close_set) % yield_interval == 0:
            yield {'visited': list(close_set)}

        for i, j, move_cost in neighbors:
            neighbor = current[0] + i, current[1] + j

            if not (0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1]) or grid[neighbor] == 1:
                continue
            
            # --- OPTIMIZATION: Use the weighted move cost ---
            tentative_g_score = gscore[current] + move_cost
            
            if tentative_g_score < gscore.get(neighbor, float('inf')):
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, end)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))

    path = []
    if path_found:
        current = end
        while current in came_from:
            path.append(current)
            current = came_from[current]
        path.append(start)
        path.reverse()
        
    # Yield the final state
    yield {'visited': list(close_set), 'path': path}

def dijkstra(grid, start, end):
    neighbors, visited, parent, cost = [(0,1),(-1,0),(1,0),(0,-1)], set(), {}, {start: 0}
    pq = [(0, start)]
    yield_interval = int((grid.size * 0.3) / FRAME_COUNT) or 1
    path_found = False
    while pq:
        d, current = heapq.heappop(pq)
        if current == end: path_found = True; break
        if current in visited: continue
        visited.add(current)
        if len(visited) % yield_interval == 0: yield {'visited': list(visited)}
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            if not (0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1]) or grid[neighbor] == 1: continue
            new_cost = cost[current] + 1
            if new_cost < cost.get(neighbor, float('inf')):
                cost[neighbor], parent[neighbor] = new_cost, current
                heapq.heappush(pq, (new_cost, neighbor))
    path = []
    if path_found:
        current = end
        while current in parent: path.append(current); current = parent[current]
        path.append(start); path.reverse()
    yield {'visited': list(visited), 'path': path}

class RRTStarNode:
    def __init__(self, y, x): self.y, self.x, self.parent, self.cost = y, x, None, 0.0

def rrt_star(grid, start, end, max_iter=4000, step_size=8.0, search_radius=15.0, goal_bias=0.1):
    """
    A corrected RRT* implementation that ensures the final path connection
    is collision-free and uses a robust collision checker.
    """
    start_node = RRTStarNode(start[0], start[1])
    end_node = RRTStarNode(end[0], end[1])
    node_list = [start_node]
    frame_yield_interval = max_iter // FRAME_COUNT
    
    def is_collision(n1, n2):
        """
        Robust collision checker using high-frequency sampling to prevent
        'skimming' or passing through thin obstacles.
        """
        dist = np.hypot(n1.x - n2.x, n1.y - n2.y)
        # Check one point for at least every half-pixel of distance
        num_samples = int(dist * 2) + 2
        points = np.linspace((n1.y, n1.x), (n2.y, n2.x), num_samples)
        for p in points:
            iy = np.clip(int(p[0]), 0, grid.shape[0] - 1)
            ix = np.clip(int(p[1]), 0, grid.shape[1] - 1)
            if grid[iy, ix] == 1:
                return True
        return False
        
    for i in range(max_iter):
        # 1. Sample a random point
        rnd_node = RRTStarNode(end[0], end[1]) if random.random() < goal_bias else RRTStarNode(random.uniform(0, grid.shape[0]), random.uniform(0, grid.shape[1]))
        
        # 2. Find nearest node in tree
        nearest_node = min(node_list, key=lambda n: np.hypot(rnd_node.x - n.x, rnd_node.y - n.y))
        
        # 3. Steer from nearest node towards the random point
        theta = math.atan2(rnd_node.y - nearest_node.y, rnd_node.x - nearest_node.x)
        new_node = RRTStarNode(nearest_node.y + step_size*math.sin(theta), nearest_node.x + step_size*math.cos(theta))

        # 4. Check if the new node is valid (in bounds and collision-free)
        if not (0 <= new_node.x < grid.shape[1] and 0 <= new_node.y < grid.shape[0]) or is_collision(nearest_node, new_node):
            if i > 0 and i % frame_yield_interval == 0: yield {'nodes': node_list, 'sample': rnd_node, 'new_node': None}
            continue
        
        # 5. Choose Parent (the "star" part): find the best connection in the local neighborhood
        near_nodes = [n for n in node_list if np.hypot(n.x - new_node.x, n.y - new_node.y) < search_radius]
        best_parent = nearest_node
        min_cost = nearest_node.cost + np.hypot(new_node.x - nearest_node.x, new_node.y - nearest_node.y)
        
        for n in near_nodes:
            cost = n.cost + np.hypot(new_node.x - n.x, new_node.y - n.y)
            if cost < min_cost and not is_collision(n, new_node):
                min_cost, best_parent = cost, n
        
        new_node.cost, new_node.parent = min_cost, best_parent
        node_list.append(new_node)

        # 6. Rewire the tree: check if the new node provides a better path for its neighbors
        for n in near_nodes:
            cost_via_new = new_node.cost + np.hypot(n.x - new_node.x, n.y - new_node.y)
            if cost_via_new < n.cost and not is_collision(new_node, n):
                n.parent = new_node
                n.cost = cost_via_new
        
        if i > 0 and i % frame_yield_interval == 0: yield {'nodes': node_list, 'sample': rnd_node, 'new_node': new_node}
            
    # --- Final Path Reconstruction (Corrected) ---
    path = []
    # Find the node in the tree closest to the goal
    last_node = min(node_list, key=lambda n: np.hypot(n.x - end[1], n.y - end[0]))
    
    # Check if this closest node can connect to the goal without collision
    if np.hypot(last_node.x - end[1], last_node.y - end[0]) <= step_size and not is_collision(last_node, end_node):
        node = last_node
        while node: 
            path.append((node.y, node.x))
            node = node.parent
        path.reverse()
        path.append(end) # Explicitly add the final goal to the path
        
    yield {'nodes': node_list, 'path': path}
    

def main():
    print("--- Pathfinding Algorithm Simulation ---")
    print("Select Obstacle Difficulty:")
    print("  1: Easy (Few, small obstacles)")
    print("  2: Medium (More, medium obstacles)")
    print("  3: Hard (Many, larger obstacles)")
    difficulty_map = {'1': (15, 8), '2': (30, 12), '3': (45, 16)}
    choice = input("Enter your choice (1, 2, or 3): ")
    num_obstacles, max_obstacle_size = difficulty_map.get(choice, difficulty_map['2'])
    grid = generate_grid(GRID_SIZE, num_obstacles, max_obstacle_size)
    
    planners = {"A*": a_star, "Dijkstra": dijkstra, "RRT*": rrt_star, "Advanced Potential Field": advanced_potential_field}
    planner_data = {}

    print("\n--- Running Algorithms ---")
    for name, func in planners.items():
        frames = [frame for frame in func(grid, START_NODE, END_NODE)]
        planner_data[name] = {"frames": frames}
        final_path = frames[-1].get('path', [])
        planner_data[name]['path_found'] = bool(final_path)
        print(f"  - {name} finished. Path found: {bool(final_path)}")
    print("--- Computation Complete ---\n")

    fig, axes = plt.subplots(2, 2, figsize=(12, 12), constrained_layout=True)
    axes = axes.flatten()
    fig.suptitle('Live Pathfinding Simulation', fontsize=16)

    max_frames = max(len(data['frames']) for data in planner_data.values())
    artists = []
    
    for i, (name, ax) in enumerate(zip(planners.keys(), axes)):
        ax.imshow(grid, cmap='Greys', vmin=0, vmax=1)
        ax.plot(START_NODE[1], START_NODE[0], 'o', markersize=10, color=COLOR_START, label='Start')
        ax.plot(END_NODE[1], END_NODE[0], 'X', markersize=12, markeredgecolor='k', color=COLOR_END, label='End')
        if not planner_data[name]['path_found']: ax.set_title(f"{name}\n(Path Not Found)", color='red')
        else: ax.set_title(name)
        ax.set_xticks([]); ax.set_yticks([])
        
        artist_dict = {}
        if name in ["A*", "Dijkstra"]:
            artist_dict['visited'] = ax.scatter([], [], s=5, c=COLOR_VISITED)
        elif name == "RRT*":
            artist_dict['tree'] = ax.plot([], [], '-', color=COLOR_RRT_TREE, linewidth=0.4)[0]
            artist_dict['sample'] = ax.plot([], [], 'o', markersize=6, color=COLOR_RRT_RANDOM_SAMPLE, label='Random Sample')[0]
            artist_dict['new_node'] = ax.plot([], [], 'o', markersize=6, color=COLOR_RRT_NEW_NODE, label='New Node')[0]
        elif name == "Advanced Potential Field":
            artist_dict['waypoints'] = ax.plot([], [], '--o', color=COLOR_APF_WAYPOINTS, markersize=4, linewidth=0.8, label='A* Waypoints')[0]
            artist_dict['current_waypoint'] = ax.plot([], [], 'o', markersize=12, markeredgecolor='k', mfc='none', color=COLOR_APF_CURRENT_WAYPOINT, label='Target Waypoint')[0]
            artist_dict['current_pos'] = ax.plot([], [], '.', markersize=8, color=COLOR_APF_CURRENT_POS, label='Current Position')[0]
            # Use a different key for the evolving APF path to distinguish from the final path
            artist_dict['apf_trail'] = ax.plot([], [], color='orange', linewidth=1.5, label='APF Trail')[0]

        artist_dict['path'] = ax.plot([], [], color=COLOR_PATH, linewidth=2.5, label='Final Path')[0]
        artists.append(artist_dict)
        ax.legend(loc='upper left', fontsize='small')

    def update(frame_num):
        print(f"\rDisplaying frame {frame_num+1}/{max_frames}", end="")
        for i, name in enumerate(planners.keys()):
            data = planner_data[name]
            frame_idx = min(frame_num, len(data['frames']) - 1)
            frame_data = data['frames'][frame_idx]

            # --- MODIFIED LOGIC ---
            # Show the final path only on the last frame for each algorithm
            if frame_idx == len(data['frames']) - 1:
                final_path = frame_data.get('path', [])
                if final_path:
                    artists[i]['path'].set_data(np.array(final_path)[:, 1], np.array(final_path)[:, 0])
            else:
                artists[i]['path'].set_data([], [])


            if name in ["A*", "Dijkstra"]:
                visited = frame_data.get('visited', [])
                if visited: artists[i]['visited'].set_offsets(np.c_[np.array(visited)[:, 1], np.array(visited)[:, 0]])
            elif name == "RRT*":
                nodes, rnd_node, new_node = frame_data.get('nodes', []), frame_data.get('sample'), frame_data.get('new_node')
                tx, ty = [], []
                for node in nodes:
                    if node.parent: tx.extend([node.x, node.parent.x, None]); ty.extend([node.y, node.parent.y, None])
                artists[i]['tree'].set_data(tx, ty)
                if rnd_node: artists[i]['sample'].set_data([rnd_node.x], [rnd_node.y])
                else: artists[i]['sample'].set_data([], [])
                if new_node: artists[i]['new_node'].set_data([new_node.x], [new_node.y])
                else: artists[i]['new_node'].set_data([], [])
            elif name == "Advanced Potential Field":
                # Always show the evolving trail for APF
                apf_trail = frame_data.get('current_path', [])
                if apf_trail: artists[i]['apf_trail'].set_data(np.array(apf_trail)[:, 1], np.array(apf_trail)[:, 0])
                
                waypoints, waypoint_idx, current_pos = frame_data.get('waypoints', []), frame_data.get('waypoint_idx'), frame_data.get('pos')
                if waypoints: artists[i]['waypoints'].set_data(np.array(waypoints)[:, 1], np.array(waypoints)[:, 0])
                if current_pos is not None: artists[i]['current_pos'].set_data([current_pos[1]], [current_pos[0]])
                if waypoints and waypoint_idx is not None and waypoint_idx < len(waypoints):
                    w_y, w_x = waypoints[waypoint_idx]
                    artists[i]['current_waypoint'].set_data([w_x], [w_y])
                else: artists[i]['current_waypoint'].set_data([], [])
        return []

    ani = animation.FuncAnimation(fig, update, frames=max_frames, blit=False, interval=ANIMATION_INTERVAL, repeat=False)
    
    try:
        print(f"\nSaving video to '{VIDEO_FILENAME}'. This may take a moment...")
        ani.save(VIDEO_FILENAME, writer='ffmpeg', fps=10, dpi=200)
        print(f"✅ Success! Video saved to: {VIDEO_FILENAME}")
    except Exception as e:
        print(f"\n❌ Error saving video: {e}")
        print("   Please ensure FFmpeg is installed and accessible in your system's PATH.")
        print("   You can install it via Homebrew: `brew install ffmpeg`")

    print("\nDisplaying live simulation...")
    plt.show()
    print("\n--- Simulation Finished ---")

if __name__ == '__main__':
    main()
