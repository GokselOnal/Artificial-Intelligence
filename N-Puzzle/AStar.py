import queue as Q


class AStar:
    def __init__(self, graph, root):
        self.graph = graph
        self.root = root
        self.visited = dict()
        self.queue = Q.PriorityQueue()
        self.counter = 0

    def run(self, target):
        self.queue.put((self.root.step, self.root, int(self.root.UID)))
        while self.queue:
            current_state = self.queue.get()[1]
            self.counter += 1
            if current_state.is_equal(target):
                return True, self.counter, current_state.step
            if self.visited.get(current_state.UID) is None:
                self.visited[current_state.UID] = current_state
                neighbor_nodes = self.graph.reveal_neighbors(current_state)
                for neighbor in neighbor_nodes:
                    self.queue.put((neighbor.step + self.manhattan_distance(neighbor,target), neighbor, int(neighbor.UID)))
        return False, self.counter, 0

    def manhattan_distance(self, node, end):
        arr = [0] * (self.graph.size + 1)
        brr = [0] * (self.graph.size + 1)
        for i in range(len(node.g_node)):
            for j in range(len(node.g_node[i])):
                arr[node.g_node[i][j]] = [i, j]

        for i in range(len(end.g_node)):
            for j in range(len(end.g_node[i])):
                brr[end.g_node[i][j]] = [i, j]
        dist = 0
        for i in range(1, len(arr)):
            dist += abs(arr[i][0] - brr[i][0]) + abs(arr[i][1] - brr[i][1])
        return dist
