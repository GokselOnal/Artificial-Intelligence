class BFS:
    def __init__(self, graph, root):
        self.graph = graph
        self.visited = dict()
        self.queue = list()
        self.counter = 0
        self.visited[root.UID] = root
        self.queue.append(root)

    def run(self, target):
        match = False
        depth = 0
        while self.queue:
            current_state = self.queue.pop(0)
            self.counter += 1
            if current_state.is_equal(target):
                match = True
                depth = current_state.step
                break
            neighbor_nodes = self.graph.reveal_neighbors(current_state)
            for neighbor in neighbor_nodes:
                if self.visited.get(neighbor.UID) is None:
                    self.visited[neighbor.UID] = neighbor
                    self.queue.append(neighbor)
        return match, self.counter, depth
