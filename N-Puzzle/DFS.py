class DFS:
    def __init__(self, graph, root):
        self.graph = graph
        self.visited = dict()
        self.stack = list()
        self.stack.append(root)
        self.counter = 0

    def run(self, target):
        match = False
        depth = 0
        while self.stack:
            current_state = self.stack.pop()
            self.counter += 1
            if current_state.is_equal(target):
                match = True
                depth = current_state.step
                break
            if self.visited.get(current_state.UID) is None:
                self.visited[current_state.UID] = current_state
                neighbor_nodes = self.graph.reveal_neighbors(current_state)
                for neighbor in neighbor_nodes:
                    self.stack.append(neighbor)
        return match, self.counter, depth

