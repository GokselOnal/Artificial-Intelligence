import queue as Q


class UCS:
    def __init__(self, graph, root):
        self.graph = graph
        self.visited = dict()
        self.queue = Q.PriorityQueue()
        self.counter = 0
        self.visited[root.UID] = root
        self.queue.put((root.step, root, int(root.UID)))

    def run(self, target):
        match = False
        depth = 0
        while self.queue:
            current_state = self.queue.get()[1]
            self.counter += 1
            if current_state.is_equal(target):
                match = True
                depth = current_state.step
                break
            if self.visited.get(current_state.UID) is None:
                self.visited[current_state.UID] = current_state
            neighbor_nodes = self.graph.reveal_neighbors(current_state)
            for neighbor in neighbor_nodes:
                self.queue.put((neighbor.step, neighbor, int(neighbor.UID)))
        return match, self.counter, depth
