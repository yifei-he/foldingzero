class Node:
    def __init__(self, state, action=None, prob=1, parent=None):
        self.visit = 0
        self.total_reward = 0
        self.mean_reward = 0
        self.prob = prob
        self.children = []
        self.parent = parent
        self.action = action
        self.state = state
        if self.parent is not None:
            self.depth = self.parent.depth + 1
        else:
            self.depth = 1

    def is_leaf(self):
        if len(self.children) == 0:
            return True
        return False

    def backprop(self, reward):
        parent_node = self
        while parent_node is not None:
            parent_node.total_reward += reward
            parent_node.visit += 1
            parent_node.mean_reward = parent_node.total_reward / parent_node.visit
            parent_node = parent_node.parent
