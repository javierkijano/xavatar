import numpy as np
from itertools import combinations

class Node:
    """
        A node class for A* Pathfinding
        parent is parent of the current Node
        position is current position of the Node in the maze
        g is cost from start to current Node
        h is heuristic based estimated cost for current Node to end Node
        f is total cost of present node i.e. :  f = g + h
    """

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position
        self.g = np.Inf
        self.h = np.inf
        self.f = np.inf

    def set_parent(self, parent):
        self.parent = parent

    def __eq__(self, other):
        return self.position == other.position

    def __contains__(self, others):
        return np.any([self.position == other.position for other in others])


class astar_planner:

    def __init__(self, frames, f_heuristic_threhold=0):

        assert (len(frames) > 1)

        self.frames = frames
        self.numFrames = len(frames)
        self.costMatrix_f_hat = np.nan * np.zeros(self.numFrames)
        self.costMatrix_g = np.nan * np.zeros(self.numFrames)
        self.costMatrix_g_inc = np.nan * np.zeros((self.numFrames, self.numFrames))
        self.costMatrix_h = np.nan * np.zeros(self.numFrames)
        self.f_heuristic_threhold = f_heuristic_threhold


    # This function return the path of the search
    def return_path(self, end_node=None):

        if end_node:
            current_node = end_node
        else:
            current_node = self.end_node
        path = []

        path.append(current_node.position)
        while current_node.parent is not None:
            current_node = current_node.parent
            path.append(current_node.position)
        # Return reversed path as we need to show from start to end path
        path = path[::-1]
        return path

    def return_best_path(self):

        # if end_node has been reached then compute path as usual
        if self.end_node.parent:
            return self.return_path(self.end_node)
        visited_nodes = self.computed_nodes + self.open_nodes
        best_node = visited_nodes[np.argmin([node.h for node in visited_nodes])]
        return self.return_path(best_node)

        




    def search(self, start, end, f_heuristic, maxIterations=10, aproximate_frames_number=1):
        """
            Returns a list of tuples as a path from the given start to the given end in the given maze
            :param maze:
            :param cost
            :param start:
            :param end:
            :return:
        """

        # From here we will find the lowest cost node to expand next
        self.open_nodes = []
        self.computed_nodes = []

        # Create start and end node with initized values for g, h and f
        self.start_node = Node(None, start)
        self.start_node.g = 0

        self.end_node = Node(None, end)
        self.end_node.h = 0

        # Add the start node
        current_node = self.start_node

        # Loop until you find the end
        outer_iterations = 0
        self.open_nodes.append(self.start_node)
        while len(self.open_nodes) > 0:

            # Every time any node is referred from yet_to_visit list, counter of limit operation incremented
            outer_iterations += 1

            # Get the current node
            print('... Candidate nodes for expansion  %s: ' % np.array2string(
                np.array([node.position for node in self.open_nodes])))

            current_node = self.open_nodes[np.argmin(
                [node.f for node in self.open_nodes if node.position != self.end_node.position])]

            self.computed_nodes.append(current_node)
            print('... expanding node: %d ' % current_node.position)

            # if we hit this point return the path such as it may be no solution or
            # computation cost is too high
            if outer_iterations > maxIterations:
                print("giving up on pathfinding too many iterations")
                return None
            if current_node == self.end_node and self.end_node.f < self.f_heuristic_threhold:
                print('Cost f min threshold met. Returning suboptimal path')
                return self.return_path(current_node)

            # Pop current node out off yet_to_visit list, add to visited list
            self.open_nodes.pop(np.nonzero([node.position == current_node.position for node in self.open_nodes])[0][0])


            # Generate children from all adjacent squares
            new_positions = list(
                    set(range(self.numFrames)) -
                    set([current_node.position]) -
                    set([node.position for node in self.computed_nodes]))
            for new_position in new_positions:

                # try to get child node from existing open nodes, and create it if not
                child_node = [node for node in self.open_nodes if node.position == new_position]
                if child_node:
                    child_node = child_node[0]
                elif new_position == self.end_node.position:
                    child_node = self.end_node
                else:
                    child_node = Node(current_node, new_position)
                    self.open_nodes.append(child_node)
                print('...child node %d/%d:' % (child_node.position, current_node.position))
                # if child is not already in self.open_nodes and is not the final node


                # heuristic g as a mean of  current-->child & child-->current. Need to do this as
                # the optical flow does not give equal values. with this calculus, it will give the same
                # value. Actually, one could argue that it is better also from estimation perspective
                child_node_update_g_inc = np.mean([
                        f_heuristic(self.frames[current_node.position], self.frames[child_node.position]),
                        f_heuristic(self.frames[child_node.position], self.frames[current_node.position])
                        ])
                self.costMatrix_g_inc[current_node.position, child_node.position] = child_node_update_g_inc
                child_node_update_g = current_node.g + child_node_update_g_inc
                child_node_update_path_length = len(self.return_path(current_node)) - 1 + 1
                child_node_update_g = child_node_update_g/child_node_update_path_length
                if child_node_update_g < child_node.g:
                    child_node.parent = current_node
                    child_node.g = child_node_update_g
                    # do not need to do mean of the inverted combination because the otheh comination
                    # will never be produced
                    child_node.h = f_heuristic(
                        self.frames[child_node.position], self.frames[self.end_node.position])
                    h_normalization_factor = aproximate_frames_number - child_node_update_path_length
                    if h_normalization_factor < 1:
                        h_normalization_factor = 1
                    child_node.h = child_node.h / h_normalization_factor
                    child_node.f = child_node.g + child_node.h
                    self.costMatrix_g[child_node.position] = child_node.g
                    self.costMatrix_h[child_node.position] = child_node.h
                    self.costMatrix_f_hat[child_node.position] = child_node.f
            # log best current path until moment
            print('... best current path: %s\n' % str(self.return_best_path()))

        print("FINISHED after computing all possible paths")
        return self.return_path(self.end_node)