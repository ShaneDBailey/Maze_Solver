"""
Author: Shane Bailey

File Description:
- 

Library Resources:
- pip install numpy
- pip install opencv-python

Important References:
- https://docs.opencv.org/4.x/df/d2d/group__ximgproc.html

"""

#external libraries
import cv2
import numpy
#python libraries
import heapq
import random

"""
Graph Structure:
    - G(v,E)
    - Composed of vertices IE the grid cells
    - and their edges, IE their neighbors

    - set of vertices with a list of their neighbors and weight associated
"""
class Graph:
    def __init__(self):
        self.vertices = {}  

    def add_vertex(self, vertex):
        if vertex not in self.vertices:
            self.vertices[vertex] = []

    def add_edge(self, vertex1, vertex2, weight):
        if vertex1 in self.vertices and vertex2 in self.vertices:
            self.vertices[vertex1].append((vertex2, weight))
            self.vertices[vertex2].append((vertex1, weight)) 
"""
Maze:
    - keeps track of a grid and its walls
    - the idea is to start with a black canvas
    - make each cell grid white
    - find the minimum spanning tree of the walls
    - make those walls white, visually we know can start from 
    - any cell and make it to any other cell
"""
class Maze:
    def __init__(self,number_of_cells = 25):
        self.number_of_cells = number_of_cells
        self.graph = Graph()
        self.maze_image = None
        self.create_grid_graph()
        self.find_minimum_spanning_tree()
        self.draw_grid()
    #creates the grid graph
    def create_grid_graph(self):
        self.graph = Graph()
        for row in range(self.number_of_cells):
            for column in range(self.number_of_cells):
                vertex = (row, column)
                self.graph.add_vertex(vertex)
                if row > 0:  
                    weight = random.randint(1, self.number_of_cells)  
                    self.graph.add_edge(vertex, (row - 1, column), weight)
                if column > 0: 
                    weight = random.randint(1, self.number_of_cells)
                    self.graph.add_edge(vertex, (row, column - 1), weight)

    def find_minimum_spanning_tree(self):
        start_vertex = next(iter(self.graph.vertices))
        #tracks what edges are part of the tree
        mst_edges = []
        #tracks what cells we have access to
        visited = set([start_vertex])
        #tracks what edges to check through to determien if apart of the mst
        edges = [(weight, start_vertex, to) for to, weight in self.graph.vertices[start_vertex]]
        heapq.heapify(edges)#sorts the edges by weight

        while edges:#while there are edges to check through
            weight, current_cell, looking_at = heapq.heappop(edges)#grab the edge with the lowest weight
            #if edge goes to a cell that has been visited, ignore this as a possible candidate of the mst
            if looking_at not in visited:
                visited.add(looking_at)#add this cell to the set of visited
                mst_edges.append((current_cell, looking_at, weight))#add the edge to the mst tree
                #add the edges in the next cell to the heap
                for next_to, next_weight in self.graph.vertices[looking_at]:
                    if next_to not in visited:
                        heapq.heappush(edges, (next_weight, looking_at, next_to))
        #set the graph edges to that of those in the mst 
        for vertex in self.graph.vertices:
            self.graph.vertices[vertex] = [(neighbor, weight) for neighbor, weight in self.graph.vertices[vertex] if (vertex, neighbor, weight) in mst_edges]

    #start with a black screen and draw the cells on top white, and the walls of the mst_tree on top white
    #
    def draw_grid(self, cell_size=25, wall_size=3):
        image_size = self.number_of_cells * cell_size + (self.number_of_cells + 1) * wall_size
        self.maze_image = numpy.ones((image_size, image_size, 3), dtype=numpy.uint8) * 0 

        for row in range(self.number_of_cells):
            for column in range(self.number_of_cells):
                x = column * (cell_size + wall_size) + wall_size
                y = row * (cell_size + wall_size) + wall_size
                # Draw cell
                cv2.rectangle(self.maze_image, (x, y), (x + cell_size, y + cell_size), ((255, 255, 255)), -1)
                # Draw walls
                for edge, _ in self.graph.vertices[(row, column)]:
                    neighbor_x, neighbor_y = edge
                    if neighbor_x == row and neighbor_y == column + 1:  # Right neighbor
                        cv2.rectangle(self.maze_image, (x + cell_size, y), (x + cell_size + wall_size, y + cell_size), (255, 255, 255), -1)
                    elif neighbor_x == row and neighbor_y == column - 1:  # Left neighbor
                        cv2.rectangle(self.maze_image, (x - wall_size, y), (x, y + cell_size), (255, 255, 255), -1)
                    elif neighbor_x == row + 1 and neighbor_y == column:  # Bottom neighbor
                        cv2.rectangle(self.maze_image, (x, y + cell_size), (x + cell_size, y + cell_size + wall_size), (255, 255, 255), -1)
                    elif neighbor_x == row - 1 and neighbor_y == column:  # Top neighbor
                        cv2.rectangle(self.maze_image, (x, y - wall_size), (x + cell_size, y), (255, 255, 255), -1)
        
        column = random.randint(0, self.number_of_cells - 2)
        x = column * (cell_size + wall_size) + wall_size
        y =  wall_size
        cv2.rectangle(self.maze_image, (x, y - wall_size), (x + cell_size, y), (255,255,255), -1)

        column = random.randint(0, self.number_of_cells - 2)
        x = column * (cell_size + wall_size) + wall_size
        y = image_size 
        cv2.rectangle(self.maze_image, (x, y - wall_size), (x + cell_size, y), (255,255,255), -1)

        self.maze_image = cv2.copyMakeBorder(self.maze_image, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=(255, 255, 255))