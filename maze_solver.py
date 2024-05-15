"""
Author: Shane Bailey

File Description:
- This file serves to do image recongnition and return a sudoku board
- It returns the board as a list of cells, order from left to right: top to bottom

Library Resources:
- pip install numpy
- pip install opencv-python
- pip install tensorflow

Important References:
- https://docs.opencv.org/4.x/df/d2d/group__ximgproc.html

"""
#external libraries
import cv2
import numpy
#--------------------------------Constants-----------------------------------------
DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))
#--------------------------------Maze_Class----------------------------------------
"""
Goal:
Given an image of a maze be able to solve it

Data:
- The orginal iamge
- A skeleton of the image obtained by inverting the orginal and then thinning
    - Serves to have a centerized line in the maze paths
- Solution of the maze
- End points to determine start and stop points of paths for the maze solutions

Functions:
- find_skeleton_path: finds the skeleton of the image
- find_exit_points: finds the exits of the maze

- solve: solves the maze
    - heuristic: returns a heuristic value
    - get_neighbors: returns the surrounding valid neighbors
"""

#TODO: default constructor for this class that makes a 500 by 500 pixel maze
class Maze:
    #----------------------------Initializer----------------------------------------
    def __init__(self, image):
        self.orginal_maze = image
        self.skeleton = None
        self.solution_maze = None
        self.exit_points = []
    #----------------------------Finders--------------------------------------------
    def find_skeleton_path(self, inset = 1):
        #gray scale and then threshold to get a pure black and white image
        gray = cv2.cvtColor(self.orginal_maze, cv2.COLOR_BGR2GRAY)
        _, black_white_bin = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)

        #inverts the image, and then scales down the white pixel sections to 
        self.skeleton = cv2.ximgproc.thinning(~black_white_bin)

        #Sets the border pixels to black base on the inset
            # some images have 
        mask = numpy.ones_like(black_white_bin, dtype=numpy.uint8) * 255
        mask[:inset, :] = 0  # Top border
        mask[-inset:, :] = 0  # Bottom border
        mask[:, :inset] = 0  # Left border
        mask[:, -inset:] = 0  # Right border
        self.skeleton = cv2.bitwise_and(self.skeleton, self.skeleton, mask=mask)
        
        cv2.imshow('Skeleton Path', self.skeleton)#for debugging

    def find_exit_points(self, inset = 1):
        height, width = self.skeleton.shape

        # Iterate over the top and bottom edges to find end points
        for x in range(inset, width - inset):
            if self.skeleton[inset, x] == 255:
                self.exit_points.append((x, inset))
            if self.skeleton[height - (inset +1), x] == 255:
                self.exit_points.append((x, height - 6))

        # Iterate over the left and right edges to find end points
        for y in range(inset, height - inset):
            if self.skeleton[y, inset] == 255:
                self.exit_points.append((inset, y))
            if self.skeleton[y, width - (inset + 1)] == 255:
                self.exit_points.append((width - (inset + 1), y))
    #----------------------------Solver---------------------------------------------
    #assign a heursitic value to a give point
    def distance_away(self, a, b):
        return numpy.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)# 
    #returns the valid neighbors, the 8 directions, plus withinside the image
    def get_neighbors(self, node):
        x, y = node
        neighbors = []

        for direction_x, direction_y in DIRECTIONS:
            neighbor_x, neighbor_y = x + direction_x, y + direction_y
            if 0 <= neighbor_x < self.skeleton.shape[1] and 0 <= neighbor_y < self.skeleton.shape[0] and self.skeleton[neighbor_y, neighbor_x] == 255:
                neighbors.append((neighbor_x, neighbor_y))

        return neighbors
    #returns the path by looking at where the current node or pixel came from
    #as each pixel will only ever orginate from one parent, this way we can walk
    #from the end back to the start
    def reconstruct_path(self, parent_pixel, current):
        solved_path = [current]

        while current in parent_pixel.keys():
            current = parent_pixel[current]
            solved_path.insert(0, current)

        return solved_path
    
    def solve(self):
        solutions = []
        
        for start_point in range(len(self.exit_points)):
            for end_point in range(start_point + 1, len(self.exit_points)):
                start_point = self.exit_points[start_point]
                end_point = self.exit_points[end_point]

                #tracks the parent for every valid pixel
                parent_pixel = {}
                #tracks the pixel/paths to explore, when checking a pixel to be apart of the valid path
                #remove it, the current node neighbors are added to the paths to explore
                paths_to_explore = {start_point}
                #tracks the cost from the start node to the given pixel in the set
                cost_from_start = {start_point: 0}#also known as g score
                #gives a position a prediction cost, prediction_cost = cost from start + distance from end_point
                cost_from_prediction = {start_point: self.distance_away(start_point, end_point)}#also known as f score


                while paths_to_explore:
                    current = min(paths_to_explore , key=lambda x: cost_from_prediction.get(x, float('inf')))
                    paths_to_explore.remove(current)
                    
                    if current == end_point:
                        path = self.reconstruct_path(parent_pixel, current)
                        solutions.append(path)
                        break

                    for neighbor in self.get_neighbors(current):
                        if neighbor not in cost_from_start:#if not cost_from_start[neighbor]
                            parent_pixel[neighbor] = current
                            cost_from_start[neighbor] = cost_from_start[current] + self.distance_away(current, neighbor)
                            cost_from_prediction[neighbor] = cost_from_start[neighbor] + self.distance_away(neighbor, end_point)
                            if neighbor not in paths_to_explore:
                                paths_to_explore.add(neighbor)

        # Mark the solution paths in the solution_maze
        self.solution_maze = numpy.copy(self.orginal_maze)
        for path in solutions:
            for point in path:
                cv2.circle(self.solution_maze, point, 2, (0, 0, 255), -1)

        return None

if __name__ == "__main__":
    image = cv2.imread("maze2.jpg")

    maze = Maze(image)
    maze.find_skeleton_path()
    maze.find_exit_points()
    maze.solve()
    cv2.imshow('Solution Maze', maze.solution_maze)


cv2.waitKey(0)
cv2.destroyAllWindows()