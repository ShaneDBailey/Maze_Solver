# Maze Solver

## Author
Shane Bailey

## File Description
This Python script performs image recognition on a maze image and returns the maze solution as a list of cells, ordered from left to right and top to bottom.

Library Resources:
- pip install numpy
- pip install opencv-python

Important References:
- [OpenCV Documentation](https://docs.opencv.org/4.x/df/d2d/group__ximgproc.html)

## Maze Generator

### Author
Shane Bailey

### File Description
This Python script generates a maze using the minimum spanning tree algorithm. It starts with a grid of cells and their corresponding edges, then finds the minimum spanning tree of the walls using Prim's algorithm. Finally, it draws the maze with OpenCV library.

### Implementation Details
#### Graph Structure
- The script defines a `Graph` class representing a graph composed of vertices (grid cells) and their edges (neighbors with associated weights).

#### Maze Generation
- The `Maze` class creates a grid graph and finds the minimum spanning tree of the walls using Prim's algorithm.
- It then draws the maze by coloring the cells white and drawing the walls of the minimum spanning tree in white.

## Maze Solver

### Author
Shane Bailey

### File Description
This Python script performs image recognition on a maze image and returns the maze solution as a list of cells, ordered from left to right and top to bottom.

### Implementation Details
#### Goal
Given an image of a maze, the script aims to find the solution path.

#### Data
- The original image of the maze.
- A skeleton of the maze obtained by inverting the original image and then thinning it.
- Solution of the maze.
- End points to determine start and stop points of paths for the maze solutions.

#### Functions
- `find_skeleton_path()`: Finds the skeleton of the image.
- `find_exit_points()`: Finds the exits of the maze.
- `solve()`: Solves the maze using the A* algorithm.

#### Constants
- `DIRECTIONS`: Tuple representing 8 possible directions for movement in the maze.

