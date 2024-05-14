import cv2
import numpy as np
from heapq import heappush, heappop
"""
TODO: solve a maze via image
grab contours find the outer most contour and extrapulate the exits of the maze


maze class
start point
end point
maze
altered_maze
solved_maze store
"""

class Maze:

    def __init__(self, image):
        self.orginal_maze = image
        self.skeleton = None
        self.solution_maze = None
        self.start_point = None
        self.end_point = None

    def find_outer_bounds(self):
        gray = cv2.cvtColor(self.orginal_maze, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(self.orginal_maze, [contours[0]], -1, (0, 255, 0), 2)
        cv2.imshow('Image with Contours', self.orginal_maze)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def find_skeleton_path(self, inset = 1):
        gray = cv2.cvtColor(self.orginal_maze, cv2.COLOR_BGR2GRAY)
        _, black_white_bin = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

        mask = np.ones_like(black_white_bin, dtype=np.uint8) * 255
        mask[:inset, :] = 0  # Top border
        mask[-inset:, :] = 0  # Bottom border
        mask[:, :inset] = 0  # Left border
        mask[:, -inset:] = 0  # Right border

        self.skeleton = cv2.ximgproc.thinning(~black_white_bin)
        self.skeleton = cv2.bitwise_and(self.skeleton, self.skeleton, mask=mask)
        
        cv2.imshow('Skeleton Path', self.skeleton)

    def find_end_points(self, inset = 5):
        height, width = self.skeleton.shape
        end_points = []

        # Iterate over the top and bottom edges (with inset)
        for x in range(inset, width - inset):
            if self.skeleton[inset, x] == 255:
                end_points.append((x, inset))
            if self.skeleton[height - (inset +1), x] == 255:
                end_points.append((x, height - 6))

        # Iterate over the left and right edges (with inset)
        for y in range(inset, height - inset):
            if self.skeleton[y, inset] == 255:
                end_points.append((inset, y))
            if self.skeleton[y, width - (inset + 1)] == 255:
                end_points.append((width - (inset + 1), y))

        # Set the start point as the first end point and the end point as the second end point
        self.start_point = end_points[0]
        self.end_point = end_points[1]
        print(len(end_points))
        print(self.start_point)
        print(self.end_point)


if __name__ == "__main__":
    image = cv2.imread("maze2.jpg")

    maze = Maze(image)
    maze.find_skeleton_path()
    maze.find_end_points()
    #maze.solve()


cv2.waitKey(0)
cv2.destroyAllWindows()