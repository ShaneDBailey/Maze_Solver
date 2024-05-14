import cv2
import numpy
"""
TODO: solve a maze via image
grab contours find the outer most contour and extrapulate the exits of the maze

solved_maze store
"""
DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))
class Maze:

    def __init__(self, image):
        self.orginal_maze = image
        self.skeleton = None
        self.solution_maze = None
        self.start_point = None
        self.end_point = None

    def find_skeleton_path(self, inset = 1):
        gray = cv2.cvtColor(self.orginal_maze, cv2.COLOR_BGR2GRAY)
        _, black_white_bin = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

        mask = numpy.ones_like(black_white_bin, dtype=numpy.uint8) * 255
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

    def heuristic(self, a, b):
        return numpy.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

    def reconstruct_path(self, came_from, current):
        total_path = [current]
        while current in came_from.keys():
            current = came_from[current]
            total_path.insert(0, current)
        return total_path
    
    def get_neighbors(self, node):
        x, y = node
        neighbors = []
        for direction_x, direction_y in DIRECTIONS:
            neighbor_x, neighbor_y = x + direction_x, y + direction_y
            if 0 <= neighbor_x < self.skeleton.shape[1] and 0 <= neighbor_y < self.skeleton.shape[0] and self.skeleton[neighbor_y, neighbor_x] == 255:
                neighbors.append((neighbor_x, neighbor_y))
        return neighbors
    
    def solve(self):
        open_set = {self.start_point}
        came_from = {}
        g_score = {self.start_point: 0}
        f_score = {self.start_point: self.heuristic(self.start_point, self.end_point)}

        while open_set:
            current = min(open_set, key=lambda x: f_score.get(x, float('inf')))
            if current == self.end_point:
                path = self.reconstruct_path(came_from, current)
                # Mark the solution path in the solution_maze
                self.solution_maze = numpy.copy(self.orginal_maze)
                for point in path:
                    cv2.circle(self.solution_maze, point, 2, (0, 0, 255), -1)
                return path

            open_set.remove(current)
            for neighbor in self.get_neighbors(current):
                tentative_g_score = g_score[current] + self.heuristic(current, neighbor)
                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, self.end_point)
                    if neighbor not in open_set:
                        open_set.add(neighbor)

        return None

if __name__ == "__main__":
    image = cv2.imread("maze_test.jpg")

    maze = Maze(image)
    maze.find_skeleton_path()
    maze.find_end_points()
    maze.solve()
    cv2.imshow('Solution Maze', maze.solution_maze)


cv2.waitKey(0)
cv2.destroyAllWindows()