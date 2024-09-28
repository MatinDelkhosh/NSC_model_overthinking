import random
import pygame

WALL_COLOR = (0, 0, 0)
PATH_COLOR = (255, 255, 255)
AGENT_COLOR = (0, 0, 255)
GOAL_COLOR = (255, 0, 0)

class Maze:
    def __init__(self, size=8): 
        self.size = size
        self.maze = [[1] * (2 * size + 1) for _ in range(2 * size + 1)]  # 1 indicates a wall, 0 indicates a path
        self.visited = set()
        self.goal_location = None
        self.agent_location = None 
        self.walls = set()

    def neighbors(self, s):
        x, y = s
        potential_neighbors = [(x-2, y), (x+2, y), (x, y-2), (x, y+2)]
        neighbors = []
        
        for nx, ny in potential_neighbors:
            if 0 <= nx < self.size * 2 + 1 and 0 <= ny < self.size * 2 + 1:
                neighbors.append((nx, ny))
        
        return neighbors

    def remove_wall(self, s, n):
        sx, sy = s
        nx, ny = n
        
        if sx == nx:  # Vertical neighbors
            wall_pos = (sx, min(sy, ny) + 1)
            self.maze[wall_pos[1]][wall_pos[0]] = 0
        elif sy == ny:  # Horizontal neighbors
            wall_pos = (min(sx, nx) + 1, sy)
            self.maze[wall_pos[1]][wall_pos[0]] = 0

        self.maze[ny][nx] = 0
        self.walls.discard(wall_pos)

    def walk_maze(self, s):
        self.visited.add(s)
        neighbors = self.neighbors(s)
        random.shuffle(neighbors)
        
        for n in neighbors:
            if n not in self.visited:
                self.remove_wall(s, n)
                self.walk_maze(n)

    def generate_maze(self):
        start = (random.randint(0, self.size - 1) * 2 + 1, random.randint(0, self.size - 1) * 2 + 1)
        self.maze[start[1]][start[0]] = 0
        self.walk_maze(start)

        for x in range(1, self.size * 2, 2):
            for y in range(1, self.size * 2, 2):
                self.walls.add((x, y))

        for _ in range(4):
            if not self.walls:
                break
            wall = random.choice(list(self.walls))
            self.maze[wall[1]][wall[0]] = 0
            self.walls.remove(wall)
        
        self.goal_location = (random.randint(0, self.size - 1) * 2 + 1, random.randint(0, self.size - 1) * 2 + 1)
        
        while True:
            agent_start = (random.randint(0, self.size - 1) * 2 + 1, random.randint(0, self.size - 1) * 2 + 1)
            if agent_start != self.goal_location:
                self.agent_location = agent_start
                break

        return self.maze

    def move_agent(self, direction):
        x, y = self.agent_location
        moved = False
        if direction == 'up' and y > 0 and self.maze[y-1][x] == 0:
            self.agent_location = (x, y-1)
            moved = True
        elif direction == 'down' and y < self.size * 2 and self.maze[y+1][x] == 0:
            self.agent_location = (x, y+1)
            moved = True
        elif direction == 'left' and x > 0 and self.maze[y][x-1] == 0:
            self.agent_location = (x-1, y)
            moved = True
        elif direction == 'right' and x < self.size * 2 and self.maze[y][x+1] == 0:
            self.agent_location = (x+1, y)
            moved = True

        if self.agent_location == self.goal_location:
            return True
        
        return moved

def draw_maze(screen, maze, cell_size):
    for y in range(len(maze.maze)):
        for x in range(len(maze.maze[0])):
            color = WALL_COLOR if maze.maze[y][x] == 1 else PATH_COLOR
            pygame.draw.rect(screen, color, (x * cell_size, y * cell_size, cell_size, cell_size))

    # Goal
    gx, gy = maze.goal_location
    pygame.draw.rect(screen, GOAL_COLOR, (gx * cell_size, gy * cell_size, cell_size, cell_size))

    # Agent
    ax, ay = maze.agent_location
    pygame.draw.ellipse(screen, AGENT_COLOR, (ax * cell_size, ay * cell_size, cell_size, cell_size))

def main():
    pygame.init()

    infoObject = pygame.display.Info()
    screen_width = infoObject.current_w
    screen_height = infoObject.current_h

    rounds = 1
    for current_round in range(rounds):
        maze = Maze()
        maze.generate_maze()

       
        cell_size = min(screen_width // (maze.size * 2 + 1), screen_height // (maze.size * 2 + 1))

        screen = pygame.display.set_mode((screen_width, screen_height), pygame.FULLSCREEN)
        pygame.display.set_caption(f"Maze Game - Round {current_round + 1}")

        running = True
        clock = pygame.time.Clock()

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

            keys = pygame.key.get_pressed()
            moved = False
            if keys[pygame.K_w]:
                moved = maze.move_agent('up')
            elif keys[pygame.K_s]:
                moved = maze.move_agent('down')
            elif keys[pygame.K_a]:
                moved = maze.move_agent('left')
            elif keys[pygame.K_d]:
                moved = maze.move_agent('right')

            if moved:
                screen.fill((0, 0, 0))
                draw_maze(screen, maze, cell_size)
                pygame.display.flip()

            if maze.agent_location == maze.goal_location:
                print(f"Round {current_round + 1} complete! Press Enter to start the next round.")
                waiting_for_next_round = True
                while waiting_for_next_round:
                    for event in pygame.event.get():
                        if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                            waiting_for_next_round = False

                running = False

            clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()