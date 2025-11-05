import pygame
import numpy as np
import math
import heapq
from dataclasses import dataclass
from typing import List, Tuple, Optional
import random

pygame.init()

WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
GRID_SIZE = 40  
CELL_SIZE = WINDOW_HEIGHT // GRID_SIZE 

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
DARK_GRAY = (50, 50, 50)
LIGHT_GRAY = (200, 200, 200)
ORANGE = (255, 165, 0)
PURPLE = (128, 0, 128)

@dataclass
class CarState:
    x: float
    y: float
    theta: float  
    velocity: float
    steering_angle: float
    
class Car:
    def __init__(self, x: float, y: float, theta: float = 0):
        self.state = CarState(x, y, theta, 0, 0)
        self.length = 20  
        self.width = 10   
        self.max_velocity = 6  
        self.max_steering_angle = math.pi / 4  
        self.wheelbase = 15  
        self.trail = []  
        self.max_trail_length = 100
        
    def update(self, dt: float, target_velocity: float, target_steering: float):
        target_velocity = np.clip(target_velocity, -self.max_velocity, self.max_velocity)
        target_steering = np.clip(target_steering, -self.max_steering_angle, self.max_steering_angle)
        
        self.state.velocity = target_velocity
        self.state.steering_angle = target_steering
        
        if abs(self.state.velocity) > 0.01:  
            if abs(self.state.steering_angle) > 0.001:
                turning_radius = self.wheelbase / math.tan(self.state.steering_angle)
                angular_velocity = self.state.velocity / turning_radius
                
                self.state.theta += angular_velocity * dt
                self.state.x += self.state.velocity * math.cos(self.state.theta) * dt
                self.state.y += self.state.velocity * math.sin(self.state.theta) * dt
            else:
                self.state.x += self.state.velocity * math.cos(self.state.theta) * dt
                self.state.y += self.state.velocity * math.sin(self.state.theta) * dt
            
        self.state.theta = (self.state.theta + math.pi) % (2 * math.pi) - math.pi
        
        self.trail.append((self.state.x, self.state.y))
        if len(self.trail) > self.max_trail_length:
            self.trail.pop(0)
    
    def draw(self, screen: pygame.Surface):
        if len(self.trail) > 1:
            for i in range(1, len(self.trail)):
                alpha = i / len(self.trail)
                color = (int(255 * alpha), int(165 * alpha), 0)
                pygame.draw.line(screen, color, self.trail[i-1], self.trail[i], 2)
        
        corners = [
            (-self.length/2, -self.width/2),
            (self.length/2, -self.width/2),
            (self.length/2, self.width/2),
            (-self.length/2, self.width/2)
        ]
        
        rotated_corners = []
        for cx, cy in corners:
            rx = cx * math.cos(self.state.theta) - cy * math.sin(self.state.theta)
            ry = cx * math.sin(self.state.theta) + cy * math.cos(self.state.theta)
            rotated_corners.append((self.state.x + rx, self.state.y + ry))
        
        pygame.draw.polygon(screen, BLUE, rotated_corners)
        
        front_x = self.state.x + (self.length/2) * math.cos(self.state.theta)
        front_y = self.state.y + (self.length/2) * math.sin(self.state.theta)
        pygame.draw.circle(screen, YELLOW, (int(front_x), int(front_y)), 3)

class PathPlanner:
    def __init__(self, grid_size: int):
        self.grid_size = grid_size
        self.obstacles = set()
        self.path = []
        self.smooth_path = []
        
    def add_obstacle(self, x: int, y: int):
        self.obstacles.add((x, y))
        
    def remove_obstacle(self, x: int, y: int):
        self.obstacles.discard((x, y))
        
    def is_collision(self, x: float, y: float, car_length: float = 20, car_width: float = 10) -> bool:
        grid_x = int(x // CELL_SIZE)
        grid_y = int(y // CELL_SIZE)
        
        if grid_x < 0 or grid_x >= self.grid_size or grid_y < 0 or grid_y >= self.grid_size:
            return True
        
        if (grid_x, grid_y) in self.obstacles:
            return True
            
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                check_x = grid_x + dx
                check_y = grid_y + dy
                if (check_x, check_y) in self.obstacles:
                    obs_x = check_x * CELL_SIZE + CELL_SIZE/2
                    obs_y = check_y * CELL_SIZE + CELL_SIZE/2
                    dist = math.sqrt((x - obs_x)**2 + (y - obs_y)**2)
                    if dist < CELL_SIZE * 0.8: 
                        return True
        return False
    
    def heuristic(self, pos: Tuple[float, float], goal: Tuple[float, float]) -> float:
        return math.sqrt((pos[0] - goal[0])**2 + (pos[1] - goal[1])**2)
    
    def find_path(self, start: Tuple[float, float], goal: Tuple[float, float]) -> List[Tuple[float, float]]:
        start_grid = (int(start[0] // CELL_SIZE), int(start[1] // CELL_SIZE))
        goal_grid = (int(goal[0] // CELL_SIZE), int(goal[1] // CELL_SIZE))
        
        if goal_grid in self.obstacles:
            print("Goal is in an obstacle!")
            return []
        
        open_set = [(0, 0, start_grid, [start_grid])]
        visited = set()
        
        while open_set:
            f_score, g_score, current, path = heapq.heappop(open_set)
            
            if current in visited:
                continue
                
            visited.add(current)
            
            if current == goal_grid:
                world_path = [(p[0] * CELL_SIZE + CELL_SIZE/2, 
                              p[1] * CELL_SIZE + CELL_SIZE/2) for p in path]
                self.path = world_path
                self.smooth_path = self.smooth_path_cubic(world_path)
                return self.smooth_path
            
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (1,1), (-1,1), (1,-1)]:
                next_pos = (current[0] + dx, current[1] + dy)
                
                if (next_pos[0] < 0 or next_pos[0] >= self.grid_size or
                    next_pos[1] < 0 or next_pos[1] >= self.grid_size):
                    continue
                
                if next_pos in self.obstacles:
                    continue
                    
                if abs(dx) == 1 and abs(dy) == 1:
                    if (current[0] + dx, current[1]) in self.obstacles and \
                       (current[0], current[1] + dy) in self.obstacles:
                        continue  
                
                if next_pos in visited:
                    continue
                
                move_cost = math.sqrt(dx*dx + dy*dy) * CELL_SIZE
                new_g_score = g_score + move_cost
                h_score = self.heuristic(
                    (next_pos[0] * CELL_SIZE, next_pos[1] * CELL_SIZE),
                    (goal_grid[0] * CELL_SIZE, goal_grid[1] * CELL_SIZE)
                )
                new_f_score = new_g_score + h_score
                
                heapq.heappush(open_set, 
                             (new_f_score, new_g_score, next_pos, path + [next_pos]))
        
        print("No path found!")
        return []  
    
    def smooth_path_cubic(self, path: List[Tuple[float, float]], 
                         smoothness: float = 0.3) -> List[Tuple[float, float]]:
        if len(path) < 3:
            return path
            
        smooth_path = [path[0]]
        
        for i in range(1, len(path) - 1):
            prev_point = path[i - 1]
            curr_point = path[i]
            next_point = path[i + 1]
            
            smooth_x = curr_point[0] + smoothness * (
                (prev_point[0] + next_point[0]) / 2 - curr_point[0]
            )
            smooth_y = curr_point[1] + smoothness * (
                (prev_point[1] + next_point[1]) / 2 - curr_point[1]
            )
            
            if not self.is_collision(smooth_x, smooth_y):
                smooth_path.append((smooth_x, smooth_y))
            else:
                smooth_path.append(curr_point)
        
        smooth_path.append(path[-1])
        return smooth_path

class PurePursuit:
    def __init__(self, lookahead_distance: float = 30):  
        self.lookahead_distance = lookahead_distance
        
    def calculate_control(self, car_state: CarState, path: List[Tuple[float, float]]) -> Tuple[float, float]:
        if not path:
            return 0, 0
        
        min_dist = float('inf')
        closest_idx = 0
        for i, point in enumerate(path):
            dist = math.sqrt((point[0] - car_state.x)**2 + (point[1] - car_state.y)**2)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        lookahead_idx = min(closest_idx + 5, len(path) - 1)  
        lookahead_point = path[lookahead_idx]
        
        if len(path) - closest_idx < 3:
            lookahead_point = path[-1]
        
        dx = lookahead_point[0] - car_state.x
        dy = lookahead_point[1] - car_state.y
        
        angle_to_target = math.atan2(dy, dx)
        angle_diff = angle_to_target - car_state.theta
        
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
        
        steering_angle = angle_diff * 0.8 
        
        goal_dist = math.sqrt((path[-1][0] - car_state.x)**2 + (path[-1][1] - car_state.y)**2)
        
        if goal_dist < 25:
            target_velocity = 2.0  
        elif goal_dist < 50:
            target_velocity = 3.0
        elif abs(steering_angle) > 0.5:
            target_velocity = 2.5
        elif abs(steering_angle) > 0.3:
            target_velocity = 3.5
        else:
            target_velocity = 4.5
        
        return target_velocity, steering_angle

class Simulator:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Self-Driving Car Path Planning")
        self.clock = pygame.time.Clock()
        
        self.car = Car(60, WINDOW_HEIGHT - 60) 
        self.path_planner = PathPlanner(GRID_SIZE)
        self.controller = PurePursuit()
        
        self.goal = None
        self.path = []
        self.running = True
        self.paused = False
        self.show_grid = True
        self.autonomous_mode = False
        self.dynamic_obstacles = []
        
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        
    def add_dynamic_obstacle(self):
        x = random.randint(5, GRID_SIZE - 5) * CELL_SIZE
        y = random.randint(5, GRID_SIZE - 5) * CELL_SIZE
        vx = random.uniform(-2, 2)
        vy = random.uniform(-2, 2)
        self.dynamic_obstacles.append({'x': x, 'y': y, 'vx': vx, 'vy': vy})
        
    def update_dynamic_obstacles(self, dt: float):
        for obs in self.dynamic_obstacles:
            obs['x'] += obs['vx']
            obs['y'] += obs['vy']
            
            if obs['x'] < 0 or obs['x'] > WINDOW_WIDTH:
                obs['vx'] *= -1
            if obs['y'] < 0 or obs['y'] > WINDOW_HEIGHT:
                obs['vy'] *= -1
                
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_g:
                    self.show_grid = not self.show_grid
                elif event.key == pygame.K_a:
                    self.autonomous_mode = not self.autonomous_mode
                    print(f"Autonomous mode: {self.autonomous_mode}")
                    if self.autonomous_mode and self.goal:
                        self.path = self.path_planner.find_path(
                            (self.car.state.x, self.car.state.y),
                            self.goal
                        )
                        if self.path:
                            print(f"Path calculated with {len(self.path)} waypoints")
                        else:
                            print("No path found!")
                elif event.key == pygame.K_c:
                    self.path_planner.obstacles.clear()
                    self.dynamic_obstacles.clear()
                elif event.key == pygame.K_r:
                    self.car = Car(60, WINDOW_HEIGHT - 60)
                    self.car.trail.clear()
                    self.goal = None
                    self.path = []
                    print("Car reset to start position")
                elif event.key == pygame.K_d:
                    self.add_dynamic_obstacle()
                elif event.key == pygame.K_m:
                    self.generate_maze()
                    
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                grid_x = x // CELL_SIZE
                grid_y = y // CELL_SIZE
                
                if event.button == 1:  
                    self.goal = (x, y)
                    self.path = self.path_planner.find_path(
                        (self.car.state.x, self.car.state.y),
                        self.goal
                    )
                    if not self.path:
                        print("No path found to goal! Try a different location.")
                    else:
                        print(f"Path found with {len(self.path)} waypoints")
                elif event.button == 3:
                    if (grid_x, grid_y) in self.path_planner.obstacles:
                        self.path_planner.remove_obstacle(grid_x, grid_y)
                    else:
                        self.path_planner.add_obstacle(grid_x, grid_y)
                    if self.goal:
                        self.path = self.path_planner.find_path(
                            (self.car.state.x, self.car.state.y),
                            self.goal
                        )
    
    def generate_maze(self):
        self.path_planner.obstacles.clear()
        
        for _ in range(int(GRID_SIZE * 1.5)):  
            x = random.randint(0, GRID_SIZE - 1)
            y = random.randint(0, GRID_SIZE - 1)
            if not (x < 4 and y > GRID_SIZE - 4):
                self.path_planner.add_obstacle(x, y)
                
        for i in range(GRID_SIZE):
            if random.random() > 0.7:
                for j in range(random.randint(3, 8)):
                    if i + j < GRID_SIZE:
                        self.path_planner.add_obstacle(i + j, random.randint(0, GRID_SIZE - 1))
                        
    def update(self, dt: float):
        if self.paused:
            return
            
        self.update_dynamic_obstacles(dt)
        
        keys = pygame.key.get_pressed()
        
        if self.autonomous_mode and self.path:
            velocity, steering = self.controller.calculate_control(self.car.state, self.path)
            
            front_check_x = self.car.state.x + 25 * math.cos(self.car.state.theta)
            front_check_y = self.car.state.y + 25 * math.sin(self.car.state.theta)
            
            if self.path_planner.is_collision(front_check_x, front_check_y):
                velocity *= 0.5 
                self.path = self.path_planner.find_path(
                    (self.car.state.x, self.car.state.y),
                    self.goal
                )
                if not self.path:
                    print("Path blocked! No alternative found.")
                    velocity = 0
            
            self.car.update(dt, velocity, steering)
            
            if self.goal:
                dist_to_goal = math.sqrt((self.car.state.x - self.goal[0])**2 + 
                                        (self.car.state.y - self.goal[1])**2)
                if dist_to_goal < 20:  
                    self.goal = None
                    self.path = []
                    self.car.state.velocity = 0  
                    print("Goal reached!")
        else:
            velocity = 0
            steering = 0
            
            if keys[pygame.K_UP]:
                velocity = 4
            if keys[pygame.K_DOWN]:
                velocity = -4
            if keys[pygame.K_LEFT]:
                steering = -0.8
            if keys[pygame.K_RIGHT]:
                steering = 0.8
                
            self.car.update(dt, velocity, steering)
        
        for obs in self.dynamic_obstacles:
            dist = math.sqrt((self.car.state.x - obs['x'])**2 + 
                           (self.car.state.y - obs['y'])**2)
            if dist < 30:
                if self.goal and self.autonomous_mode:
                    self.path = self.path_planner.find_path(
                        (self.car.state.x, self.car.state.y),
                        self.goal
                    )
                self.car.state.velocity *= -0.5
    
    def draw(self):
        self.screen.fill(WHITE)
        
        if self.show_grid:
            for x in range(0, WINDOW_WIDTH, CELL_SIZE):
                pygame.draw.line(self.screen, LIGHT_GRAY, (x, 0), (x, WINDOW_HEIGHT), 1)
            for y in range(0, WINDOW_HEIGHT, CELL_SIZE):
                pygame.draw.line(self.screen, LIGHT_GRAY, (0, y), (WINDOW_WIDTH, y), 1)
        
        for obstacle in self.path_planner.obstacles:
            x, y = obstacle
            pygame.draw.rect(self.screen, DARK_GRAY,
                           (x * CELL_SIZE + 2, y * CELL_SIZE + 2, CELL_SIZE - 4, CELL_SIZE - 4))
        
        for obs in self.dynamic_obstacles:
            pygame.draw.circle(self.screen, PURPLE, 
                             (int(obs['x']), int(obs['y'])), 15)
        
        if self.path_planner.smooth_path and len(self.path_planner.smooth_path) > 1:
            if len(self.path_planner.path) > 1:
                pygame.draw.lines(self.screen, (255, 200, 200), False, 
                                self.path_planner.path, 2)
            
            pygame.draw.lines(self.screen, GREEN, False, 
                            self.path_planner.smooth_path, 3)
            
            for point in self.path_planner.smooth_path[::5]:
                pygame.draw.circle(self.screen, GREEN, 
                                 (int(point[0]), int(point[1])), 4)
        
        if self.goal:
            pygame.draw.circle(self.screen, RED, self.goal, 10)
            pygame.draw.circle(self.screen, RED, self.goal, 20, 2)
        
        self.car.draw(self.screen)
        
        self.draw_ui()
        
    def draw_ui(self):
        panel_rect = pygame.Rect(10, 10, 380, 220)  
        pygame.draw.rect(self.screen, (240, 240, 240), panel_rect)
        pygame.draw.rect(self.screen, BLACK, panel_rect, 2)
        
        y_offset = 20
        
        title = self.font.render("Self-Driving Car Simulator", True, BLACK)
        self.screen.blit(title, (20, y_offset))
        y_offset += 40
        
        status_texts = [
            f"Mode: {'Autonomous' if self.autonomous_mode else 'Manual'}",
            f"Velocity: {self.car.state.velocity:.1f}",
            f"Steering: {math.degrees(self.car.state.steering_angle):.1f}°",
            f"Position: ({int(self.car.state.x)}, {int(self.car.state.y)})",
            f"Dynamic Obstacles: {len(self.dynamic_obstacles)}",
            f"Path Points: {len(self.path) if self.path else 0}",
            f"Status: {'PAUSED' if self.paused else 'RUNNING'}"
        ]
        
        for text in status_texts:
            rendered = self.small_font.render(text, True, BLACK)
            self.screen.blit(rendered, (20, y_offset))
            y_offset += 25
        
        control_texts = [
            "Controls:",
            "Left Click: Set Goal",
            "Right Click: Add/Remove Obstacle",
            "Arrow Keys: Manual Drive",
            "A: Toggle Autonomous Mode",
            "D: Add Dynamic Obstacle",
            "M: Generate Maze",
            "C: Clear Obstacles",
            "R: Reset Car",
            "G: Toggle Grid",
            "Space: Pause"
        ]
        
        y_offset = 250  
        for text in control_texts:
            rendered = self.small_font.render(text, True, BLACK)
            self.screen.blit(rendered, (20, y_offset))
            y_offset += 22
    
    def run(self):
        while self.running:
            dt = 0.5  
            
            self.handle_events()
            self.update(dt)
            self.draw()
            
            self.clock.tick(60)  
            pygame.display.flip()
        
        pygame.quit()

if __name__ == "__main__":
    print("Starting Simulator...")
    print("This demonstrates path planning algorithms for autonomous navigation")
    print("=" * 60)
    
    simulator = Simulator()
    simulator.run()