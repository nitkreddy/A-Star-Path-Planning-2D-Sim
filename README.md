# Self-Driving Car Path Planning Simulator

This project simulates a simple autonomous car navigating a grid world.  
The car plans a path, smooths it, and follows it using a steering controller while avoiding obstacles.

## Features
- Kinematic car model with steering and motion trail  
- Grid-based A*-style path planning  
- Path smoothing for smoother trajectories  
- Pure-pursuit controller for following the path  
- User-placed static and moving obstacles  
- Optional random maze generation  
- Manual and autonomous driving modes  
- Real-time visualization using Pygame  

## Installation
```
pip install pygame numpy
```

## Run
```
python3 path_planning.py
```

## How It Works
- Click to select a destination point  
- A grid search finds a collision-free path  
- The path is smoothed to prevent jerky motion  
- The pure-pursuit algorithm computes steering commands to follow the path  
- The car adjusts speed based on turn sharpness and distance to the goal  
- Obstacles can block the path, in which case the system may replan  

## Notes
- Works best with Python 3.9+  
- Press A to enable or disable autonomous mode  
- Maze generation and dynamic obstacles allow testing different scenarios  
- If a goal is unreachable, try moving the goal or clearing obstacles  
