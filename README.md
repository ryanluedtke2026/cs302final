# Chrysalis Robot Simulation

## Overview
Chrysalis is a simulated robotic system inspired by caterpillars, designed to evolve and improve its locomotion over multiple generations. This simulation leverages **DiffTaichi**, a differentiable physics engine, to iteratively optimize the robot's parameters, mimicking an evolutionary process.

## Features
- **Taichi-based Simulation:** Uses Taichi to efficiently compute physics-based simulations.
- **Evolutionary Optimization:** Implements an iterative mutation strategy to improve performance over generations.
- **Neural Network Control:** A simple neural network determines the actuation of springs within the robot.
- **Visualization & Logging:** Generates simulation videos and logs key parameters for analysis.

## Installation
Ensure you have the required dependencies installed:
```sh
pip install taichi numpy matplotlib pandas
```

## Running the Simulation
To start the evolutionary optimization:
```sh
python chrysalis_robot.py
```
This will run multiple generations of robot optimization and save the results in `evolution_json_data/`.

## Evolutionary Process
1. **Initialization**: A base robot with specific parameters is created.
2. **Mutation & Selection**: New generations introduce mutations to stiffness, branching, and segment length parameters.
3. **Simulation & Optimization**: Each variant undergoes a physics simulation to compute movement efficiency.
4. **Best Selection**: The most efficient robots are selected to form the next generation.

## File Structure
- `chrysalis_robot.py`: Main script that runs the simulation.
- `evolution_json_data/`: Stores parameter logs for each simulation run.
- `rigid_body/`: Contains visualization frames of robot movement.
- `simulation.log`: Logs error messages and debugging information.

## Key Functions
- **`build_robot_skeleton_vii()`**: Constructs the robot based on branching and segment parameters.
- **`optimize()`**: Runs Taichi-based optimization to improve movement efficiency.
- **`evolutionary_optimization()`**: Iteratively refines the robot’s parameters over multiple generations.
- **`evaluate_robot()`**: Simulates an individual robot’s movement and logs results.
- **`plot_gen_losses()`**: Generates plots of loss values across generations.

## Visualizing Results
Once the simulation completes, you can visualize results by generating a video:
```sh
ffmpeg -framerate 30 -i rigid_body/generation1_robot0/%04d.png -c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p output.mp4
```
Alternatively, you can inspect parameter impact:
```sh
python plot_parameter_impact.py
```

## Future Improvements
- **Refining Neural Network Architecture**: Enhance control algorithms for better adaptation.
- **Incorporating More Environmental Factors**: Introduce obstacles and varied terrains.
- **Real-world Deployment**: Investigate hardware implementation for rover applications.

## Credits
Created by **Ryan Luedtke**, adapted from provided code from https://github.com/taichi-dev/difftaichi as part of an evolutionary robotics project. Inspired by natural movement strategies in biological organisms.
