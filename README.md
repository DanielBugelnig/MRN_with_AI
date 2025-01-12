# Project in Mobile Robot Programming using Artificial Intelligence

## Description
All tasks can be executed in a single workspace.

### LSTM for Position Estimaiton

### Object Detection

### Naviagation with RL
already works, aswell with an inference script

```bash
roslaunch my_turtlebot3_openai_example start_training.launch
```

#### Inference
```bash
roslaunch my_turtlebot3_openai_example inference.launch
```


## Setup
1. Clone the repository
2. switch to an own branch 
3. Adjustments in the code (change path): 
   1. /home/danielbugelnig/mobile_robot_navigation_ws3/src/turtlebot3_simulations/turtlebot3_gazebo/worlds/turtlebot3_world_objects_light.world   (multiple times)
   2. /home/danielbugelnig/mobile_robot_navigation_ws3/src/my_turtlebot3_openai_example/config/my_turtlebot3_openai_qlearn_params_v2.yaml
   3. /home/danielbugelnig/mobile_robot_navigation_ws3/src/my_turtlebot3_openai_example/config/my_turtlebot3_openai_deepqlearn_params.yaml
