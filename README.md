# 📌 Project: Mobile Robot Programming using Artificial Intelligence

## Description
This project involves tasks related to mobile robots, such as position estimation, object detection, and navigation using reinforcement learning. All tasks can be executed in a single workspace.

---
### LSTM for Position Estimation
**Files Location:** `lstm_trajectory_prediction` package.
- `extract_imu_data_2.py`: Collects IMU data for training the LSTM model.
- `inference.py`: Listens to the IMU data from the TurtleBot and outputs a position estimation.
- `model.py`: Script for training and testing the LSTM model.

**Output Topic:** `/predicted_pose`

**⚠️ Notes:**
- Errors may occur due to fluctuating IMU topic frequency during recording and operation.
- The initial position updates only once when the script starts.

---
### Object Detection with ResNet
**Files Location:** `image_classification` package.

**Datasets:**
- `ycbv_classification`: Standard dataset provided by Prof. Steinbrener.
- `ycbv_classification_new`: Dataset obtained directly from the TurtleBot environment.
- `original dataset`: For YCBV objects (not included in this repository).

**Scripts:**
- `res_net.py`: ResNet model for training.
- `generate_dataset.py`: Captures images while manually controlling the TurtleBot.
- `image_classification.py`: Performs inference and outputs predictions.

**Model Results:** Located under `/results/`:
- `res_model_100`: Trained with the standard dataset.
- `res_model_new`: Trained with the YCBV new dataset.
- `res_model_new2`: Trained with the original dataset.

**Output Topic:** `/object_classification_result`

---
### Navigation with Reinforcement Learning (Deep Q-Learning)
**Files Location:** `my_turtlebot3_openai_example` package.

**Scripts:**
- `inference.py`: Runs the trained model in the TurtleBot environment.
- `start_deepqlearning.py`: Trains the model.
- `start_deepqlearning.py` (with visualization): Trains the model with visual feedback.

**Trained Models:** Stored under `training_results/`.

**Run Training:**
```bash
roslaunch my_turtlebot3_openai_example start_training.launch
```

---
### Inference Commands
```bash
roslaunch my_turtlebot3_openai_example inference.launch
roslaunch image_classification classify_image.launch
roslaunch lstm_trajectory_prediction predict_trajectory.launch
```

### Listening to Results
```bash
rostopic echo /predicted_pose
rostopic echo /object_classification_result
```

---
### ⚙️ Setup Instructions
1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   ```
2. **Switch to your branch (if working on it):**
   ```bash
   git checkout -b <your-branch-name>
   ```
3. **Adjust File Paths:**
   Update paths in the following files:
   - `src/turtlebot3_simulations/turtlebot3_gazebo/worlds/turtlebot3_world_objects_light.world` (multiple occurrences)
   - `src/my_turtlebot3_openai_example/config/my_turtlebot3_openai_qlearn_params_v2.yaml`
   - `src/my_turtlebot3_openai_example/config/my_turtlebot3_openai_deepqlearn_params.yaml`
   - ...

   **Note:** Ensure all root paths are correctly adjusted using the search function.
4. Install all dependencies
   ```bash
   pip install -r requirements.txt
   ```
   
5. **Build and Source the workspace**
   ```bash
   catkin build
   source devel/setup.bash
   ```
6. **Execute the roslaunch files**
   Described above in **Inference commands** and **Listening to topics**
   Note, that the navigation launchfile should be started first, as it builds the turtlebotenvironment




