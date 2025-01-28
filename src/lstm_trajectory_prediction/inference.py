import rospy
import torch
from model import LSTMModel
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Pose
import numpy as np

class GazeboInference:
    def __init__(self):
        """
        Initialize the GazeboInference class.

        - Sets up the ROS node.
        - Loads a trained LSTM model.
        - Initializes a data buffer for sequence input.
        - Subscribes to IMU data and sets up a publisher for predicted pose.
        """
        # Initialize the ROS node
        rospy.init_node('gazebo_inference_node', anonymous=True)

        # Load the trained model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = LSTMModel(input_size=10, hidden_size=128, output_size=7, num_layers=2, dropout_rate=0.2)
        self.model.load_state_dict(torch.load('model_seq50_100epochs.pth', map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # Buffer for LSTM input
        self.sequence_length = 50
        self.data_buffer = [[0.0] * 10] * self.sequence_length  # Pre-fill buffer with zeros to avoid delays

        # Subscribe to the /imu topic
        self.imu_sub = rospy.Subscriber('/imu', Imu, self.imu_callback)

        # Publisher for predicted pose
        self.pose_pub = rospy.Publisher('/predicted_pose', Pose, queue_size=10)

    def imu_callback(self, data):
        """
        Callback function for processing IMU messages.

        Args:
            data (sensor_msgs.msg.Imu): The incoming IMU message containing accelerations, angular velocities, and orientations.
        """
        # Extract IMU data for model input
        imu_data = [
            data.linear_acceleration.x,
            data.linear_acceleration.y,
            data.linear_acceleration.z,
            data.angular_velocity.x,
            data.angular_velocity.y,
            data.angular_velocity.z,
            data.orientation.x,
            data.orientation.y,
            data.orientation.z,
            data.orientation.w
        ]

        # Update the data buffer
        self.data_buffer.append(imu_data)
        if len(self.data_buffer) > self.sequence_length:
            self.data_buffer.pop(0)

        # Run the model if buffer is full
        if len(self.data_buffer) == self.sequence_length:
            self.run_model()

    def run_model(self):
        """
        Perform inference using the buffered IMU data and publish the predicted pose.
        """
        # Prepare input tensor
        input_tensor = torch.tensor(self.data_buffer, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            predictions = self.model(input_tensor)

        # Process and publish the result
        self.publish_pose(predictions)

    def publish_pose(self, predictions):
        """
        Publish the predicted pose as a ROS Pose message.

        Args:
            predictions (torch.Tensor): The model's output containing position and orientation predictions.
        """
        # Extract pose predictions
        position = predictions[0, :3].cpu().numpy()
        orientation = predictions[0, 3:].cpu().numpy()

        # Create Pose message
        pose_msg = Pose()
        pose_msg.position.x = position[0]
        pose_msg.position.y = position[1]
        pose_msg.position.z = position[2]
        pose_msg.orientation.x = orientation[0]
        pose_msg.orientation.y = orientation[1]
        pose_msg.orientation.z = orientation[2]
        pose_msg.orientation.w = orientation[3]

        # Publish the Pose message
        self.pose_pub.publish(pose_msg)

if __name__ == '__main__':
    """
    Main script to initialize the GazeboInference node and start processing IMU data.
    """
    try:
        inference_node = GazeboInference()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
