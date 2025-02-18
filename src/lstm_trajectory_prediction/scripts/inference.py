#!/usr/bin/env python3


import rospy
import torch
from model import LSTMModel
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Pose
from nav_msgs.msg import Odometry
import numpy as np

class GazeboInference:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('gazebo_inference_node', anonymous=True)

        # Load the trained model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = LSTMModel(input_size=10, hidden_size=128, output_size=7, num_layers=2, dropout_rate=0.2)
        self.model.load_state_dict(torch.load('/home/danielbugelnig/mobile_robot_navigation/src/lstm_trajectory_prediction/models/model_seq30.pth', map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # Buffer for LSTM input
        self.data_buffer = []
        self.sequence_length = 10

        # Initial position and orientation
        self.initial_position = None
        self.initial_orientation = None
        self.got_initial_odom = False

        # Subscribe to the /imu topic
        self.imu_sub = rospy.Subscriber('/imu', Imu, self.imu_callback)

        #Subscrib to the /odom topic to get the initial psoition
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)

        # Publisher for predicted pose
        self.pose_pub = rospy.Publisher('/predicted_pose', Pose, queue_size=10)

    def odom_callback(self, data):
        """Get initial position from /odom only once."""
        if not self.got_initial_odom:
            position = data.pose.pose.position
            orientation = data.pose.pose.orientation
            self.initial_position = np.array([position.x, position.y, position.z])
            self.initial_orientation = np.array([orientation.x, orientation.y, orientation.z, orientation.w])
            self.got_initial_odom = True

            rospy.loginfo(f"Initial position from /odom: {self.initial_position}")
            rospy.loginfo(f"Initial orientation from /odom: {self.initial_orientation}")

            # Unsubscribe from /odom after capturing initial position
            self.odom_sub.unregister()
            rospy.loginfo("Unsubscribed from /odom after getting intila position")
    

    def imu_callback(self, data):
        # extract IMU data for model input
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
        # Prepare input tensor
        input_tensor = torch.tensor(self.data_buffer, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            predictions = self.model(input_tensor) 

        # Process and publish the result
        self.publish_pose(predictions)

    def publish_pose(self, predictions):
        # Extract pose predictions
        #position = predictions[0, :3].cpu().numpy() 
        position = predictions[0, :3].cpu().numpy() + self.initial_position
        
        #orientation = predictions[0, 3:].cpu().numpy() 
        orientation = predictions[0, 3:].cpu().numpy() + self.initial_orientation

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
    try:
        inference_node = GazeboInference()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
