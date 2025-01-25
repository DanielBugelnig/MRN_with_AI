#!/usr/bin/env python
#to execute ROS files, deactivate venvs and use python 3.8 , (restart terminal if necessary)

# Generate a script to integrate IMU data recorded from Turtlebot sim in Gazebo using the classical equations of motion.

import rospy
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64
from geometry_msgs.msg import PoseStamped

import numpy as np
import time

# Data structure for the IMU message
velocity = np.array([0.0,0.0,0.0])
position = np.array([0.0,0.0,0.0])
orientation = np.array([0.0, 0.0, 0.0])
last_time = None

#Create Publishers for processed data
estimated_pose_pub = rospy.Publisher('/imu_listener/estimated_pose', PoseStamped, queue_size=10) # buffer

def quaternion_from_angle(angle_rad):
    # Calculate the components of the quaternion
    w = np.cos(angle_rad / 2)
    x = 0
    y = 0
    z = np.sin(angle_rad / 2)
    return (w, x, y, z)

def imu_callback(data):
    global velocity, position, orientation, last_time
    #rospy.loginfo(f"Linear Acc: {data.linear_acceleration}") # Log linear acceleration and angular velocity
    #rospy.loginfo(f"Angular Vel: {data.angular_velocity}")
    # Linear Acc in x,y,z
    # Angular Vel in x,y,z

    current_time = rospy.Time.now() # Get the current time
    lin_acc = np.array([data.linear_acceleration.x, data.linear_acceleration.y, data.linear_acceleration.z]) # Extract linear acceleration
    ang_vel = np.array([data.angular_velocity.x, data.angular_velocity.y, data.angular_velocity.z]) # Extract angular velocity

    if last_time is not None:

        # Calculation of the orientation, velocity and position
        dt = (current_time - last_time).to_sec()
        orientation += ang_vel * dt
        orientation = orientation%(2*np.pi)  # Normalize the orientation to the range [0, 2*pi)

        # Coordinate Transform
        x_rot = (lin_acc[0]*np.cos(-orientation[2]) - lin_acc[1]*np.sin(-orientation[2]))
        y_rot = (lin_acc[0]*np.sin(-orientation[2]) + lin_acc[1]*np.cos(-orientation[2]))

        velocity += np.array([x_rot, y_rot, 0]) * dt
        position += velocity * dt

        
        # Create and publish the pose
        pose_msg = PoseStamped()
        pose_msg.pose.position.x = position[0]
        pose_msg.pose.position.y = position[1]
        pose_msg.pose.position.z = 0

        # Convert the orientation to a quaternion
        quat = quaternion_from_angle(orientation[2])
        pose_msg.pose.orientation.w = quat[0]
        pose_msg.pose.orientation.x = 0
        pose_msg.pose.orientation.y = 0
        pose_msg.pose.orientation.z = quat[3]

        pose_msg.header.stamp = data.header.stamp  # Same timestamp as incoming message
 

        # Publish the pose message
        estimated_pose_pub.publish(pose_msg)  

    last_time = current_time


def image_classification_callback(data):
    return


def imu_listener():
   
    rospy.init_node('imu_processing')  # Initialize the ROS node
    rospy.Subscriber('/imu', Imu, imu_callback, queue_size=10) # Subscribe to the IMU topic (imu_callback processes each message)
    rospy.Subscrber('/image_raw', image_classification_callback, queue_size=10) # Subscribe to the image topic (image_classification_callback processes each message)
    rospy.spin() # Keep the node running until it is stopped

if __name__ == '__main__':
    imu_listener()
    