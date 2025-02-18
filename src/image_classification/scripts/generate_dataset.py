#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os

# Initialize CvBridge
bridge = CvBridge()

# Variables for image saving
image_counter = 0
base_filename = "Frenchis"
save_directory = "/home/danielbugelnig/mobile_robot_navigation/src/image_classification/src/ycbv_classification_new/Frenchis/"

# Function to handle keyboard input
def check_keypress():
    key = cv2.waitKey(1) & 0xFF
    return key == 13  # Return True if the Enter key is pressed

# Callback function for image processing
def image_callback(image_msg):
    global image_counter
    #   print("Image received")

    try:
        # Convert ROS Image message to OpenCV image
        cv_image = bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        
        # Display the image in a window
        cv2.imshow("Camera Feed", cv_image)

        input("Enter for saving")
        print("Enter key pressed")
        # Generate the image filename
        image_filename = f"{base_filename}_{image_counter:04d}.png"
        image_path = os.path.join(save_directory, image_filename)

        # Save the image
        cv2.imwrite(image_path, cv_image)
        rospy.loginfo(f"Image saved: {image_path}")

        # Increment the counter
        image_counter += 1
    except Exception as e:
        rospy.logerr(f"Error processing image: {e}")

# Main function to set up the subscriber
def main():
    global base_filename, save_directory

    # Initialize the ROS node
    rospy.init_node("image_saver", anonymous=True)

    # Get parameters from the user
    #base_filename = input("Enter the base filename: ") or "image"
    #save_directory = input("Enter the directory to save images (default: ./images): ") or "./images"

    # Ensure the save directory exists
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Subscribe to the image topic
    rospy.Subscriber('/turtlebot3_burger/camera1/image_raw', Image, image_callback, queue_size=10)

    rospy.loginfo("Image saver node started. Press Enter in the OpenCV window to save an image.")

    # Keep the program alive
    rospy.spin()

    # Clean up OpenCV windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        rospy.loginfo("Image saver node terminated.")
