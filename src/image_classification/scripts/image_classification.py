#!/usr/bin/env python3
import torch
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String


import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
import copy
from cv_bridge import CvBridge
import cv2

# Initialize CvBridge
bridge = CvBridge()

transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize(mean= [0.485, 0.456, 0.406], std=[0.229,0.224,0.225])
])

labels_map = {
    0: "Dominos",
    1: "Cheezit",
    2: "Frenchis"
}
publisher = None

def callback_function(image):
    """
    Image topic callback to run a neural network to classify objects
    """
    #rospy.loginfo(f"Received message: {image.height} x {image.width}")
    #rospy.loginfo(f"{type(image)}")
    try:
        # Convert the ROS Image message to a format compatible with OpenCV
        cv_image = bridge.imgmsg_to_cv2(image, desired_encoding='bgr8')
        # Process the image as needed
        resized_image = cv2.resize(cv_image, (224, 224), interpolation=cv2.INTER_AREA)
    except Exception as e:
        rospy.logerr(f"Error converting image: {e}")
    #rospy.loginfo(type(resized_image))

    resized_image = resized_image / 255.0  # Normalize to [0, 1]
    tensor_image = transform(resized_image)
    tensor_image = tensor_image.unsqueeze(0)  # Shape: [1, 3, 224, 224]
    tensor_image = tensor_image.float()
    #print(f"Tensor dtype: {tensor_image.dtype}")
    #print(f"Model dtype: {next(model.parameters()).dtype}")

    output = model(tensor_image)
    output = torch.softmax(output, dim=1) # Convert to probabilities
    max_value, predicted_label = torch.max(output, dim=1)
    print(output)
    result = ""
    if (max_value > threshold):
        result = f"Object detected: {labels_map[predicted_label.item()]}"
    else:
        result = "No object detected"
    rospy.loginfo(result)
    publisher.publish(result)


def subscriber_node():
    # Initialize the node
    global publisher
    rospy.init_node('object_classification_node')

    publisher = rospy.Publisher('/object_classification_result', String, queue_size=10)

    # Subscribe to the topic
    rospy.Subscriber('/turtlebot3_burger/camera1/image_raw', Image, callback_function, queue_size=10)
    
    # Keep the node running
    rospy.spin()

if __name__ == '__main__':
    # request param from param file
    threshold = rospy.get_param('image_classification/threshold', 0.5)  # Default to 0.5 if not set
    model_path = rospy.get_param('image_classification/model', '/home/danielbugelnig/mobile_robot_navigation/src/image_classification/results/res_model_new2.pth')

    print(f"Threshold: {threshold}")
    print(f"Model path: {model_path}")
    # selecting device and loading model
    device = torch.device("cpu")
    model = torch.load(model_path, weights_only=False)
    model = model.to(device)
    model.eval()

# subscribe to image node, then in callback classify image
    try:
        subscriber_node()
    except rospy.ROSInterruptException:
        rospy.loginfo("Node terminated.")
