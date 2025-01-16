#!/usr/bin/env python
import rospy
import rosbag
from geometry_msgs.msg import PoseStamped
from actionlib_msgs.msg import *
from gazebo_msgs.msg import ModelStates
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool
# import actionlib
from move_base_msgs.msg import MoveBaseActionGoal
from multiprocessing import Process

import argparse


class WaypointHandlerNode:
    def __get_goal_msg__(self):
        # copy next waypoint to msg
        msg = MoveBaseActionGoal()
        pose = PoseStamped()
        pose.pose.position.x = self.wpts[self.current_wpt][0]
        pose.pose.position.y = self.wpts[self.current_wpt][1]
        pose.pose.position.z = 0.
        pose.pose.orientation.x = 0.
        pose.pose.orientation.y = 0.
        pose.pose.orientation.z = 0.
        pose.pose.orientation.w = 1.

        # set the goal id
        msg.goal_id.id = str(self.current_wpt)

        # initialize header with current time
        msg.goal.target_pose.header.stamp = rospy.Time.now()
        msg.goal.target_pose.header.frame_id = 'map'
        msg.goal.target_pose.pose = pose.pose
        return msg

    def wpt_check_callback(self, data):

        if len(data.status_list) != 0:
            # if waypoint has been reached, then publish next waypoint unless we're finished
            if ((data.status_list[0].status == 3) & (data.status_list[0].goal_id.id == str(self.current_wpt))):
                # increment wpt counter
                self.current_wpt += 1
                if self.current_wpt < len(self.wpts):
                    msg = self.__get_goal_msg__()

                    # publish message
                    self.goal_pub.publish(msg)
                    rospy.loginfo(msg)

    def __init__(self):
        
        #[-2,0],[0,-2],[-1.6, -1.1],[-2 ,0],[-1.8, -1.2], [0.77, -0.67], [0.57, 0.76], [-1.57, 1.37],[-1.8, -1.2], [0.77, -0.67], [0.57, 0.76], [-1.57, 1.37]
        self.wpts = [[-1.8, -1.2]]
        #this are my waypoints that the robot must go
        #trajectory 1 = [[-1.8, -1.2], [-0.55, -0.67]]
        #trajectory 2 = [[-1.8, -1.2], [-0.1, -0.1]]
        #self.wpts = [[-1.8, -1.2], [0.77, -0.67], [0.57, 0.76], [-1.57, 1.37]]
        self.current_wpt = 0

        # define our wpt publisher and done publisher
        self.goal_pub = rospy.Publisher("/move_base/goal", MoveBaseActionGoal, queue_size=1)
        self.done_pub = rospy.Publisher("/turtlebot3_waypoints/done", Bool, queue_size=1)

        # sleep so that publishers have time to be registered
        rospy.sleep(3)

        # now send the first waypoint before we go into the loop
        msg = self.__get_goal_msg__()

        # publish message
        self.goal_pub.publish(msg)
        rospy.loginfo(msg)
        while not rospy.is_shutdown():
            # subscribe to base status to check if target is reached
            rospy.Subscriber("/move_base/status", GoalStatusArray, self.wpt_check_callback, queue_size=1)

            if (self.current_wpt == len(self.wpts)):
                # we are done and send the done signal
                msg = Bool()
                msg.data = True
                self.done_pub.publish(msg)
                print('Trajectory finished. Exiting...')
                break


# this is the recorder class
class RecorderNode:
    def record_topic(self, data, topic):
        # save the message to the bagfile
        #print(data)
        #print(topic[0])
        self.bag.write(topic[0], data)
        pass

    def done_callback(self, data):
        if data.data == True:
            rospy.signal_shutdown('Trajectory finished - stop recording')
            self.bag.close()
            print('Trajectory finished - stopped recording')

    def __init__(self, args):
        self.bag = rosbag.Bag(args.bagfile[0], 'w')
        rospy.Subscriber("/move_base/status", GoalStatusArray, self.record_topic, ("/move_base/status",))
        rospy.Subscriber("/gazebo/model_states", ModelStates, self.record_topic, ("/gazebo/model_states",))
        rospy.Subscriber("/imu", Imu, self.record_topic, ("/imu",))
        rospy.Subscriber("/turtlebot3_waypoints/done", Bool, self.done_callback)

        # alternative is to wrap it in a while loop as above
        rospy.spin()


# this launches the waypoint handler
def start_waypoint_handler():
    rospy.init_node('turtlebot3_waypoints')
    try:
        WPTHandler = WaypointHandlerNode()
    except rospy.ROSInterruptException:
        pass


# this launches the recording of topics of interest
def start_rosbag_recorder(args):
    rospy.init_node('turtlebot3_recorder')
    try:
        Recorder = RecorderNode(args)
    except rospy.ROSInterruptException:
        pass


# Main function.
if __name__ == '__main__':
    #parser = argparse.ArgumentParser(description='Send waypoints to turtlebot and record rosbag')
    #parser.add_argument('bagfile', metavar='bagfile', type=str, nargs=1, help='full filepath for output bagfile')

    #args = parser.parse_args()

    p_waypoint_handler = Process(target=start_waypoint_handler)
    p_waypoint_handler.start()

    #p_recorder = Process(target=start_rosbag_recorder, args=(args, ))
    #p_recorder.start()


