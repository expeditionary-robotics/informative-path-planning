#!/usr/bin/python

'''
Copyright 2018 Massachusetts Institute of Technology
'''

import numpy as np
import rospy
import roslib
import actionlib
from std_msgs.msg import *
from composit_planner.srv import *
from composit_planner.msg import *
from geometry_msgs.msg import *
from trajectory_msgs.msg import *
from nav_msgs.msg import *
from sensor_msgs.msg import *

'''
This is a node which prevents crashes into nearby obstacles that reveal themselves
'''

class Hindbrain(object):
    ''' Node which monitors current trajectory and costmap in order to avoid collisions.
    '''
    def __init__(self):
        ''' Initialize the class.
        '''

        # pull information from the launch config file
        self.safe_threshold = rospy.get_param('cost_limit', 50.) # costmap cutoff for collision
        self.allow_backup = rospy.get_param('allow_to_backup', True) # call backup controller

        # initialize variables that will be updated through subscribers
        self.path = None
        self.pose = None

        # subscribe to costmap and trajectory
        self.traj_sub = rospy.Subscriber('/trajectory/current', PolygonStamped, self.handle_trajectory)
        self.cost_srv = rospy.ServiceProxy('obstacle_map', GetCostMap)
        self.pose_sub = rospy.Subscriber("/pose", PoseStamped, self.handle_pose)

        # publish to generate new plan
        self.replan = actionlib.SimpleActionClient('replan', ReplanRequestAction)
        self.replan.wait_for_server()

        #create polygon object to kill current trajectory, then backup
        self.path_pub = rospy.Publisher('/trajectory/current', PolygonStamped,
                                        queue_size=1)

        #run the node at a certain rate to check things
        r = rospy.Rate(30)
        while not rospy.is_shutdown():
            # Pubish current belief map
            self.check_trajectory()
            r.sleep()

    def handle_trajectory(self, msg):
        coordinates = []
        for coord in msg.polygon.points:
            if coord.x != -100. or coord.x != -1.: #these are the failure cases
                coordinates.append([coord.x,coord.y])
        self.path = coordinates

    def handle_pose(self, msg):
        ''' Update the current pose of the robot.
        Input: msg (geometry_msgs/PoseStamped)
        Output: None ''' 
        #print 'Updating Pose'
        odom_pose = msg.pose # of type odometry messages
        trans_pose = Point32()
        trans_pose.x = odom_pose.position.x
        trans_pose.y = odom_pose.position.y
        q = odom_pose.orientation
        trans_pose.z = euler_from_quaternion((q.x, q.y, q.z, q.w))[2]
        self.pose = trans_pose


    def check_trajectory(self):
        map_resp = self.cost_srv(GetCostMapRequest())
        current_map = map_resp.map
        data = self.make_array(current_map.data, current_map.info.height, current_map.info.width)

        if len(self.path) > 0 and self.path is not None:
            idx = [int(round((x[0]-current_map.info.origin.position.x)/current_map.info.resolution)) for x in self.path]
            idy = [int(round((x[1]-current_map.info.origin.position.y)/current_map.info.resolution)) for x in self.path]
            try:
                # print data[idy,idx]
                cost_vals = [k for k in data[idy,idx] if k>=0.]
                cost = np.sum(cost_vals)
            except:
                cost = 0
                for m, n in zip(idx, idy):
                    try:
                        if data[n,m] >= 0:
                            cost += data[n,m]
                    except:
                        break
            if cost > self.safe_threshold:
                self.path = []
                # cancel the current path trajectory in the controller
                abort_mission = PolygonStamped()
                abort_mission.header.frame_id = 'world'
                abort_mission.header.stamp = rospy.Time(0)
                abort_mission.polygon.points = [Point32(-100.,-100.,-100.)]
                self.path_pub.publish(abort_mission)

                # send a back-up signal to the controller if allowed
                if self.allow_backup is True:
                    abort_mission = PolygonStamped()
                    abort_mission.header.frame_id = 'world'
                    abort_mission.header.stamp = rospy.Time(0)
                    abort_mission.polygon.points = [Point32(-1,0,0)]
                    self.path_pub.publish(abort_mission)
                    rospy.sleep(5)

                self.replan.send_goal(self.pose)
                self.replan.wait_for_result(rospy.Duration.from_sec(5.0))
            

    def make_array(self,data,height,width):
        return np.array(data).reshape((height,width),order='C')


def main():
	#initialize node
	rospy.init_node('hindbrain')
	try:
		Hindbrain()
	except rospy.ROSInterruptException:
		pass


if __name__ == '__main__':
	main()
