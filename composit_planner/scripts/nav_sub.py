#!/usr/bin/env python

#code heavily modified from simple tutorials by Fiorella Sibona (https://github.com/FiorellaSibona) which use actionlib with python interfaces

import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import GoalStatus
from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Pose, Point, Quaternion
from tf.transformations import quaternion_from_euler

class MoveBaseSeq():
	def __init__(self):
		#initialize node and callback signal
		rospy.init_node('nav_sub')
		self.ready_for_next = True

		#want to wait for all other nodes to be established before connecting to server
		rospy.sleep(20.)
		self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
		rospy.loginfo("Waiting for move_base server...")
		wait = self.client.wait_for_server(rospy.Duration(30.0))
		if not wait:
			rospy.logerr("Action server not initialized")
			rospy.signal_shutdown("Action server not initialized. Aborting mission.")
			return
		rospy.loginfo("Connected to server")
		rospy.loginfo("Starting goal navigation")

		#subscribe to the navigation points of interest
		self.sub = rospy.Subscriber("/possible_points", PointCloud, self.cloud_cb, queue_size=10)
		
		#spin until shutdown
		while not rospy.is_shutdown():
			rospy.spin()

	def cloud_cb(self, cloud):
		''' Callback function for target points published. Selects the highest utility point to navigate to.
		Input:
			cloud: PointCloud object with (x,y) points and channel utility values
		Output:
			accesses the navigation client to publish target
		'''
		#if the last target has been reached, go ahead and update
		if self.ready_for_next:
			#parse the callback data
			frame_id = cloud.header.frame_id
			points = cloud.points
			vals = cloud.channels

			#identify the highest valued point
			current_max = -1000
			target = None
			for p,v in zip(points,vals):
				if v > current_max:
					current_max = v
					target = p
			t = [target.x, target.y, 0]

			#turn point into navigation goal
			p_select = Pose(Point(*(t)), Quaternion(*(quaternion_from_euler(0,0,90*3.14/180, axes='sxyz'))))
			goal = MoveBaseGoal()
			goal.target_pose.header.frame_id = frame_id
			goal.target_pose.header.stamp = rospy.Time.now()
			goal.target_pose.pose = p_select
			rospy.loginfo("Sending new pose to server")
			self.client.send_goal(goal, self.done_cb, self.active_cb, self.feedback_cb)
			self.ready_for_next = False
		else:
			pass

	def active_cb(self):
		''' Native callback for the navigation client. Indicates when goal point is received.'''
		rospy.loginfo("Goal pose is now bring processed")

	def feedback_cb(self, feedback):
		''' Native callback for the navigation client. Indicates activity for goal point.'''
		rospy.loginfo("Feedback for goal pose received")

	def done_cb(self,status,result):
		''' Native callback for the navigation client. Indicates the state of the server. Our default behavior is to prepare to receive a new target if something goes wrong or the last target was successful'''

		if status == 2:
			rospy.loginfo("Goal pose canceled")
			self.ready_for_next = True

		if status == 3:
			rospy.loginfo("Goal pose reached")
			self.ready_for_next = True

		if status == 4:
			rospy.loginfo("Goal pose aborted")
			self.ready_for_next = True

		if status == 5:
			rospy.loginfo("Goal pose rejected")
			self.ready_for_next = True

		if status == 8:
			rospy.loginfo("Goal pose canceled")
			self.ready_for_next = True


if __name__ == '__main__':
	try:
		MoveBaseSeq()
	except rospy.ROSInterruptException:
		rospy.loginfo("Navigation test finished.")