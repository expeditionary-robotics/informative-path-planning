#!/usr/bin/env python

# Copyright 2018 Massachusetts Institute of Technology

import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import GoalStatus
from nav_msgs.msg import Path
from geometry_msgs.msg import Pose, Point, Quaternion
from tf.transformations import quaternion_from_euler

class ExecuteDubinSeq():
	def __init__(self):
		#initialize node and callback
		rospy.init_node('execute_dubin')
		self.ready_for_next = True

		#make sure all nodes are established
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

		#subscribe to trajectory topic
		self.sub = rospy.Subscriber("/selected_trajectory", Path, self.handle_trajectory, queue_size=1)

		#spin until shutdown
		while not rospy.is_shutdown():
			rospy.spin()

	def handle_trajectory(self, traj):
		'''
		The trajectory comes in as a series of poses. It is assumed that the desired angle has already been determined
		'''
		print 'Executing new Trajectory'
		self.client.cancel_goal()
		self.new_goals = traj.poses
		if len(self.new_goals) != 0:
			goal=MoveBaseGoal()
			goal.target_pose.header.frame_id = traj.header.frame_id
			#goal.target_pose.header.stamp = rospy.Time.now()
			goal.target_pose.header.stamp = rospy.Time(0)
			goal.target_pose.pose = self.new_goals[0].pose
			self.client.send_goal(goal, self.done_cb, self.active_cb, self.feedback_cb)
			self.new_goals.pop(0)
		else:
			print 'No trajectory is viable'

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

		if status == 3:
			rospy.loginfo("Goal pose reached")
			if len(self.new_goals) != 0:
				goal=MoveBaseGoal()
				goal.target_pose.header.frame_id = self.new_goals[0].header.frame_id
				#goal.target_pose.header.stamp = rospy.Time.now()
				goal.target_pose.header.stamp = rospy.Time(0)
				goal.target_pose.pose = self.new_goals[0].pose
				self.client.send_goal(goal, self.done_cb, self.active_cb, self.feedback_cb)
				self.new_goals.pop(0)

		if status == 4:
			rospy.loginfo("Goal pose aborted")

		if status == 5:
			rospy.loginfo("Goal pose rejected")

		if status == 8:
			rospy.loginfo("Goal pose canceled")


if __name__ == '__main__':
	try:
		ExecuteDubinSeq()
	except rospy.ROSInterruptException:
		rospy.loginfo("Navigation test finished.")
