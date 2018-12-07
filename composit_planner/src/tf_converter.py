#!/usr/bin/env python  
import rospy
import math
import tf
from tf.msg import * 
import geometry_msgs.msg

class tfChanger(object):
    def __init__(self):
	rospy.init_node('pose_tf_listener')

	self.br = tf.TransformBroadcaster()
	pose_listener = rospy.Subscriber('/tf', tfMessage, self.pose_changer)
	rospy.spin()
	

    def pose_changer(self, msg):
	for transform in msg.transforms:
	    if transform.header.frame_id == 'world' and transform.child_frame_id == 'body':
		transform.transform.rotation.x = 0.0
		transform.transform.rotation.y = 0.0
		transform.child_frame_id = 'body_flat'

                self.br.sendTransform((transform.transform.translation.x,
                                    transform.transform.translation.y,
                                    0.01),
                                    (transform.transform.rotation.x,
                                    transform.transform.rotation.y,
                                    transform.transform.rotation.z,
                                    transform.transform.rotation.w),
                                    # transform.header.stamp,
                                    rospy.Time.now(),
                                    'body_flat',
                                    'world')

if __name__ == '__main__':
    #initialize node
    rospy.init_node('pose_tf_listener')
    try:
	    tfChanger()
    except rospy.ROSInterruptException:
		pass

