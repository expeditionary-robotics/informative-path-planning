#!/usr/bin/env python

'''
Node to perform a lidar transform hack to prevent 3D and 2D conversion headaches.

Copyright 2018 Massachusetts Institute of Technology
'''

import rospy
import tf
from tf.msg import *


class tfChanger(object):
    '''
    Class to change the lidar transform.
    '''

    def __init__(self):
        rospy.init_node('pose_tf_listener')

        self.br = tf.TransformBroadcaster()
        rospy.Subscriber('/tf', tfMessage, self.pose_changer)
        rospy.spin()

    def pose_changer(self, msg):
        '''
        When the transform of interest is published, update the desired transform.
        '''
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
    # initialize node
    rospy.init_node('pose_tf_listener')
    try:
        tfChanger()
    except rospy.ROSInterruptException:
        pass
