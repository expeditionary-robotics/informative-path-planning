#include "grid_map_core/GridMap.hpp"
#include "grid_map_ros/grid_map_ros.hpp"
#include <grid_map_msgs/GridMap.h>
#include "grid_map_ipp/ObstacleGridConverter.hpp"
#include "grid_map_ipp/grid_map_ipp.hpp"
#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/Odometry.h>
#include <iostream>
#include <Eigen/Dense>
#include <list>
#include <vector>
#include <ros/ros.h>
#include <typeinfo>

using namespace std; 
using namespace grid_map;

int main(int argc, char** argv)
{
    ros::init(argc, argv, "test_gt_map");
    ros::NodeHandle nh("");
    ros::Rate rate(30.0);

    ///Generate Ground-truth map environment. 
    int num_box = 5;
    double dim_x = 5.0; double dim_y = 5.0;
    std::list<std::pair<double,double> > center;
    pair<double, double> center1 = make_pair(10.0, 20.0);
    pair<double, double> center2 = make_pair(50.0, 30.0);
    pair<double, double> center3 = make_pair(60.0, 80.0);
    pair<double, double> center4 = make_pair(30.0, 60.0);
    pair<double, double> center5 = make_pair(70.0, 90.0);
    
    center.push_back(center1);
    center.push_back(center2);
    center.push_back(center3);
    center.push_back(center4);
    center.push_back(center5);
    list<pair<double, double>>::iterator iter = center.begin();
    vector<Eigen::Array4d> obstacles;
    
    for(iter=center.begin(); iter!=center.end(); iter++)
    {
        Eigen::Array4d point((*iter).first - dim_x / 2.0, (*iter).second - dim_y / 2.0, (*iter).first + dim_x / 2.0, (*iter).second + dim_y / 2.0);
        obstacles.push_back(point);
    }

    grid_map::ObstacleGridConverter converter(100.0, 100.0, 5, obstacles);
    grid_map::GridMap gt_map = converter.GridMapConverter();
    nav_msgs::OccupancyGrid occ_grid = converter.OccupancyGridConverter();



    // cur_pose.x = 10.0; cur_pose.y = 25.0; cur_pose.yaw = 0.0;
    RayTracer::Pose cur_pose(10.0, 25.0, 0.0); 
    nav_msgs::Odometry odom;
    odom.header.frame_id = "map";
    odom.pose.pose.orientation.w = 1.0;
    odom.pose.pose.orientation.x = 0.0;
    odom.pose.pose.orientation.y = 0.0;
    odom.pose.pose.orientation.z = 0.0;
    odom.pose.pose.position.x = cur_pose.x;
    odom.pose.pose.position.y = cur_pose.y;
    odom.pose.pose.position.z = 0.0;
    
    ros::Publisher pub_odom = nh.advertise<nav_msgs::Odometry>("odom", 1, true);
    pub_odom.publish(odom);
    
    double range_max = 5.0; double range_min = 0.5; 
    double hangle_max = 180; double hangle_min = -180; double angle_resol = 5.0;
    double resol = 1.0;

    // RayTracer::Raytracer raytracer(gt_map);

    // RayTracer::Lidar_sensor lidar(range_max, range_min, hangle_max, hangle_min, angle_resol, 100.0, 100.0, resol, raytracer);
    // nav_msgs::OccupancyGrid belief_map;
    // grid_map::GridMap belief_grid = lidar.get_belief_map();
    // GridMapRosConverter::toOccupancyGrid(belief_grid, "base", 0.0, 1.0, belief_map);

    // ros::Publisher pub_belief_map = nh.advertise<nav_msgs::OccupancyGrid>("belief_map", 1, true);
    // ros::Publisher pub_gt_map = nh.advertise<nav_msgs::OccupancyGrid>("gt_map", 1, true);
    // ros::Publisher pub_grid_map = nh.advertise<grid_map_msgs::GridMap>("grid_map", 1, true);
    
    // grid_map_msgs::GridMap message; 
    // GridMapRosConverter::toMessage(gt_map, message);
    // while(nh.ok())
    // {
    //     pub_grid_map.publish(message);
    //     pub_belief_map.publish(belief_map);
    //     pub_gt_map.publish(occ_grid);
    //     grid_map::GridMap belief_grid = lidar.get_belief_map();
    //     GridMapRosConverter::toOccupancyGrid(belief_grid, "base", 0.0, 1.0, belief_map);
    //     lidar.get_measurement(cur_pose);

    //     rate.sleep();
    // }

    return 0;
}