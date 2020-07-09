#include "grid_map_core/GridMap.hpp"
#include "grid_map_ros/grid_map_ros.hpp"
#include <grid_map_msgs/GridMap.h>
#include "grid_map_ipp/ObstacleGridConverter.hpp"
#include "grid_map_ipp/grid_map_ipp.hpp"
#include <nav_msgs/OccupancyGrid.h>
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

    RayTracer::Pose cur_pose; 
    cur_pose.x = 0.0; cur_pose.y = 0.0; cur_pose.yaw = 0.0;

    double range_max = 5.0; double range_min = 0.5; 
    double hangle_max = 180; double hangle_min = -180; double angle_resol = 5.0;
    double resol = 1.0;
    RayTracer::Lidar_sensor lidar(range_max, range_min, hangle_max, hangle_min, angle_resol, resol);

    // vector<string> name;
    // name.push_back("base");

    // grid_map::GridMap gt_map(name);
    // gt_map.setFrameId("map");
    // gt_map.setGeometry(Length(100.0, 100.0), 1.00);
    // gt_map.add("base", 0.0);

    // double buffer = 1.0;
    // //Iteration and fill the obstacle region
    // for (GridMapIterator it(gt_map); !it.isPastEnd(); ++it)
    // {
    //     Position position;
    //     gt_map.getPosition(*it, position);
    //     double x = position.x() + 50.0; 
    //     double y = position.y() + 50.0;
    //     bool is_obs = false;
    //     //Check current pos. is inside any obstacles. If yes, it set the grid map value to 1.0
    //     for (vector<Eigen::Array4d>::iterator iter=obstacles.begin(); iter!=obstacles.end(); iter++)
    //     {
    //         Eigen::Array4d size = (*iter);
    //         if( x > size(0,0) - buffer && y >size(1,0) - buffer ){
    //             if( x<size(2,0) + buffer && y < size(3,0) + buffer){
    //                 is_obs = true;
    //             }
    //         }
    //     }
    //     if(is_obs)
    //     {
    //         gt_map.at("base", *it) = 1.0; //Obstacle
    //     }
    // }

    // ros::init(argc, argv, "test_gt_map");
    // ros::NodeHandle nh("");
    // ros::Publisher pub = nh.advertise<grid_map_msgs::GridMap>("grid_map", 1, true);
    // ros::Rate rate(30.0);
    // ros::Publisher pub_occ = nh.advertise<nav_msgs::OccupancyGrid>("occu_grid", 1, true);
    // while(nh.ok())
    // {
    //     ros::Time time = ros::Time::now();
    //     grid_map_msgs::GridMap message;
    //     nav_msgs::OccupancyGrid occ_message;
    //     GridMapRosConverter::toMessage(gt_map, message);
    //     GridMapRosConverter::toOccupancyGrid(gt_map, "base", 0.0, 1.0, occ_message);

    //     pub.publish(message);
    //     pub_occ.publish(occ_message);
    //     rate.sleep();    
    // }

    return 0;
}