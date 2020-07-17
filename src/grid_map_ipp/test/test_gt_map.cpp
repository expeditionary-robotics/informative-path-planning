#include "grid_map_core/GridMap.hpp"
#include "grid_map_ros/grid_map_ros.hpp"
#include <grid_map_msgs/GridMap.h>
#include <grid_map_ipp/ObstacleGridConverter.hpp>
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
    

    grid_map::ObstacleGridConverter converter(100.0, 100.0, 5, obstacles);
    grid_map::GridMap gt_map = converter.GridMapConverter();

    nav_msgs::OccupancyGrid occ_grid = converter.OccupancyGridConverter();


    ros::init(argc, argv, "test_gt_map");
    ros::NodeHandle nh("");
    ros::Publisher pub = nh.advertise<grid_map_msgs::GridMap>("grid_map", 1, true);
    ros::Rate rate(30.0);
    ros::Publisher pub_occ = nh.advertise<nav_msgs::OccupancyGrid>("occu_grid", 1, true);

    grid_map::Size size = gt_map.getSize();
    cout<< size(0,0) << endl;

    while(nh.ok())
    {
        ros::Time time = ros::Time::now();
        // grid_map_msgs::GridMap message;
        nav_msgs::OccupancyGrid occ_message;
        nav_msgs::OccupancyGrid &occ_m = occ_message;
        
        // GridMapRosConverter::toMessage(gt_map, message);
        GridMapRosConverter::toOccupancyGrid(gt_map, "base", 0.0, 1.0, occ_m);
        // pub.publish(message);
        pub_occ.publish(occ_grid);
        rate.sleep();    
    }

    return 0;
}