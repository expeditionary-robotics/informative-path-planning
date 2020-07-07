#include "grid_map_core/GridMap.hpp"
#include "grid_map_ros/grid_map_ros.hpp"
#include <grid_map_msgs/GridMap.h>
#include <iostream>
#include <Eigen/Dense>
#include <list>
#include <vector>
#include <ros/ros.h>


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
    vector<string> name;
    name.push_back("base");

    grid_map::GridMap gt_map(name);
    gt_map.setFrameId("map");
    gt_map.setGeometry(Length(100.0, 100.0), 1.00);
    gt_map.add("base", 0.0);

    //Iteration and fill the obstacle region
    for (GridMapIterator it(gt_map); !it.isPastEnd(); ++it)
    {
        Position position;
        gt_map.getPosition(*it, position);
        double x = position.x() + 50.0; 
        double y = position.y() + 50.0;
        bool is_obs = false;
        for (vector<Eigen::Array4d>::iterator iter=obstacles.begin(); iter!=obstacles.end(); iter++)
        {
            Eigen::Array4d size = *iter;
            if( x > size(1) && y >size(2) ){
                if( x<size(3) && y < size(4)){
                    is_obs = true;
                }
            }
        }
        if(is_obs)
        {
            gt_map.at("base", *it) = 1.0; //Obstacle
        }
    }

    ros::init(argc, argv, "test_gt_map");
    ros::NodeHandle nh("");
    ros::Publisher pub = nh.advertise<grid_map_msgs::GridMap>("grid_map", 1, true);
    ros::Rate rate(30.0);

    while(nh.ok())
    {
        ros::Time time = ros::Time::now();
        grid_map_msgs::GridMap message;
        GridMapRosConverter::toMessage(gt_map, message);

        pub.publish(message);
        rate.sleep();    
    }

    return 0;
}