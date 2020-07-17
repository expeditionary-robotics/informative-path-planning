#ifndef GRIDMAPIPP
#define GRIDMAPIPP

#include "grid_map_core/GridMap.hpp"
#include "grid_map_core/iterators/iterators.hpp"
#include "grid_map_ipp/ObstacleGridConverter.hpp"
#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <string>

using namespace std;

namespace RayTracer{
    
    struct Pose{
        double x; 
        double y; 
        double yaw;

        Pose(double x1, double y1, double yaw1){ x=x1; y=y1; yaw = yaw1; }

    };
    class Lidar_sensor;
    
    class Raytracer{
        private:            
            grid_map::GridMap gt_map_;
            // grid_map::LineIterator raytracer_;

        public:
            // RayTracer(grid_map::Index &startIndex, grid_map::Index &endIndex)
            // {
            //     // raytracer_(startIndex, endIndex);
            // }
            Raytracer(double map_size_x, double map_size_y, int num_obstacle, std::vector<Eigen::Array4d> obstacles)
            {
                grid_map::ObstacleGridConverter converter(map_size_x, map_size_y, num_obstacle, obstacles);
                gt_map_ = converter.GridMapConverter();
            }
            ~Raytracer() {}
            
            grid_map::GridMap get_grid_map(){ return gt_map_;}
            void set_gt_map(grid_map::Matrix &data);
            void set_raytracer();
            pair<vector<grid_map::Index>, bool> raytracing(Lidar_sensor& sensor, grid_map::Index& startIndex, grid_map::Index& endIndex);
            grid_map::Index get_final();
    };


    //Lidar sensor class is permanent for the robot. It has belief_map which recurrently updated with measurement values. 
    class Lidar_sensor{
        private:
            double range_max_;
            double range_min_;
            double hangle_max_;
            double hangle_min_;
            double angle_resol_;
            double resol_;

            double map_size_x_;
            double map_size_y_;
            grid_map::GridMap belief_map_;
            Raytracer raytracer_;
            string layer_;
            
        public:
            Lidar_sensor(double range_max, double range_min, double hangle_max, double hangle_min, double angle_resol, double map_size_x, double map_size_y, double resol, Raytracer& raytracer)
             : range_max_(range_max), range_min_(range_min), hangle_max_(hangle_max), hangle_min_(hangle_min), angle_resol_(angle_resol), map_size_x_(map_size_x), map_size_y_(map_size_y)
             , resol_(resol), raytracer_(raytracer)
             {
                 belief_map_ = init_belief_map();
             }

            ~Lidar_sensor() {}
            
            grid_map::GridMap init_belief_map()                
            {
                vector<string> name;
                name.clear();
                name.push_back("base");
                vector<string> x = name;
                grid_map::GridMap map(x);

                grid_map::Length len(map_size_x_, map_size_y_);
                grid_map::Position zero(0.0, 0.0); //Zero Position of belief grid
                // zero.x = 0.0; zero.y = 0.0;
                map.setGeometry(len, resol_, zero);
                map.add("base", 0.5); //Initialize map of prob. value with 0.5 (unknown)
                // belief_map_ = map;
                // cout << map.getLayers().at(0) << endl;
                // cout << belief_map_.getLayers().at(0) << endl;

                grid_map::Length size; size = map.getLength();
                cout << "size " << size(0) << " " << size(1) << endl;
                return map;
            }

            void get_measurement(Pose& cur_pos);//Lidar measurement from current pose. 
            pair<vector<grid_map::Index>, bool> gen_single_ray(grid_map::Position& start_pos, grid_map::Position& end_pos); //Single raycasting
            void update_map(vector<grid_map::Index>& free_vec, vector<grid_map::Index>& index_vec); //
            double inverse_sensor(double cur_val, double meas_val);
            
            double get_occ_value(double x, double y)
            {
                grid_map::Position pos(x,y);
                // pos << x, y;
                grid_map::Index idx;
                belief_map_.getIndex(pos, idx);
                return belief_map_.at("base", idx);
            }

            grid_map::GridMap get_belief_map(){
                return belief_map_;
            }
    };

}

#endif