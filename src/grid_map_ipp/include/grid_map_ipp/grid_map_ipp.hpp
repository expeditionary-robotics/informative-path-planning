#include "grid_map_core/GridMap.hpp"
#include "grid_map_core/iterators/iterators.hpp"
#include <Eigen/Dense>
#include <vector>
#include <iostream>

using namespace std;

namespace RayTracer{
    
    struct Pose{
        double x; 
        double y; 
        double yaw;
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

        public:
            Lidar_sensor(double range_max, double range_min, double hangle_max, double hangle_min, double angle_resol, double map_size_x, double map_size_y, double resol)
             : range_max_(range_max), range_min_(range_min), hangle_max_(hangle_max), hangle_min_(hangle_min), angle_resol_(angle_resol), map_size_x_(map_size_x), map_size_y_(map_size_y), resol_(resol)
             {
                 belief_map_ = init_belief_map();
             }
            
            grid_map::GridMap init_belief_map();
            
            void get_measurement(Pose& cur_pos);//Lidar measurement from current pose. 
            pair<vector<grid_map::Index>, grid_map::Index> gen_single_ray(grid_map::Position& start_pos, grid_map::Position& end_pos); //Single raycasting
            void update_map(vector<grid_map::Index>& free_vec, vector<grid_map::Index>& index_vec); //
            double inverse_sensor(double cur_val, double meas_val);
    };

    class RayTracer{
        private:            
            grid_map::GridMap& gt_map_;
            // grid_map::LineIterator raytracer_;

        public:
            // RayTracer(grid_map::Index &startIndex, grid_map::Index &endIndex)
            // {
            //     // raytracer_(startIndex, endIndex);
            // }
            void set_gt_map(grid_map::Matrix &data);
            void set_raytracer();
            pair<vector<grid_map::Index>, bool> raytracing(Lidar_sensor& sensor, grid_map::Index& startIndex, grid_map::Index& endIndex);
            grid_map::Index get_final();
    };
}