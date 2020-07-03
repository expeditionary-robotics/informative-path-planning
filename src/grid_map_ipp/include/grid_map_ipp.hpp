#include "grid_map_core/GridMap.hpp"
#include "grid_map_core/iterators/iterators.hpp"
#include <Eigen/Dense>
#include <vector>
#include <iostream>

using namespace std;

namespace RayTracer{
    
    struct Pose{
        double x; double y; double yaw;
    };

    class Lidar_sensor{

        private:
            double range_max_;
            double range_min_;
            double hangle_max_;
            double hangle_min_;
            double angle_resol_;
            double resol_;
            grid_map::GridMap belief_map_;

        public:
            Lidar_sensor(double range_max, double range_min, double hangle_max, double hangle_min, double angle_resol, double resol)
             : range_max_(range_max), range_min_(range_min), hangle_max_(hangle_max), hangle_min_(hangle_min), angle_resol_(angle_resol), resol_(resol)
             {
                 belief_map_ = init_belief_map();
             }
            
            grid_map::GridMap init_belief_map();
            
            void get_measurement(Pose& cur_pos);//Lidar measurement from current pose. 
            void gen_single_ray(grid_map::Position& start_pos, grid_map::Position& end_pos); //Single raycasting
            void update_map(vector<grid_map::Index> index_vec); //
    };

    class RayTracer{
        private:
            
            grid_map::GridMap gt_map_;
            grid_map::LineIterator raytracer_;

        public:
            void set_gt_map(grid_map::Matrix &data);
            void set_raytracer();
            void raytracing(Lidar_sensor sensor, grid_map::Index startIndex, grid_map::Index endIndex);
            grid_map::Index get_final();
    };
}