#include <include/grid_map_ipp.hpp>
#include <Eigen/Dense>

namespace RayTracer{
    grid_map::GridMap Lidar_sensor::init_belief_map()
    {
        vector<string> name;
        name.clear();
        name.push_back("base");
        vector<string>& x = name;
        grid_map::GridMap map(x);

        grid_map::Length len;
        len.x = 100.0; len.y = 100.0;
        grid_map::Position zero; //Zero Position of belief grid
        zero.x = 0.0; zero.y = 0.0;
        map.setGeometry(len, resol_, zero);
        map.add("base", 0.5); //Initialize map of prob. value with 0.5 (unknown)
    }

    void Lidar_sensor::gen_single_ray(grid_map::Position& start_pos, grid_map::Position& end_pos) //Single raycasting
    {   
        //TODO: Transform position to index

        int start_idxx = 0; int start_idxy = 1;
        int end_idxx = 0; int end_idxy = 1;
        grid_map::Index startIndex(start_idxx, start_idxy);
        grid_map::Index endIndex(end_idxx, end_idxy);


    }

    void Lidar_sensor::get_measurement(Pose& cur_pos)
    {
        int ray_num = floor( (hangle_max_ - hangle_min_)/angle_resol_ );
        for(int i=0; i< ray_num; i++)
        {

        }     

    }

    void Lidar_sensor::update_map(vector<grid_map::Index> index_vec)
    {
        // belief_map_
    }

    /**
     * @brief RayTracing and return lidar values.  
     * 
     * @param sensor 
     * @param startIndex 
     * @param endIndex 
     */
    void RayTracer::raytracing(Lidar_sensor sensor, grid_map::Index startIndex, grid_map::Index endIndex)
    {
        for (grid_map::GridMapIterator iterator(gt_map_); !iterator.isPastEnd(); ++iterator) {
            cout << "The value at index " << (*iterator).transpose() << " is " << gt_map_.at("layer", *iterator) << endl;
        }

    }
}