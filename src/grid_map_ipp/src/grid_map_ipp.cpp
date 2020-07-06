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

        grid_map::Length len(100.0, 100.0);
        // len.x = 100.0; len.y = 100.0;
        grid_map::Position zero(0.0, 0.0); //Zero Position of belief grid
        // zero.x = 0.0; zero.y = 0.0;
        map.setGeometry(len, resol_, zero);
        map.add("base", 0.5); //Initialize map of prob. value with 0.5 (unknown)
    }


    void Lidar_sensor::get_measurement(Pose& cur_pos)
    {
        grid_map::Position start_pos(cur_pos.x, cur_pos.y);
        // start_pos.x = cur_pos.x; 
        // start_pos.y = cur_pos.y;
        vector<grid_map::Index> lidar_free_vec; //Free voxels
        vector<grid_map::Index> lidar_collision_vec; //Occupied voxels
        lidar_free_vec.clear();
        lidar_collision_vec.clear();

        int ray_num = floor( (hangle_max_ - hangle_min_)/angle_resol_ );
        for(int i=0; i< ray_num; i++)
        {   
            double angle = cur_pos.yaw + angle_resol_ * ray_num;
            double end_pos_x = cur_pos.x + range_max_ * cos(angle);
            double end_pos_y = cur_pos.y + range_max_ * sin(angle);
            grid_map::Position end_pos(end_pos_x, end_pos_y);
            pair< vector<grid_map::Index>, grid_map::Index> idx = gen_single_ray(start_pos, end_pos); //Return free voxel index & Occupied voxel index

            lidar_free_vec.insert(lidar_free_vec.end(), idx.first.begin(), idx.first.end()); //Concatenate two vectors
            lidar_collision_vec.push_back(idx.second);
        }     
        update_map(lidar_free_vec, lidar_collision_vec);
    }

    void Lidar_sensor::update_map(vector<grid_map::Index>& free_vec, vector<grid_map::Index>& occupied_vec)
    {        
        double free = 0.1; double occupied = 0.9;
        double cur_occ_val; double update_occ_val;
        
        //Inverse sensor model
        //1. Free voxels
        for(vector<grid_map::Index>::iterator iter = free_vec.begin(); iter!=free_vec.end(); iter++)
        {
            cur_occ_val = belief_map_.at("base", *iter);
            update_occ_val = inverse_sensor(cur_occ_val, free);
        }
        //2. Occupied voxels
        for(vector<grid_map::Index>::iterator iter = occupied_vec.begin(); iter!=occupied_vec.end(); iter++)
        {
            cur_occ_val = belief_map_.at("base", *iter);
            update_occ_val = inverse_sensor(cur_occ_val, occupied);
        }

        // Update belief 
        // belief_map_.at("base", cur_idx) = value; 
    }

    double Lidar_sensor::inverse_sensor(double cur_val, double meas_val)
    {
        double log_cur = log(cur_val / (1.0 - cur_val));
        double log_prior = log( 0.5/ 0.5); 
        double log_meas = log(meas_val / (1.0 - meas_val));

        double log_update = log_meas + log_cur - log_prior;

        return 1.0-1.0/(1.0+exp(log_update));
    }

    pair<vector<grid_map::Index>, grid_map::Index> Lidar_sensor::gen_single_ray(grid_map::Position& start_pos, grid_map::Position& end_pos) //Single raycasting
    {   
        //TODO: Transform position to index
        int start_idxx = 0; int start_idxy = 1;
        int end_idxx = 0; int end_idxy = 1;
        grid_map::Index startIndex(start_idxx, start_idxy);
        grid_map::Index endIndex(end_idxx, end_idxy);
        
        // RayTracer raytracer; 
        // pair<vector<grid_map::Index>, bool> result = raytracer.raytracing(this, startIndex, endIndex);
    }

    /**
     * @brief RayTracing and return pair of voxel indices & whether collision occured. If 2nd element is true, beam is collided & 
     *        the last element of vector is occupied voxel. 
     * 
     * @param sensor 
     * @param startIndex 
     * @param endIndex 
     */
    pair<vector<grid_map::Index>, bool> RayTracer::raytracing(Lidar_sensor& sensor, grid_map::Index& startIndex, grid_map::Index& endIndex)
    {
        grid_map::LineIterator ray_tracer(gt_map_, startIndex, endIndex);
        vector<grid_map::Index> free_voxel;
        free_voxel.clear();
        grid_map::Index occupied_idx;

        for(ray_tracer; !ray_tracer.isPastEnd(); ++ray_tracer)
        {
            if(gt_map_.at("base", *ray_tracer) > 0.95) //Occupied Voxels
            {
                occupied_idx = *ray_tracer;
                free_voxel.push_back(occupied_idx);
                pair<vector<grid_map::Index>, bool> return_pair = make_pair(free_voxel, true);
                return return_pair;
            }
            else
            {
                free_voxel.push_back(*ray_tracer);
            }
        }
        pair<vector<grid_map::Index>, bool> return_pair = make_pair(free_voxel, false);

        // // for 
        // for (grid_map::GridMapIterator iterator(gt_map_); !iterator.isPastEnd(); ++iterator) {
        //     cout << "The value at index " << (*iterator).transpose() << " is " << gt_map_.at("layer", *iterator) << endl;
        // }

    }
}