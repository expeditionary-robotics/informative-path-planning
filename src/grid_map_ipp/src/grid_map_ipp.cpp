#include <grid_map_ipp/grid_map_ipp.hpp>
#include <grid_map_ipp/grid_map_sdf.hpp>
#include <Eigen/Dense>

namespace RayTracer{

    //Transform euclidean (x,y) position value to grid map reference frame
    Eigen::Vector2d Lidar_sensor::euc_to_gridref(Eigen::Vector2d pos)
    {
        // cout << "WHAT?" << endl;
        // pos(0) = pos(0) - map_size_x_/2.0;
        // pos(1) = pos(1) - map_size_y_/2.0;
        // cout << "WHAT2?" << endl;
        //Rotation (-pi/2) w.r.t. z direction
        Eigen::Vector2d grid_pos;
        grid_pos(0) = pos(1) - map_size_x_ /2.0;
        grid_pos(1) = -1.0*pos(0) + map_size_y_ /2.0;
        // cout << "WHAT3?" << grid_pos(0) << " " << grid_pos(1) << endl;
        return grid_pos;
    }
    
    void Lidar_sensor::get_measurement(Pose& cur_pos)
    {   /**
        cur_pos : Eucliden reference frame
        start_pos, end_pos : GridMap frame
        **/

        grid_map::Position pre_transform_pos(cur_pos.x, cur_pos.y);
        grid_map::Position start_pos = euc_to_gridref(pre_transform_pos);
        // cout << start_pos(0) << " " << start_pos(1) << endl;
        grid_map::Index startIndex;
        belief_map_.getIndex(start_pos, startIndex);
        // cout << startIndex(0) << " " << startIndex(1) << endl;

        // cout << "X " << cur_pos.x << " Y " << cur_pos.y << endl;
        vector<grid_map::Index> lidar_free_vec; //Free voxels
        vector<grid_map::Index> lidar_collision_vec; //Occupied voxels
        lidar_free_vec.clear();
        lidar_collision_vec.clear();

        int ray_num = floor( (hangle_max_ - hangle_min_)/angle_resol_ );
        for(int i=0; i< ray_num; i++)
        {   
            double angle = cur_pos.yaw + angle_resol_ * i;
            double end_pos_x = cur_pos.x + range_max_ * cos(angle);
            double end_pos_y = cur_pos.y + range_max_ * sin(angle);

            //Make sure each ray stays in environment range. 
            if(end_pos_x <0.0){
                end_pos_x = 0.1;
            }
            if(end_pos_x >= map_size_x_){
                end_pos_x = map_size_x_ - 0.5;
            }
            if(end_pos_y < 0.0){
                end_pos_y = 0.1;
            }
            if(end_pos_y >= map_size_y_){
                end_pos_y = map_size_y_ - 0.5;
            }

            grid_map::Position pre_transform_pos(end_pos_x, end_pos_y);
            grid_map::Position end_pos(euc_to_gridref(pre_transform_pos));
            pair< vector<grid_map::Index>, bool> idx = gen_single_ray(start_pos, end_pos); //Return free voxel index & true: Occupied voxel
                                                                                           //                          false: no Occupied voxel

            if(idx.second){
                lidar_free_vec.insert(lidar_free_vec.end(), idx.first.begin(), --idx.first.end()); //Concatenate two vectors
                lidar_collision_vec.push_back(idx.first.back());
            }
            else{
                lidar_free_vec.insert(lidar_free_vec.end(), idx.first.begin(), idx.first.end()); //Concatenate two vectors
            }
            // cout <<"After raycasting " << (*(--idx.first.end()))(0) <<" " << (*(--idx.first.end()))(1) <<endl;
            // cout << idx.second << endl;
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
            
            // cout << "CUR " << cur_occ_val << endl;
            // cout << "Free " << update_occ_val << endl;
            belief_map_.at("base", *iter) = update_occ_val;
        }
        //2. Occupied voxels
        for(vector<grid_map::Index>::iterator iter = occupied_vec.begin(); iter!=occupied_vec.end(); iter++)
        {
            cur_occ_val = belief_map_.at("base", *iter);
            update_occ_val = inverse_sensor(cur_occ_val, occupied);
            belief_map_.at("base", *iter) = update_occ_val;
            // cout << "Occ " << update_occ_val << endl;
        } 
    }

    double Lidar_sensor::inverse_sensor(double cur_val, double meas_val)
    {
        double log_cur = log(cur_val / (1.0 - cur_val));
        double log_prior = log( 0.5/ 0.5); 
        double log_meas = log(meas_val / (1.0 - meas_val));

        double log_update = log_meas + log_cur - log_prior;

        return 1.0-1.0/(1.0+exp(log_update));
    }

    pair<vector<grid_map::Index>, bool> Lidar_sensor::gen_single_ray(grid_map::Position& start_pos, grid_map::Position& end_pos) //Single raycasting
    {   
        grid_map::Index startIndex;
        belief_map_.getIndex(start_pos, startIndex);
        grid_map::Index endIndex;
        belief_map_.getIndex(end_pos, endIndex);
        
        // cout << "start_index " <<startIndex(0) << " " << startIndex(1) << endl;
        // cout << startIndex << endl;
        // cout <<"end_pos " << end_pos(0) << " " << end_pos(1) << endl;
        // cout << "end_index " << endIndex(0) << " " << endIndex(1) << endl;
                
        // RayTracer raytracer; 
        pair<vector<grid_map::Index>, bool> result = raytracer_.raytracing(*this, startIndex, endIndex);
        return result;
    }

    /**
     * @brief RayTracing and return pair of voxel indices & whether collision occured. If 2nd element is true, beam is collided & 
     *        the last element of vector is occupied voxel. 
     * 
     * @param sensor 
     * @param startIndex 
     * @param endIndex 
     */
    pair<vector<grid_map::Index>, bool> RayTracer::Raytracer::raytracing(Lidar_sensor& sensor, grid_map::Index& startIndex, grid_map::Index& endIndex)
    {
        grid_map::LineIterator line_iter(gt_map_, startIndex, endIndex);
        vector<grid_map::Index> free_voxel;
        free_voxel.clear();
        grid_map::Index occupied_idx;
        // cout << "Start" << endl;
        for(line_iter; !line_iter.isPastEnd(); ++line_iter)
        {
            //    cout << "Before IF" << endl;
            if(gt_map_.at("base", *line_iter) > 0.95) //Occupied Voxels
            {   //Out of map bound/???
                // cout << "In IF " << *line_iter <<endl;
                occupied_idx = *line_iter;
                free_voxel.push_back(occupied_idx);
                pair<vector<grid_map::Index>, bool> return_pair = make_pair(free_voxel, true);
                return return_pair;
            }
            else
            {
                // cout << "In Else" << endl;
                free_voxel.push_back(*line_iter);
            }
        }
        // cout << "After for loop" << endl;
        pair<vector<grid_map::Index>, bool> return_pair = make_pair(free_voxel, false);

        return return_pair;
    }
}