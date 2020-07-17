#ifndef GRIDMAPSDF
#define GRIDMAPSDF

#include "grid_map_core/GridMap.hpp"
#include "grid_map_core/iterators/iterators.hpp"
#include "grid_map_sdf/SignedDistanceField.hpp"
#include <vector>
#include <Eigen/Dense>
#include <string>

namespace grid_map
{
    class GridMapSDF
    {
        private:
        double buffer_;
        grid_map::GridMap map_;
        grid_map::SignedDistanceField sdf_field_;

        public:
        GridMapSDF(double buffer, double map_size_x, double map_size_y, int num_obstacle, std::vector<Eigen::Array4d> obstacles)
        : buffer_(buffer)
        {
            grid_map::ObstacleGridConverter converter(map_size_x, map_size_y, num_obstacle, obstacles);
            map_ = converter.GridMapConverter();
        }

        void generate_SDF(std::string& layer)
        {
            sdf_field_.calculateSignedDistanceField(map_, layer, 1.5);
        }
        grid_map::Vector3 get_GradientValue(grid_map::Position& pos, std::string& layer)
        {
            sdf_field_.calculateSignedDistanceField(map_, layer, 1.5);
            grid_map::Vector3 gradient;
            gradient = sdf_field_.getDistanceGradientAt(grid_map::Vector3(pos.x(), pos.y(), 0.0));
            return gradient;
        }
        double get_Distance(grid_map::Position& pos, std::string& layer)
        {
            sdf_field_.calculateSignedDistanceField(map_, layer, 1.5);
            auto distance = sdf_field_.getDistanceAt(grid_map::Vector3(pos.x(), pos.y(), 0.0));
            return distance;
        }

    };


}

#endif