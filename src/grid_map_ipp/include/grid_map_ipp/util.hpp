#include <grid_map_core/GridMap.hpp>
#include <grid_map_ipp/grid_map_ipp.hpp>
#include <grid_map_ipp/ObstacleGridConverter.hpp>

namespace grid_map
{
    class util
    {
        grid_map::GridMap map_;
        Lidar_sensor lidar_;

        public:
        util(){}
        ~util(){}
        
        grid_map::Position get_position(int idx, int idy)
        {   
            grid_map::Position pos;
            grid_map::Index index(idx, idy);
            
        }

    };
}