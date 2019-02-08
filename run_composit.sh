#!/usr/bin/env bash

# Configurations
DO_PLAYBACK_DATA=false
BAG_FILE=
DO_BAG_DATA=true
DO_START_RVIZ=false

while [ "$1" != "" ]; do
    case $1 in
	--playback ) shift
		     BAG_FILE=$1
		     DO_PLAYBACK_DATA=true
		     ;;
	--record ) DO_BAG_DATA=true
		   ;;
	--rviz ) DO_START_RVIZ=true
		 ;;
	* ) exit 1
    esac
    shift
done


set -e

cleanup() {
        local pids=$(jobs -pr)
        [ -n "$pids" ] && kill $pids
	killall -9 bot-param-server
	deactivate
}
trap "cleanup" INT QUIT TERM EXIT

if [ "$DO_PLAYBACK_DATA" = true ]; then
    DO_USE_SIM_TIME=true
    DO_ENABLE_SENSORS=false
else
    DO_USE_SIM_TIME=false
    DO_ENABLE_SENSORS=true
fi

rosparam set use_sim_time $DO_USE_SIM_TIME

# Start rviz
if [ "$DO_START_RVIZ" = true ]; then
    { rviz & } &> /dev/null
    sleep 5
fi


# Give all devices the correct R/W permissions 
#echo "Granting permissions to devices"
sudo chmod 777 /dev/ttyACM*
sudo chmod 777 /dev/vesc

# Start LCM + scan matcher
source ~/rrg/dependencies/setup.sh
echo "Starting LCM & scan matcher"
{ source ~/rrg/src/platform/quad/perception_launch/scripts/quad_scan_matcher.sh & } &> /dev/null
sleep 5

source ~/rrg/devel/setup.bash
echo "Starting the estimator"
{ roslaunch perception_launch estimation.launch & } &> /dev/null
sleep 5

source ~/rrg/devel/setup.bash
echo "Starting Octomap"
{ roslaunch octomap_metrics_msgs octomap_server_composit.launch & } &> /dev/null
sleep 5

# { roslaunch composit_planner mapping.launch & } &> /dev/null
# sleep 2

#source ~/rrg/devel/setup.sh
#echo "Starting the COMPOSIT Planner"
# { roslaunch composit_planner car.launch real_sensor:=0 & }

# yorai learning 
#source ~/rrg/devel/setup.sh
#echo "starting the car_nav package -- ~~yorai learning~~"
# { roslaunch car_nav car_nav_launcher.launch & }
# sleep 2


# Start the vehicle
if [ "$DO_ENABLE_SENSORS" = true ]; then
    source ~/racecar-ws/venv/bin/activate
    source ~/racecar-ws/devel/setup.bash
    export PYTHONPATH=$PYTHONPATH:/usr/lib/python2.7/dist-packages
    echo "Starting car/sensors"
    { roslaunch racecar rrg_teleop.launch imu_name:=microstrain run_pp:=true & }
    sleep 5
fi


if [ "$DO_BAG_DATA" = true ]; then
    echo "Bagging Data"
    pushd ~/Desktop
    rosbag record /true_maxima /maxima_map /sample_map /pose /tf /tf_static /trajectory/current /projected_map /costmap /chem_data /chem_map /path_options /vis_chemworld
    popd
elif [ "$DO_PLAYBACK_DATA" = true ]; then
    echo "Playing back data"
    pushd ~/Desktop
    rosbag play $BAG_FILE --clock
else
    echo "Looping until sensors killed (no playback or bagging)"
    while :
    do
	sleep 1
    done
fi

echo "Killing the background processes"
cleanup
wait

echo "Done!"
