import rospy
import rosbag
import navpy
import numpy as np
import GPy

def truncate_by_distance(xvals, dist_lim=250.0):
    dist = 0
    last = None

    for i, pt in enumerate(xvals):
        if i == 0: 
            last = pt
            continue

        dist += np.sqrt((pt[0] - last[0])**2 + (pt[1]-last[1])**2)
        print dist
        last = pt

        if dist > dist_lim:
            break

    return i

def read_fulldataset(home = [13.1916987, -59.6419202, 0.00000]):
    # Read data from the bag file. Sets home to be the first lat-long coordiante observed. 
    # Every recieved Micron echo data point is associated with the lat-long message recieved immediately after it.
    # This could potentially be refined to interpolated between the lat-long message before/after.
    
    position_topic = '/slicklizard/gnc/mavros/global_position/global'
    data_topic = '/slicklizard/sensors/micron_echo/data'

    file_list = [#'/home/genevieve/mit-whoi/barbados/rosbag_16Jan_slicklizard/slicklizard_2019-01-16-16-12-40.bag',
		#'/home/genevieve/mit-whoi/barbados/rosbag_16Jan_slicklizard/slicklizard_2019-01-16-17-44-02.bag',
		'/home/genevieve/mit-whoi/barbados/rosbag_16Jan_slicklizard/slicklizard_2019-01-16-18-19-53.bag',
		#'/home/genevieve/mit-whoi/barbados/rosbag_16Jan_slicklizard/slicklizard_2019-01-17-01-51-47.bag']
		 '/home/genevieve/mit-whoi/barbados/rosbag_16Jan_slicklizard/slicklizard_2019-01-17-02-16-18.bag',
		 '/home/genevieve/mit-whoi/barbados/rosbag_16Jan_slicklizard/slicklizard_2019-01-17-03-01-44.bag',
		 '/home/genevieve/mit-whoi/barbados/rosbag_16Jan_slicklizard/slicklizard_2019-01-17-03-43-09.bag',
		 '/home/genevieve/mit-whoi/barbados/rosbag_16Jan_slicklizard/slicklizard_2019-01-17-04-07-11.bag']

    all_altitude = []
    all_locations = []

    prev_loc = current_loc = current_alt = None

    for fname in file_list:
	bag = rosbag.Bag(fname)

	for topic, msg, t in bag.read_messages(topics = [data_topic, position_topic]):
	    if topic == position_topic:
		# If a more recent altitude data point has been recieved, save the following lat-long coordinate
		if current_alt is not None:
		    loc = navpy.lla2ned(msg.latitude, msg.longitude, 0.0, home[0], home[1], 0.0)
		    if loc[0] <= 50.0 and loc[1] <= 50.0 and loc[0] >= 0.0 and loc[1] >= 0.0:
			all_altitude.append(-current_alt.range) # "depth" should be netagive
			# Log both the lat-long coordinate and the location in NED frame relative to the home position
			#loc = navpy.lla2ned(msg.latitude, msg.longitude, 0.0, home[0], home[1], 0.0)
			all_locations.append([loc[1], loc[0]])

			current_alt = None

	    elif topic == '/slicklizard/sensors/micron_echo/data':
		current_alt = msg       
		recent_time = msg.header.stamp.secs

    # Convert lists to ndarrays
    all_locations = np.array(all_locations).reshape((-1, 2)); 
    print "Mean altitude:", np.mean(all_altitude)
    # all_altitude = np.array(all_altitude-np.mean(all_altitude)).reshape((-1, 1))
    all_altitude = np.array(all_altitude-np.mean(all_altitude))
    print "Mean altitude:", np.mean(all_altitude)

    FILT_N = 5
    all_altitude = np.convolve(all_altitude, np.ones((FILT_N,))/FILT_N, mode='same').reshape((-1, 1))

    # Reject outliers in the main dataset (more then 2 standard deviations from the mean)
    outlier_index = (abs(all_altitude - np.mean(all_altitude)) < 2.0 * np.std(all_altitude)).reshape(-1, )
    xvals = all_locations[outlier_index, :]#[::3]
    zvals = all_altitude[outlier_index].reshape((-1, 1))#[::3]

    max_val = np.max(all_altitude)
    max_loc = all_locations[np.argmax(all_altitude), :]

    print "Max read in val:", max_val
    print "Max read in loc:", max_loc

    ranges = (0., 50., 0., 50.)

    kern = GPy.kern.RBF(input_dim = 2, lengthscale= 4.0543111858072445, variance=0.3215773006606948) + GPy.kern.White(input_dim=2, variance =  0.0862445597387173)
    mod = GPy.models.GPRegression(xvals[::5], zvals[::5], kern)

    # Create a discrete grid over which to plot the points
    x1vals = np.linspace(ranges[0], ranges[1], 100)
    x2vals = np.linspace(ranges[2], ranges[3], 100)
    x1, x2 = np.meshgrid(x1vals, x2vals, sparse = False, indexing = 'xy') # dimension: NUM_PTS x NUM_PTS       
    data = np.vstack([x1.ravel(), x2.ravel()]).T
    obs, var = mod.predict(data, full_cov = False, include_likelihood = True)

    return data, obs
    # return all_locations, all_altitude
    # return xvals, zvals

def read_bagfile(seed_bag, subsample = 1, home = [13.1916987, -59.6419202, 0.00000]):
    # Hard coded bag file and topic names
    bag = rosbag.Bag(seed_bag)
    position_topic = '/slicklizard/gnc/mavros/global_position/global'
    data_topic = '/slicklizard/sensors/micron_echo/data'

    altitude = []
    locations = []
    latitude = []
    longitude = []
    times = []
    prev_loc = current_loc = current_alt = None

    for topic, msg, t in bag.read_messages(topics = [data_topic, position_topic]):
        if topic == position_topic:
            # If a more recent altitude data point has been recieved, save the following lat-long coordinate
            if current_alt is not None:
                # Only take data points in the correct quadrant
                if msg.latitude > home[0] and msg.longitude > home[1]:
                    altitude.append(-(current_alt.range - 3.007621677259172)) 
                    loc = navpy.lla2ned(msg.latitude, msg.longitude, 0.0, home[0], home[1], 0.0)
                    locations.append([loc[1], loc[0]])
                    times.append(current_alt.header.stamp.secs)
                    current_alt = None

        elif topic == '/slicklizard/sensors/micron_echo/data':
            current_alt = msg       
                                
    # Convert lists to ndarrays
    locations = np.array(locations).reshape((-1, 2)); 
    # altitude = np.array(altitude-np.mean(altitude)).reshape((-1, 1))
    # altitude = np.array(altitude).reshape((-1, 1))
    altitude = np.array(altitude)

    FILT_N = 5
    altitude = np.convolve(altitude, np.ones((FILT_N,))/FILT_N, mode='same').reshape((-1, 1))

    # Reject outliers (more then 2 standard deviations from the mean) and subsamples the data
    outlier_index = (abs(altitude - np.mean(altitude)) < 2.0 * np.std(altitude)).reshape(-1, )
    # locations = locations[outlier_index, :][::10]
    # altitude = altitude[outlier_index][::10]
    locations = locations[outlier_index, :][::subsample]
    altitude = altitude[outlier_index][::subsample]

    return locations, altitude 
