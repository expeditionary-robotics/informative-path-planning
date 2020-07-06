import matplotlib
# matplotlib.use('Agg')
# from continuous import *
from obstacles import *
from continuous import *

if __name__ == "__main__":
    range_max = 100.0
    ranges = (0., range_max, 0., range_max)
    world = BlockWorld(extent = ranges, num_blocks=3, dim_blocks=(5., 5.), centers = None )
    world.draw_obstacles()

    start_loc = (0.5, 0.5, 0.0)
    time_step = 150
    display = False
    gradient_on = True

    gradient_step_list = [0.0, 0.05, 0.1, 0.15, 0.20]

    ''' Options include mean, info_gain, and hotspot_info, mes'''
    reward_function = 'mean'

    world = Environment(ranges = ranges, # x1min, x1max, x2min, x2max constraints
                        NUM_PTS = 20, 
                        variance = 100.0, 
                        lengthscale = 3.0, 
                        visualize = False,
                        seed = 1)

    evaluation = Evaluation(world = world, 
                            reward_function = reward_function)

    # Gather some prior observations to train the kernel (optional)

    x1observe = np.linspace(ranges[0], ranges[1], 5)
    x2observe = np.linspace(ranges[2], ranges[3], 5)
    x1observe, x2observe = np.meshgrid(x1observe, x2observe, sparse = False, indexing = 'xy')  
    data = np.vstack([x1observe.ravel(), x2observe.ravel()]).T
    observations = world.sample_value(data)

    input_limit = [0.0, 10.0, -30.0, 30.0] #Limit of actuation 
    sample_number = 10 #Number of sample actions 

    planning_type = 'non_myopic'

    gradient_step = 0.0    
    print('range_max ' + str(range_max)+ ' iteration '+ ' gradient_step ' + str(gradient_step))
    iteration = 1
    planning = Planning_Result(planning_type, ranges, start_loc, input_limit, sample_number, time_step, display, gradient_on, gradient_step, iteration)


# if __name__ == "__main__":
#         range_exp = True
#     range_max_list = [20.0, 50.0, 100.0, 200.0]
#     if(range_exp):
#         for range_max in range_max_list:
#             ranges = (0., range_max, 0., range_max)
#             start_loc = (0.5, 0.5, 0.0)
#             time_step = 150
#             display = False
#             gradient_on = True

#             gradient_step_list = [0.0, 0.05, 0.1, 0.15, 0.20]

#             ''' Options include mean, info_gain, and hotspot_info, mes'''
#             reward_function = 'mean'

#             world = Environment(ranges = ranges, # x1min, x1max, x2min, x2max constraints
#                                 NUM_PTS = 20, 
#                                 variance = 100.0, 
#                                 lengthscale = 3.0, 
#                                 visualize = False,
#                                 seed = 1)

#             evaluation = Evaluation(world = world, 
#                                     reward_function = reward_function)

#             # Gather some prior observations to train the kernel (optional)

#             x1observe = np.linspace(ranges[0], ranges[1], 5)
#             x2observe = np.linspace(ranges[2], ranges[3], 5)
#             x1observe, x2observe = np.meshgrid(x1observe, x2observe, sparse = False, indexing = 'xy')  
#             data = np.vstack([x1observe.ravel(), x2observe.ravel()]).T
#             observations = world.sample_value(data)

#             input_limit = [0.0, 10.0, -30.0, 30.0] #Limit of actuation 
#             sample_number = 10 #Number of sample actions 

#             planning_type = 'non_myopic'
            
#             for iteration in range(5):
#                 for gradient_step in gradient_step_list:
#                     print('range_max ' + str(range_max)+ ' iteration ' + str(iteration) + ' gradient_step ' + str(gradient_step))
#                     planning = Planning_Result(planning_type, ranges, start_loc, input_limit, sample_number, time_step, display, gradient_on, gradient_step, iteration)

#     else:
#         ranges = (0., 20., 0., 20.)
#         start_loc = (0.5, 0.5, 0.0)
#         time_step = 150
#         display = False
#         gradient_on = True

#         gradient_step_list = [0.0, 0.05, 0.1, 0.15, 0.20]

#         ''' Options include mean, info_gain, and hotspot_info, mes'''
#         reward_function = 'mean'

#         world = Environment(ranges = ranges, # x1min, x1max, x2min, x2max constraints
#                             NUM_PTS = 20, 
#                             variance = 100.0, 
#                             lengthscale = 3.0, 
#                             visualize = False,
#                             seed = 1)

#         evaluation = Evaluation(world = world, 
#                                 reward_function = reward_function)

#         # Gather some prior observations to train the kernel (optional)

#         # ranges = (0., 20., 0., 20.)
#         x1observe = np.linspace(ranges[0], ranges[1], 5)
#         x2observe = np.linspace(ranges[2], ranges[3], 5)
#         x1observe, x2observe = np.meshgrid(x1observe, x2observe, sparse = False, indexing = 'xy')  
#         data = np.vstack([x1observe.ravel(), x2observe.ravel()]).T
#         observations = world.sample_value(data)

#         input_limit = [0.0, 10.0, -30.0, 30.0] #Limit of actuation 
#         sample_number = 10 #Number of sample actions 

#         planning_type = 'non_myopic'
        
#         for iteration in range(5):
#             for gradient_step in gradient_step_list:
#                 print('iteration ' + str(iteration) + ' gradient_step ' + str(gradient_step))
#                 planning = Planning_Result(planning_type, ranges, start_loc, input_limit, sample_number, time_step, display, gradient_on, gradient_step, iteration)
    
