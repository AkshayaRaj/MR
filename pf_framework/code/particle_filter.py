#! /usr/bin/env python 

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from read_data import read_world, read_sensor_data

#add random seed for generating comparable pseudo random numbers
np.random.seed(123)

#plot preferences, interactive plotting mode
plt.axis([-1, 12, 0, 10])
plt.ion()
plt.show()

def sample_normal_distribution(mu,sigma):
    x = 0.5 * np.sum(np.random.uniform(-sigma,sigma,12))
    return mu + x # why add mu ? 

def plot_state(particles, landmarks, map_limits):
    # Visualizes the state of the particle filter.
    #
    # Displays the particle cloud, mean position and landmarks.
    
    xs = []
    ys = []

    for particle in particles:
        xs.append(particle['x'])
        ys.append(particle['y'])

    # landmark positions
    lx=[]
    ly=[]

    for i in range (len(landmarks)):
        lx.append(landmarks[i+1][0])
        ly.append(landmarks[i+1][1])

    # mean pose as current estimate
    estimated_pose = mean_pose(particles)

    # plot filter state
    plt.clf()
    plt.plot(xs, ys, 'r.')
    plt.plot(lx, ly, 'bo',markersize=10)
    plt.quiver(estimated_pose[0], estimated_pose[1], np.cos(estimated_pose[2]), np.sin(estimated_pose[2]), angles='xy',scale_units='xy')
    plt.axis(map_limits)

    plt.pause(0.01)

def initialize_particles(num_particles, map_limits):
    # randomly initialize the particles inside the map limits

    particles = []

    for i in range(num_particles):
        particle = dict()

        # draw x,y and theta coordinate from uniform distribution
        # inside map limits
        particle['x'] = np.random.uniform(map_limits[0], map_limits[1])
        particle['y'] = np.random.uniform(map_limits[2], map_limits[3])
        particle['theta'] = np.random.uniform(-np.pi, np.pi)

        particles.append(particle)

    return particles

def mean_pose(particles):
    # calculate the mean pose of a particle set.
    #
    # for x and y, the mean position is the mean of the particle coordinates
    #
    # for theta, we cannot simply average the angles because of the wraparound 
    # (jump from -pi to pi). Therefore, we generate unit vectors from the 
    # angles and calculate the angle of their average 

    # save x and y coordinates of particles
    xs = []
    ys = []

    # save unit vectors corresponding to particle orientations 
    vxs_theta = []
    vys_theta = []

    for particle in particles:
        xs.append(particle['x'])
        ys.append(particle['y'])

        #make unit vector from particle orientation
        vxs_theta.append(np.cos(particle['theta']))
        vys_theta.append(np.sin(particle['theta']))

    #calculate average coordinates
    mean_x = np.mean(xs)
    mean_y = np.mean(ys)
    mean_theta = np.arctan2(np.mean(vys_theta), np.mean(vxs_theta))

    return [mean_x, mean_y, mean_theta]

def sample_motion_model(odometry, particles):
    # Samples new particle positions, based on old positions, the odometry
    # measurements and the motion noise 
    # (probabilistic motion models slide 27)

    delta_rot1 = odometry['r1']
    delta_trans = odometry['t']
    delta_rot2 = odometry['r2']

    # the motion noise parameters: [alpha1, alpha2, alpha3, alpha4]
    noise = [0.1, 0.1, 0.05, 0.05]

    # generate new particle set after motion update
    new_particles = []
    
    '''your code here'''
    '''***        ***'''
    sigma_rot1 = noise[0]*abs(delta_rot1) + noise[1]*abs(delta_trans)
    sigma_trans = noise[2]*abs(delta_trans) + noise[3]*(abs(delta_rot1)+abs(delta_rot2))
    sigma_rot2 = noise[0] * abs(delta_rot2) + noise[1] * abs(delta_trans)
    

    # move each particle by odometry + noise
    for particle in particles:
        new_particle = dict()
        delta_hat_rot1 = delta_rot1 + sample_normal_distribution(0,sigma_rot1)
        delta_hat_trans = delta_trans + sample_normal_distribution(0,sigma_trans )
        delta_hat_rot2 = delta_rot2 + sample_normal_distribution(0, sigma_rot2 )

        # calculate new particle pose
        new_particle['x'] = particle['x'] + delta_hat_trans * np.cos( particle['theta'] + delta_hat_rot1 )
        new_particle['y'] = particle['y'] + delta_hat_trans * np.sin( particle['theta'] + delta_hat_rot1 )
        new_particle['theta'] = particle['theta'] + delta_hat_rot1 + delta_hat_rot2

        new_particles.append(new_particle)

    return new_particles

def eval_sensor_model(sensor_data, particles, landmarks):
    # Computes the observation likelihood of all particles, given the
    # particle and landmark positions and sensor measurements
    # (probabilistic sensor models slide 33)
    #
    # The employed sensor model is range only.
    # the function returns a list of weights for the given particle set

    sigma_r = 0.2

    #measured landmark ids and ranges
    ids = sensor_data['id']
    ranges = sensor_data['range']

    weights = []
    
    '''your code here'''
    '''***        ***'''
    # rate each particle according to the measurement likelihood , so if a particle has higher measurement likelihood then 
    # it gets a higher weight
    for particle in particles:

        all_meas_likelihood = 1.0
        # loop for each observed landmark
        for i in range(len(ids)):
            lm_id = ids[i]
            meas_range = ranges[i]
            l_x = landmarks[lm_id][0]
            l_y = landmarks[lm_id][1]
            px = particle['x']
            py = particle['y']
            meas_range_exp = np.sqrt((l_x - px)**2 + (l_y - py)**2)
            # evaluate sensor model
            meas_likelihood = scipy.stats.norm.pdf(meas_range, meas_range_exp, sigma_r)
            # combine (independent) measurements
            all_meas_likelihood = all_meas_likelihood * meas_likelihood

        weights.append(all_meas_likelihood) # so each particle gets a correspoing weight 
    
    normalizer = sum(weights)
    #weights = weights / normalizer
    weights = [x / normalizer for x in weights]

    return weights

def resample_particles(particles, weights):
    # Returns a new set of particles obtained by performing
    # stochastic universal sampling, according to the particle weights.

    new_particles = []

    step = 1.0 / len(particles)

    # random start of first pointer
    u = np.random.uniform(0,step)

    # c = []
    # c[0] = weights[0]
    # # generate cdf
    # for i in range (len(weights) - 1):
    #     c[i+1] = c[i] + weights[i+1]

    c = weights[0]
    i = 0

    # loop over all particle weights
    for particle in particles:
        # go through the weights until we find the particle to which the pointer points
        while u > c:
            c = c + weights[i]
            i = i + 1
        # add the particle 
        new_particles.append(particles[i])
        # increase the threshold
        u = u + step
    

    return new_particles

def main():
    # implementation of a particle filter for robot pose estimation

    print "Reading landmark positions"
    landmarks = read_world("../data/world.dat")

    print "Reading sensor data"
    sensor_readings = read_sensor_data("../data/sensor_data.dat")

    #initialize the particles
    map_limits = [-1, 12, 0, 10]
    particles = initialize_particles(1000, map_limits)

    #run particle filter
    for timestep in range(len(sensor_readings)/2):

        #plot the current state
        plot_state(particles, landmarks, map_limits)

        #predict particles by sampling from motion model with odometry info
        new_particles = sample_motion_model(sensor_readings[timestep,'odometry'], particles)

        #calculate importance weights according to sensor model
        weights = eval_sensor_model(sensor_readings[timestep, 'sensor'], new_particles, landmarks)

        #resample new particle set according to their importance weights
        particles = resample_particles(new_particles, weights)

    plt.show('hold')

if __name__ == "__main__":
    main()