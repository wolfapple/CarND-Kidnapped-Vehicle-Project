/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	default_random_engine gen;
	// Create normal distributions for x, y and theta
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);
	// Set number of particles
	num_particles = 100;
	// Initialize
	for (int i = 0; i < num_particles; i++) {
		Particle particle = {i, dist_x(gen), dist_y(gen), dist_theta(gen), 1.0};
		particles.push_back(particle);
		weights.push_back(particle.weight);
	}
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	default_random_engine gen;
	// Create normal distributions for noise
	normal_distribution<double> noise_x(0.0, std_pos[0]);
	normal_distribution<double> noise_y(0.0, std_pos[1]);
	normal_distribution<double> noise_theta(0.0, std_pos[2]);
	// Process
	for (int i = 0; i < num_particles; i++) {
		if (fabs(yaw_rate) < 0.001) {
			// moving straight
			double vd = velocity * delta_t;
			particles[i].x += vd * cos(particles[i].theta);
            particles[i].y += vd * sin(particles[i].theta);
		} else {
			double p_theta = particles[i].theta + (delta_t * yaw_rate);
			double vy = velocity / yaw_rate;
			particles[i].x += vy * (sin(p_theta) - sin(particles[i].theta));
      		particles[i].y += vy * (cos(particles[i].theta) - cos(p_theta));
      		particles[i].theta = p_theta;
		}
		// Add noise
    	particles[i].x += noise_x(gen);
    	particles[i].y += noise_y(gen);
    	particles[i].theta += noise_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for (int i = 0; i < observations.size(); i++) {		
		double min_distance = 1.0e9;
		for (int j = 0; j < predicted.size(); j++) {
			double distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
			if (distance < min_distance) {
				observations[i].id = j;
				min_distance = distance;
			}
		}
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html
	for (int i = 0; i < num_particles; i++) {
		double p_x = particles[i].x;
		double p_y = particles[i].y;
		double p_theta = particles[i].theta;
		// Transform to map coordinates
		vector<LandmarkObs> transformed_observations;		
		for (int j = 0; j < observations.size(); j++) {
			int o_id = observations[j].id;
			double o_x = observations[j].x;
			double o_y = observations[j].y;
			double t_x = p_x + o_x * cos(p_theta) - o_y * sin(p_theta);
			double t_y = p_y + o_y * cos(p_theta) + o_x * sin(p_theta);
			LandmarkObs to = {o_id, t_x, t_y};
			transformed_observations.push_back(to);
		}
		// Filtering landmarks
		vector<LandmarkObs> filtered_landmarks;
		for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {
			int l_id = map_landmarks.landmark_list[j].id_i;
			double l_x = map_landmarks.landmark_list[j].x_f;
      		double l_y = map_landmarks.landmark_list[j].y_f;
			double distance = dist(l_x, l_y, p_x, p_y);
			if (distance < sensor_range) {
				LandmarkObs l = {l_id, l_x, l_y};
				filtered_landmarks.push_back(l);
			}
		}
		// Nearest Neighbor
		dataAssociation(filtered_landmarks, transformed_observations);
		// Multivariate-Gaussian Probability
		double w = 1.0;
		for (int j = 0; j < transformed_observations.size(); j++) {
			int o_id = transformed_observations[j].id;			
			double d_x = transformed_observations[j].x - filtered_landmarks[o_id].x;
			double d_y = transformed_observations[j].y - filtered_landmarks[o_id].y;
			double std_x = std_landmark[0];
			double std_y = std_landmark[1];
			double mgp = 1 / (2 * M_PI * std_x * std_x) * exp(-((d_x * d_x / (2 * std_x * std_x)) + (d_y * d_y / (2 * std_y * std_y))));
			w *= mgp;
		}
		weights[i] = w;
		particles[i].weight = w;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	random_device rd;
	mt19937 gen(rd());
	discrete_distribution<> dd(weights.begin(), weights.end());
	vector<Particle> resampled_particles;
	for (int i = 0; i < num_particles; i++) {
		resampled_particles.push_back(particles[dd(gen)]);
	}
	particles = resampled_particles;
}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
