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
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"
#include "map.h"

using namespace std;

inline double bivariate_pdf(double x, double y, double mu_x, double mu_y, double std_x, double std_y) {
  double norm = (1/(2*M_PI*std_x*std_y));
  double x_term = pow((x-mu_x)/std_x, 2);
  double y_term = pow((y-mu_y)/std_y, 2);
  return norm*exp(-0.5*(x_term + y_term));
}

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  num_particles = 100;
  default_random_engine gen;
  normal_distribution<double> dist_x(x, std[0]), dist_y(y, std[1]), dist_theta(theta, std[2]);
  for (int i=0; i<num_particles; i++) {
    Particle p;
    p.id = i;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    p.weight = 1.0;
    particles.push_back(p);
  }

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

  default_random_engine gen;
  double delta_x = 0.0, delta_y = 0.0, delta_theta = 0.0;
  for (int i=0; i<num_particles; i++) {
    if (yaw_rate == 0) {
      delta_theta = 0.0;
      delta_x = velocity * delta_t * cos(particles[i].theta);
      delta_y = velocity * delta_t * sin(particles[i].theta);
    } else {
      delta_theta = yaw_rate * delta_t;
      delta_x = (velocity/yaw_rate) * (sin(particles[i].theta + delta_theta) - sin(particles[i].theta));
      delta_y = (velocity/yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + delta_theta));
    }
    normal_distribution<double>
        dist_x(delta_x, std_pos[0]),
        dist_y(delta_y, std_pos[1]),
        dist_theta(delta_theta, std_pos[2]);
    particles[i].x += dist_x(gen);
    particles[i].y += dist_y(gen);
    particles[i].theta += dist_theta(gen);
  }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
  double min_distance = 10000000.0;
  for (LandmarkObs& obs: observations) {
    for (LandmarkObs p: predicted) {
      double distance = dist(p.x, p.y, obs.x, obs.y);
      if (min_distance > distance) {
        obs.id =  p.id;
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
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

  vector<Map::single_landmark_s> landmarks = map_landmarks.landmark_list; //filter_landmarks(map_landmarks.landmark_list, sensor_range);
  for (Particle& p: particles) {
    std::vector<LandmarkObs> predicted;
    for (Map::single_landmark_s landmark: landmarks) {
      LandmarkObs predicted_obs;
      double land_x = landmark.x_f, land_y = landmark.y_f;
      land_x -= p.x;
      land_y -= p.y;
      predicted_obs.x = land_x*cos(p.theta) + land_y*sin(p.theta);
      predicted_obs.y = - land_x*sin(p.theta) + land_y*cos(p.theta);
      predicted_obs.id = landmark.id_i;
      predicted.push_back(predicted_obs);
    }
    dataAssociation(predicted, observations);
    double weight = 1.0;
    for (LandmarkObs obs: observations) {
      for (LandmarkObs predicted_obs: predicted) {
        if (obs.id == predicted_obs.id) {
          double prob = bivariate_pdf(obs.x, obs.y, predicted_obs.x, predicted_obs.y, std_landmark[0], std_landmark[1]);
          weight *= prob;
        }
      }
    }
    p.weight = weight;
  }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  vector<double> weights;
  std::transform(particles.begin(), particles.end(), back_inserter(weights), [] (const Particle& p) { return p.weight; });
  default_random_engine gen;
  discrete_distribution<> dist(weights.begin(), weights.end());
  vector<Particle> resampled;
  for (int i=0; i<num_particles; i++) {
    int selected = dist(gen);
    resampled.push_back(particles[selected]);
  }
  particles = resampled;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
