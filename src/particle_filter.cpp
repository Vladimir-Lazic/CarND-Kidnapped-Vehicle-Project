/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>

#include <algorithm>
#include <cassert>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::normal_distribution;
using std::string;
using std::vector;

#define assertm(exp, msg) assert(((void)msg, exp))

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  num_particles = 50;  // TODO: Set the number of particles

  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(x, std[2]);

  std::default_random_engine gen;

  for (int i = 0; i < num_particles; i++) {
    Particle p;

    p.weight = 1.0;

    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    p.id = i;

    particles.push_back(p);
    weights.push_back(p.weight);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);

  std::default_random_engine gen;

  for (int i = 0; i < num_particles; i++) {
    double x_final = 0;
    double y_final = 0;
    double theta_final = 0;

    double x_initial = particles[i].x;
    double y_initial = particles[i].y;
    double theta_initial = particles[i].theta;

    if (fabs(yaw_rate) > 0.0001) {
      x_final = x_initial + (velocity / yaw_rate) *
                                (sin(theta_initial + yaw_rate * delta_t) -
                                 sin(theta_initial));

      y_final = y_initial + (velocity / yaw_rate) *
                                (cos(theta_initial) -
                                 cos(theta_initial + yaw_rate * delta_t));

      theta_final = theta_initial + yaw_rate * delta_t;
    } else {
      x_final = x_initial + velocity * delta_t * cos(theta_initial);

      y_final = y_initial + velocity * delta_t * sin(theta_initial);

      theta_final = theta_initial;
    }

    particles[i].x = x_final + dist_x(gen);
    particles[i].y = y_final + dist_y(gen);
    particles[i].theta = theta_final + dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted,
                                     vector<LandmarkObs> &observations) {
  for (int i = 0; i < observations.size(); i++) {
    double x_observed = observations[i].x;
    double y_observed = observations[i].y;

    double nearest_neighbor_distance = std::numeric_limits<double>::max();

    for (int j = 0; j < predicted.size(); j++) {
      double x_predicted = predicted[j].x;
      double y_predicted = predicted[j].y;

      double distance = sqrt(pow((x_observed - x_predicted), 2) +
                             pow((y_observed - y_predicted), 2));

      if (distance < nearest_neighbor_distance) {
        nearest_neighbor_distance = distance;
        observations[i].id = predicted[j].id;
      }
    }
  }
}

static double multiv_prob(double sig_x, double sig_y, double x_obs,
                          double y_obs, double mu_x, double mu_y) {
  // calculate normalization term

  double gauss_norm;
  gauss_norm = 1 / (2 * M_PI * sig_x * sig_y);

  // calculate exponent
  double exponent;
  exponent = (pow(x_obs - mu_x, 2) / (2 * pow(sig_x, 2))) +
             (pow(y_obs - mu_y, 2) / (2 * pow(sig_y, 2)));

  // calculate weight using normalization terms and exponent
  double weight;
  weight = gauss_norm * exp(-exponent);

  return weight;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {
  double weight_sum = 0.0;
  for (int i = 0; i < num_particles; i++) {
    double x_particle = particles[i].x;
    double y_particle = particles[i].y;
    double theta = particles[i].theta;

    vector<LandmarkObs> predicted_landmarks;
    for (int k = 0; k < map_landmarks.landmark_list.size(); k++) {
      Map::single_landmark_s lm = map_landmarks.landmark_list[k];
      LandmarkObs pred_lm{lm.id_i, lm.x_f, lm.y_f};
      double distance = dist(x_particle, y_particle, pred_lm.x, pred_lm.y);
      if (distance * distance <= sensor_range * sensor_range) {
        predicted_landmarks.push_back(pred_lm);
      }
    }

    vector<LandmarkObs> observation_landmarks;
    for (int j = 0; j < observations.size(); j++) {
      // Converting from the vehicle to map coordinate system
      double x_map, y_map;
      double x_obs = observations[j].x;
      double y_obs = observations[j].y;

      x_map = x_particle + (cos(theta) * x_obs) - (sin(theta) * y_obs);
      y_map = y_particle + (sin(theta) * x_obs) + (cos(theta) * y_obs);

      LandmarkObs landmark{observations[j].id, x_map, y_map};
      observation_landmarks.push_back(landmark);
    }

    dataAssociation(predicted_landmarks, observation_landmarks);

    particles[i].weight = 1.0;
    weights[i] = 1.0;

    double sig_x = std_landmark[0];
    double sig_y = std_landmark[1];

    for (int j = 0; j < observation_landmarks.size(); j++) {
      // Update weight
      double x = observation_landmarks[j].x;
      double y = observation_landmarks[j].y;

      double mu_x, mu_y;

      for (int n = 0; n < predicted_landmarks.size(); n++) {
        if (predicted_landmarks[n].id == observation_landmarks[j].id) {
          mu_x = predicted_landmarks[n].x;
          mu_y = predicted_landmarks[n].y;
          break;
        }
      }

      particles[i].weight *= multiv_prob(sig_x, sig_y, x, y, mu_x, mu_y);
    }
    weights[i] = particles[i].weight;
    weight_sum += weights[i];
  }

  // Normalize the particle weights
  if (fabs(weight_sum) > 0.0) {
    for (int i = 0; i < weights.size(); i++) {
      weights[i] = weights[i] / weight_sum;
    }
  }
}

void ParticleFilter::resample() {
  vector<Particle> new_particles;
  vector<double> particle_weights;

  for (int i = 0; i < weights.size(); i++) {
    particle_weights.push_back(weights[i]);
  }

  std::default_random_engine gen;
  std::discrete_distribution<> dist_particles(particle_weights.begin(),
                                              particle_weights.end());

  int index = -1;
  for (int i = 0; i < num_particles; i++) {
    index = dist_particles(gen);
    new_particles.push_back(particles[index]);
  }

  particles = new_particles;
}

void ParticleFilter::SetAssociations(Particle &particle,
                                     const vector<int> &associations,
                                     const vector<double> &sense_x,
                                     const vector<double> &sense_y) {
  // particle: the particle to which assign each listed association,
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}