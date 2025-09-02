#pragma once

#include <Eigen/Core>

void makePunchedDisk(double inner_radius, double middle_radius, double outer_radius, double triangleArea, Eigen::MatrixXd &V, Eigen::MatrixXi &F);