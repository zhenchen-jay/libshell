#include "PunchDisk.h"
#include <igl/triangle/triangulate.h>

#include <sstream>
#include <cassert>
#include <iomanip>

void makePunchedDisk(double inner_radius,
                     double middle_radius,
                     double outer_radius,
                     double triangleArea,
                     Eigen::MatrixXd &V,
                     Eigen::MatrixXi &F) {
    constexpr double PI = 3.1415926535898;
    double abs_triangle_area = PI * outer_radius * outer_radius * triangleArea;
    double targetlength = 2.0 * std::sqrt(abs_triangle_area / std::sqrt(3.0));

    int inner_samples = std::max(3, int(2 * PI * inner_radius / targetlength));
    int middle_samples = std::max(3, int(2 * PI * middle_radius / targetlength));
    int outer_samples = std::max(3, int(2 * PI * outer_radius / targetlength));

    // Create vertices for all three circles
    Eigen::MatrixXd Vin(inner_samples + middle_samples + outer_samples, 2);
    Eigen::MatrixXi E(inner_samples + middle_samples + outer_samples, 2);
    Eigen::MatrixXd H(1, 2);
    Eigen::MatrixXd V2;
    Eigen::MatrixXi F2;

    H << 0, 0;

    int vrow = 0;
    int erow = 0;

    // Create inner circle vertices
    for (int i = 0; i < inner_samples; i++) {
        double theta = 2 * PI * i / inner_samples;
        Vin(vrow, 0) = inner_radius * std::cos(theta);
        Vin(vrow, 1) = inner_radius * std::sin(theta);
        vrow++;
    }

    // Create middle circle vertices
    for (int i = 0; i < middle_samples; i++) {
        double theta = 2 * PI * i / middle_samples;
        Vin(vrow, 0) = middle_radius * std::cos(theta);
        Vin(vrow, 1) = middle_radius * std::sin(theta);
        vrow++;
    }

    // Create outer circle vertices
    for (int i = 0; i < outer_samples; i++) {
        double theta = 2 * PI * i / outer_samples;
        Vin(vrow, 0) = outer_radius * std::cos(theta);
        Vin(vrow, 1) = outer_radius * std::sin(theta);
        vrow++;
    }

    // Create edges for inner circle
    for (int i = 0; i < inner_samples; i++) {
        E(erow, 0) = i;
        E(erow, 1) = (i + 1) % inner_samples;
        erow++;
    }

    // Create edges for middle circle
    for (int i = 0; i < middle_samples; i++) {
        E(erow, 0) = inner_samples + i;
        E(erow, 1) = inner_samples + (i + 1) % middle_samples;
        erow++;
    }

    // Create edges for outer circle
    for (int i = 0; i < outer_samples; i++) {
        E(erow, 0) = inner_samples + middle_samples + i;
        E(erow, 1) = inner_samples + middle_samples + (i + 1) % outer_samples;
        erow++;
    }

    // Triangulate the entire region
    std::stringstream ss;
    ss << "a" << std::setprecision(30) << std::fixed << triangleArea << "qDY";
    igl::triangle::triangulate(Vin, E, H, ss.str(), V2, F2);

    // Convert to 3D
    V.setZero(V2.rows(), 3);
    for (int i = 0; i < V2.rows(); i++) {
        V(i, 0) = V2(i, 0);
        V(i, 1) = V2(i, 1);
        V(i, 2) = 0;
    }

    F = F2;
}