#include "ShellEnergy.h"

#include "../include/MeshConnectivity.h"
#include "../include/ElasticShell.h"
#include "../include/MidedgeAngleTanFormulation.h"
#include "../include/MidedgeAngleSinFormulation.h"
#include "../include/MidedgeAverageFormulation.h"
#include "../include/StVKMaterial.h"
#include "../include/TensionFieldStVKMaterial.h"
#include "../include/NeoHookeanMaterial.h"
#include "../include/RestState.h"
#include "../include/StVKMaterial.h"

#include "../Optimization/include/NewtonDescent.h"

#include "make_geometric_shapes/HalfCylinder.h"
#include "make_geometric_shapes/Sphere.h"

#include <polyscope/surface_vector_quantity.h>
#include <polyscope/polyscope.h>
#include <polyscope/surface_mesh.h>

#include <igl/readOBJ.h>
#include <igl/writePLY.h>
#include <igl/principal_curvature.h>
#include <set>
#include <vector>




double cokeRadius;
double cokeHeight;
double sphereRadius;

double thickness;
double poisson;

double triangleArea;

double QBEnergy;

enum class MeshType
{
    MT_CYLINDER_IRREGULAR,
    MT_CYLINDER_REGULAR,
    MT_SPHERE
};

MeshType curMeshType;

void lameParameters(double& alpha, double& beta)
{
    double young = 1.0 / thickness; // doesn't matter for static solves
    alpha = young * poisson / (1.0 - poisson * poisson);
    beta = young / 2.0 / (1.0 + poisson);
}

struct Energies
{
    double exact;
    double quadraticbending;
    double stvk;
    double stvkdir;
    double stvkCompressiveDir;
    double stvkIncompressibleDir;
};

double edgeDOFPenaltyEnergy(const Eigen::VectorXd& edgeDOFs, const Eigen::VectorXd&edge_area, double penaltyScale, Eigen::VectorXd* deriv, std::vector<Eigen::Triplet<double> >* hessian)
{
    int nedges = edge_area.size();
    if (deriv) {
        deriv->resize(edgeDOFs.size());
        deriv->setZero();
    }

    if (hessian) {
        hessian->clear();
    }

    if(edgeDOFs.size() == nedges || edgeDOFs.size() % nedges != 0)
        return 0.0;
    int ndof_per_edge = edgeDOFs.size() / nedges;

    double penalty = 0.0;
    for (int i = 0; i < nedges; i++) {
        penalty += (edgeDOFs(i * ndof_per_edge + 1) - 1) * (edgeDOFs(i * ndof_per_edge + 1) - 1) * penaltyScale * edge_area[i];
        penalty += (edgeDOFs(i * ndof_per_edge + 2) - 1) * (edgeDOFs(i * ndof_per_edge + 2) - 1) * penaltyScale * edge_area[i];

        if(deriv) {
            (*deriv)(i * ndof_per_edge + 1) += 2 * (edgeDOFs(i * ndof_per_edge + 1) - 1) * penaltyScale * edge_area[i];
            (*deriv)(i * ndof_per_edge + 2) += 2 * (edgeDOFs(i * ndof_per_edge + 2) - 1) * penaltyScale * edge_area[i];
        }

        if(hessian) {
            hessian->push_back({i * ndof_per_edge + 1, i * ndof_per_edge + 1, 2 * penaltyScale * edge_area[i]});
            hessian->push_back({i * ndof_per_edge + 2, i * ndof_per_edge + 2, 2 * penaltyScale * edge_area[i]});
        }
    } 
    return penalty;
}


void optimizeEdgeDOFs(ShellEnergy& energy, const Eigen::MatrixXd& curPos, const Eigen::VectorXd& edge_area,  Eigen::VectorXd& edgeDOFs)
{
    double tol = 1e-5;
    int nposdofs = curPos.rows() * 3;
    int nedgedofs = edgeDOFs.size();

    std::vector<Eigen::Triplet<double> > Pcoeffs;
    std::vector<Eigen::Triplet<double> > Icoeffs;
    for (int i = 0; i < nedgedofs; i++)
    {
        Pcoeffs.push_back({ i, nposdofs + i, 1.0 });
        Icoeffs.push_back({ i, i, 1.0 });
    }
    Eigen::SparseMatrix<double> P(nedgedofs, nposdofs + nedgedofs);
    P.setFromTriplets(Pcoeffs.begin(), Pcoeffs.end());
    Eigen::SparseMatrix<double> I(nedgedofs, nedgedofs);
    I.setFromTriplets(Icoeffs.begin(), Icoeffs.end());

    Eigen::SparseMatrix<double> PT = P.transpose();

    double stiffness = 1e3;
    double penaltyScale = stiffness * thickness;

    
    // energy, gradient, and hessian
    auto obj_func = [&](const Eigen::VectorXd& var, Eigen::VectorXd* grad, Eigen::SparseMatrix<double>* hessian,
                        bool psd_proj) {
        std::vector<Eigen::Triplet<double>> hessian_triplets;
        double total_energy = 0;
        
        double elastic_energy = energy.elasticEnergy(curPos, var, true, grad, hessian ? &hessian_triplets : nullptr,
                             psd_proj ? LibShell::HessianProjectType::kMaxZero : LibShell::HessianProjectType::kNone);
        total_energy += elastic_energy;

        Eigen::VectorXd edge_penalty_deriv;
        std::vector<Eigen::Triplet<double>> edge_penalty_hessian;
        double penalty = edgeDOFPenaltyEnergy(var, edge_area, penaltyScale, grad ? &edge_penalty_deriv : nullptr,
                                              hessian ? &edge_penalty_hessian : nullptr);

        total_energy += penalty;

        if (grad) {
            *grad = P * (*grad);
            *grad += edge_penalty_deriv;
        }

        if (hessian) {
            hessian->resize(var.size() + 3 * curPos.rows(), var.size() + 3 * curPos.rows());
            hessian->setFromTriplets(hessian_triplets.begin(), hessian_triplets.end());
            *hessian = P * (*hessian) * PT;

            Eigen::SparseMatrix<double> edge_penalty_hessian_mat(var.size(), var.size());
            edge_penalty_hessian_mat.setFromTriplets(edge_penalty_hessian.begin(), edge_penalty_hessian.end());
            *hessian += edge_penalty_hessian_mat;
        }
        return total_energy;
    };

    auto find_max_step = [&](const Eigen::VectorXd& x, const Eigen::VectorXd& dir) { return 1.0; };

    std::cout << "At beginning, elastic energy: " << energy.elasticEnergy(curPos, edgeDOFs, true, NULL, NULL)
              << ", penalty energy: " << edgeDOFPenaltyEnergy(edgeDOFs, edge_area, penaltyScale, NULL, NULL)
              << std::endl;

    OptSolver::NewtonSolver(obj_func, find_max_step, edgeDOFs, 1000, 1e-5, 1e-15, 1e-15, true, true, true);

    std::cout << "At the end, elastic energy: " << energy.elasticEnergy(curPos, edgeDOFs, true, NULL, NULL)
              << ", penalty energy: " << edgeDOFPenaltyEnergy(edgeDOFs, edge_area, penaltyScale, NULL, NULL)
              << std::endl;


    /*double reg = 1e-6;
    while (true)
    {
        std::vector<Eigen::Triplet<double> > Hcoeffs;
        Eigen::VectorXd F;
        double origEnergy = energy.elasticEnergy(curPos, edgeDOFs, true, &F, &Hcoeffs);
        
        std::vector<Eigen::Triplet<double>> PenaltyCoeffs;
        Eigen::VectorXd PenaltyDeriv;
        double penalty = edgeDOFPenaltyEnergy(edgeDOFs, edge_area, penaltyScale, &PenaltyDeriv, &PenaltyCoeffs);
        origEnergy += penalty;
        F.segment(nposdofs, nedgedofs) += PenaltyDeriv;
        for (auto& it : PenaltyCoeffs)
            Hcoeffs.push_back({it.row() + nposdofs, it.col() + nposdofs, it.value()});

        Eigen::VectorXd PF = P * F;
        std::cout << "Force resid now: " << PF.norm() << ", energy: " << origEnergy - penalty << ", penalty: " << penalty << ", reg: " << reg << std::endl;
        if (PF.norm() < tol) return;

        Eigen::SparseMatrix<double> H(nposdofs + nedgedofs, nposdofs + nedgedofs);
        H.setFromTriplets(Hcoeffs.begin(), Hcoeffs.end());
        Eigen::SparseMatrix<double> PHPT = P * H * PT;
        Eigen::SparseMatrix<double> M = PHPT + reg * I;
        Eigen::SimplicialLLT<Eigen::SparseMatrix<double> > solver(M);
        Eigen::VectorXd update = solver.solve(-PF);
        if (solver.info() != Eigen::Success) {
            std::cout << "Solve failed" << std::endl;
            reg *= 2.0;
            continue;
        }
        Eigen::VectorXd newedgeDOFs = edgeDOFs + update;
        double newenergy = energy.elasticEnergy(curPos, newedgeDOFs, true, NULL, NULL);
        if (newenergy > origEnergy)
        {
            std::cout << "Not a descent step, " << origEnergy << " -> " << newenergy << std::endl;
            reg *= 2.0;
            continue;
        }
        edgeDOFs = newedgeDOFs;
        reg *= 0.5;
    }*/
}


Energies measureCylinderEnergy(
    const LibShell::MeshConnectivity& mesh,
    const Eigen::MatrixXd& restPos,
    const Eigen::MatrixXd& curPos,
    double thickness,
    double lameAlpha,
    double lameBeta,
    double curRadius,
    double curHeight,
    Eigen::MatrixXd &nhForces,
    Eigen::MatrixXd &qbForces) {
    Energies result;

    // initialize default edge DOFs (edge director angles)
    Eigen::VectorXd edgeDOFs;
    LibShell::MidedgeAverageFormulation::initializeExtraDOFs(edgeDOFs, mesh,
                                                             restPos);

    Eigen::VectorXd zerodiredgeDOFs;
    LibShell::MidedgeAngleTanFormulation::initializeExtraDOFs(zerodiredgeDOFs, mesh,
                                                              restPos);

    // initialize the rest geometry of the shell
    LibShell::MonolayerRestState restState;
    LibShell::MonolayerRestState dirrestState;

    // set uniform thicknesses
    restState.thicknesses.resize(mesh.nFaces(), thickness);
    restState.lameAlpha.resize(mesh.nFaces(), lameAlpha);
    restState.lameBeta.resize(mesh.nFaces(), lameBeta);
    dirrestState.thicknesses.resize(mesh.nFaces(), thickness);
    dirrestState.lameAlpha.resize(mesh.nFaces(), lameAlpha);
    dirrestState.lameBeta.resize(mesh.nFaces(), lameBeta);

    // initialize first and second fundamental forms to those of input mesh
    LibShell::ElasticShell<LibShell::MidedgeAverageFormulation>::
        firstFundamentalForms(mesh, restPos, restState.abars);
    LibShell::ElasticShell<LibShell::MidedgeAverageFormulation>::
        secondFundamentalForms(mesh, restPos, edgeDOFs, restState.bbars);
    LibShell::ElasticShell<LibShell::MidedgeAngleTanFormulation>::
        firstFundamentalForms(mesh, restPos, dirrestState.abars);
    LibShell::ElasticShell<LibShell::MidedgeAngleTanFormulation>::
        secondFundamentalForms(mesh, restPos, zerodiredgeDOFs, dirrestState.bbars);

    std::vector<Eigen::Matrix2d> cur_bs;
    LibShell::ElasticShell<
        LibShell::MidedgeAverageFormulation>::secondFundamentalForms(mesh,
                                                                     curPos,
                                                                     edgeDOFs,
                                                                     cur_bs);

    // Make the half-cylinder rest-flat
    for (int i = 0; i < mesh.nFaces(); i++)
        restState.bbars[i].setZero();

    Eigen::VectorXd restEdgeDOFs = edgeDOFs;
    QuadraticBendingShellEnergy qbenergyModel(mesh, restState, restPos,
                                              restEdgeDOFs);
    StVKShellEnergy stvkenergyModel(mesh, restState);
    StVKDirectorShellEnergy stvk_dir_energyModel(mesh, dirrestState);
    StVKCompressiveDirectorShellEnergy stvk_compress_dir_energyModel(mesh, dirrestState);

    Eigen::VectorXd edge_area = Eigen::VectorXd::Zero(mesh.nEdges());

    for (int i = 0; i < mesh.nEdges(); i++) {
        for (int j = 0; j < 2; j++) {
            int fid = mesh.edgeFace(i, j);
            if(fid != -1) {
                edge_area[i] += std::sqrt(restState.abars[fid].determinant()) / 2.0 / 3.0;
            }
        } 
    }
   
    Eigen::VectorXd zero_compressed_diredgeDOFs;
    LibShell::MidedgeAngleCompressiveFormulation::initializeExtraDOFs(zero_compressed_diredgeDOFs, mesh, curPos);

    Eigen::VectorXd diredgeDOFs = zerodiredgeDOFs;
    std::cout << "============= Optimizing edge direction =========== " << std::endl;
    optimizeEdgeDOFs(stvk_dir_energyModel, curPos, edge_area, diredgeDOFs);

    int compressed_nedgedofs = zero_compressed_diredgeDOFs.size() / mesh.nEdges();
    for (int i = 0; i < mesh.nEdges(); i++) {
        zero_compressed_diredgeDOFs[i * compressed_nedgedofs] = diredgeDOFs[i];
        zero_compressed_diredgeDOFs[i * compressed_nedgedofs + 1] = 1;
        zero_compressed_diredgeDOFs[i * compressed_nedgedofs + 2] = 1;
    }
    result.stvkIncompressibleDir =
        stvk_compress_dir_energyModel.elasticEnergy(curPos, zero_compressed_diredgeDOFs, true, NULL, NULL);


    for (int i = 0; i < mesh.nEdges(); i++) {
        zero_compressed_diredgeDOFs[i * compressed_nedgedofs] = 0;
    }
    Eigen::VectorXd compresed_diredgeDOFs = zero_compressed_diredgeDOFs;
    Eigen::VectorXd tmp_vec = zero_compressed_diredgeDOFs;
    for (int i = 0; i < mesh.nEdges(); i++) {
        tmp_vec[i * zero_compressed_diredgeDOFs.size() / mesh.nEdges() + 1] = 0;
        tmp_vec[i * zero_compressed_diredgeDOFs.size() / mesh.nEdges() + 2] = 0;
    }
    std::cout << "zero magnitude penalty: " << edgeDOFPenaltyEnergy(tmp_vec, edge_area, 1.0, NULL, NULL) << std::endl;
    std::cout << "============= Optimizing edge direction and norm =========== " << std::endl;
    optimizeEdgeDOFs(stvk_compress_dir_energyModel, curPos, edge_area, compresed_diredgeDOFs);

    result.quadraticbending =
        qbenergyModel.elasticEnergy(curPos, edgeDOFs, true, NULL, NULL);
    result.stvk =
        stvkenergyModel.elasticEnergy(curPos, edgeDOFs, true, NULL, NULL);
    result.stvkdir = stvk_dir_energyModel.elasticEnergy(curPos, diredgeDOFs, true, NULL, NULL);
 
    result.stvkCompressiveDir =
        stvk_compress_dir_energyModel.elasticEnergy(curPos, compresed_diredgeDOFs, true, NULL, NULL);
    

    // ground truth energy
    // W = PI * r
    // r(x,y) = (r cos[x/r], r sin[x/r], y)^T
    // dr(x,y) = ((-sin[x/r], 0),
    //            ( cos[x/r], 0),
    //            ( 0, 1 ))
    Eigen::Matrix2d abar;
    abar.setIdentity();

    // n = (-sin[x/r], cos[x/r], 0) x (0, 0, 1) = ( cos[x/r], sin[x/r], 0 )
    // dn = ((-sin[x/r]/r, 0),
    //       ( cos[x/r]/r, 0),
    //       ( 0, 0 ))
    // b = dr^T dn = ((1/r, 0), (0, 0))
    Eigen::Matrix2d b;
    b << 1.0 / curRadius, 0, 0, 0;

    Eigen::Matrix2d M = abar.inverse() * b;
    double svnorm =
        lameAlpha / 2.0 * M.trace() * M.trace() + lameBeta * (M * M).trace();
    double coeff = thickness * thickness * thickness / 12.0;
    constexpr double PI = 3.1415926535898;
    double area = PI * curRadius * curHeight;

    result.exact = svnorm * coeff * area;

    return result;
}

Energies measureSphereEnergy(
    const LibShell::MeshConnectivity& mesh,
    const Eigen::MatrixXd& curPos,
    double thickness,
    double lameAlpha,
    double lameBeta,
    double radius) {
    Energies result;

    // initialize default edge DOFs (edge director angles)
    Eigen::VectorXd edgeDOFs;
    LibShell::MidedgeAverageFormulation::initializeExtraDOFs(edgeDOFs, mesh,
        curPos);

    Eigen::VectorXd zerodiredgeDOFs;
    LibShell::MidedgeAngleTanFormulation::initializeExtraDOFs(zerodiredgeDOFs, mesh,
        curPos);

    // initialize the rest geometry of the shell
    LibShell::MonolayerRestState restState;
    LibShell::MonolayerRestState dirrestState;

    // set uniform thicknesses
    restState.thicknesses.resize(mesh.nFaces(), thickness);
    restState.lameAlpha.resize(mesh.nFaces(), lameAlpha);
    restState.lameBeta.resize(mesh.nFaces(), lameBeta);
    dirrestState.thicknesses.resize(mesh.nFaces(), thickness);
    dirrestState.lameAlpha.resize(mesh.nFaces(), lameAlpha);
    dirrestState.lameBeta.resize(mesh.nFaces(), lameBeta);

    // initialize first and second fundamental forms to those of input mesh
    LibShell::ElasticShell<LibShell::MidedgeAverageFormulation>::
        firstFundamentalForms(mesh, curPos, restState.abars);
    LibShell::ElasticShell<LibShell::MidedgeAverageFormulation>::
        secondFundamentalForms(mesh, curPos, edgeDOFs, restState.bbars);
    LibShell::ElasticShell<LibShell::MidedgeAngleTanFormulation>::
        firstFundamentalForms(mesh, curPos, dirrestState.abars);
    LibShell::ElasticShell<LibShell::MidedgeAngleTanFormulation>::
        secondFundamentalForms(mesh, curPos, zerodiredgeDOFs, dirrestState.bbars);


    // Make the half-cylinder rest-flat
    for (int i = 0; i < mesh.nFaces(); i++)
    {
        restState.bbars[i].setZero();
        dirrestState.bbars[i].setZero();
    }

    Eigen::VectorXd edge_area(mesh.nEdges());

    for (int i = 0; i < mesh.nEdges(); i++) {
        for (int j = 0; j < 2; j++) {
            int fid = mesh.edgeFace(i, j);
            if (fid != -1) {
                edge_area[i] += std::sqrt(restState.abars[fid].determinant()) / 2.0 / 3.0;
            }
        }
    }

    Eigen::VectorXd restEdgeDOFs = edgeDOFs;

    QuadraticBendingShellEnergy qbenergyModel(mesh, restState, curPos,
        restEdgeDOFs);
    StVKShellEnergy stvkenergyModel(mesh, restState);
    StVKDirectorShellEnergy stvk_dir_energyModel(mesh, dirrestState);
    StVKCompressiveDirectorShellEnergy stvk_compress_dir_energyModel(mesh, dirrestState);

    Eigen::VectorXd diredgeDOFs = zerodiredgeDOFs;
    optimizeEdgeDOFs(stvk_dir_energyModel, edge_area, curPos, diredgeDOFs);

    result.quadraticbending =
        qbenergyModel.elasticEnergy(curPos, edgeDOFs, true, NULL, NULL);
    result.stvk =
        stvkenergyModel.elasticEnergy(curPos, edgeDOFs, true, NULL, NULL);
    result.stvkdir = stvk_dir_energyModel.elasticEnergy(curPos, diredgeDOFs, true, NULL, NULL);

    Eigen::VectorXd zero_compressed_diredgeDOFs;
    LibShell::MidedgeAngleCompressiveFormulation::initializeExtraDOFs(zero_compressed_diredgeDOFs, mesh, curPos);

    Eigen::VectorXd compresed_diredgeDOFs = zero_compressed_diredgeDOFs;
    optimizeEdgeDOFs(stvk_compress_dir_energyModel, curPos, edge_area, compresed_diredgeDOFs);

    result.quadraticbending = qbenergyModel.elasticEnergy(curPos, edgeDOFs, true, NULL, NULL);
    result.stvk = stvkenergyModel.elasticEnergy(curPos, edgeDOFs, true, NULL, NULL);
    result.stvkdir = stvk_dir_energyModel.elasticEnergy(curPos, diredgeDOFs, true, NULL, NULL);

    int compressed_nedgedofs = zero_compressed_diredgeDOFs.size() / mesh.nEdges();
    for (int i = 0; i < mesh.nEdges(); i++) {
        zero_compressed_diredgeDOFs[i * compressed_nedgedofs] = diredgeDOFs[i];
    }
    result.stvkCompressiveDir =
        stvk_compress_dir_energyModel.elasticEnergy(curPos, compresed_diredgeDOFs, true, NULL, NULL);
    result.stvkIncompressibleDir =
        stvk_dir_energyModel.elasticEnergy(curPos, zero_compressed_diredgeDOFs, true, NULL, NULL);

    // ground truth energy
    Eigen::Matrix2d abar;
    abar.setIdentity();

    Eigen::Matrix2d b;
    b << 1.0 / radius, 0, 0, 1.0 / radius;

    Eigen::Matrix2d M = abar.inverse() * b;
    double svnorm =
        lameAlpha / 2.0 * M.trace() * M.trace() + lameBeta * (M * M).trace();
    double coeff = thickness * thickness * thickness / 12.0;
    constexpr double PI = 3.1415926535898;
    double area = 4.0 * PI * radius * radius;

    result.exact = svnorm * coeff * area;

    return result;
}

int main(int argc, char* argv[])
{
    cokeRadius = 0.0325;
    cokeHeight = 0.122;
    sphereRadius = 0.05;

    triangleArea = 0.0000001;

    // curMeshType = MeshType::MT_CYLINDER_REGULAR;
    curMeshType = MeshType::MT_CYLINDER_IRREGULAR;
    // curMeshType = MeshType::MT_SPHERE;
    
    Energies curenergies;
    QBEnergy = 0;

    Eigen::MatrixXd nhForces;
    Eigen::MatrixXd qbForces;


    // set up material parameters
    thickness = 1.0;// 0.00010;
    poisson = 1.0 / 2.0;

    // load mesh

    Eigen::MatrixXd origV;
    Eigen::MatrixXd rolledV;
    Eigen::MatrixXi F;

    double lameAlpha, lameBeta;
    lameParameters(lameAlpha, lameBeta);
    double curRadius = cokeRadius;
    double curHeight = cokeHeight;

    int steps = 5;
    double multiplier = 4;
    std::ofstream log("log.txt");
    // Assuming log is an output stream
    log << std::left;  // Left-align columns
    log << "#V" << std::setw(5) << ":\t" << std::setw(15) << "exact energy" << std::setw(15)
        << "StVK energy" << std::setw(20)
        << "StVK_dir energy" << std::setw(20) << "StVK_incomp_dir" << std::setw(20) << "StVK_comp_dir" << std::setw(15)
        << "quadratic" << std::endl;
    if (curMeshType == MeshType::MT_SPHERE)
    {
        makeSphere(sphereRadius, triangleArea, origV, F);
        LibShell::MeshConnectivity mesh(F);
        for (int step = 0; step < steps; step++)
        {
            std::stringstream ss;
            ss << "sphere_ " << step << ".ply";
            igl::writePLY(ss.str(), origV, F);

            curenergies = measureSphereEnergy(mesh, origV, thickness, lameAlpha, lameBeta, sphereRadius);
            log << std::setw(5) << origV.rows() << ":\t" << std::setw(15) << curenergies.exact << std::setw(15)
                << curenergies.stvk << std::setw(20) << curenergies.stvkdir << std::setw(20)
                << curenergies.stvkIncompressibleDir << std::setw(20) << curenergies.stvkCompressiveDir << std::setw(15)
                << curenergies.quadraticbending << std::endl;
            triangleArea *= multiplier;
            makeSphere(sphereRadius, triangleArea, origV, F);
            mesh = LibShell::MeshConnectivity(F);            
        }
    }
    else
    {
        makeHalfCylinder(curMeshType == MeshType::MT_CYLINDER_REGULAR, cokeRadius, cokeHeight, triangleArea, origV, rolledV, F);
        LibShell::MeshConnectivity mesh(F);
        for (int step = 0; step < steps; step++)
        {
            curenergies = measureCylinderEnergy(mesh, origV, rolledV, thickness, lameAlpha, lameBeta, curRadius, curHeight, nhForces, qbForces);
            log << std::setw(5) << origV.rows() << ":\t" << std::setw(15) << curenergies.exact << std::setw(15)
                << curenergies.stvk << std::setw(20) << curenergies.stvkdir << std::setw(20)
                << curenergies.stvkIncompressibleDir << std::setw(20) << curenergies.stvkCompressiveDir << std::setw(15)
                << curenergies.quadraticbending << std::endl;
            triangleArea *= multiplier;
            makeHalfCylinder(curMeshType == MeshType::MT_CYLINDER_REGULAR, cokeRadius, cokeHeight, triangleArea, origV, rolledV, F);
            mesh = LibShell::MeshConnectivity(F);
        }
    }
    
}