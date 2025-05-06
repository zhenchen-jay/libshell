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
#include "../include/ExtraEnergyTermsBase.h"
#include "../include/ExtraEnergyTermsGeneralFormulation.h"
#include "../include/ExtraEnergyTermsGeneralSinFormulation.h"
#include "../include/ExtraEnergyTermsGeneralTanFormulation.h"
#include "../include/ExtraEnergyTermsSinFormulation.h"
#include "../include/ExtraEnergyTermsTanFormulation.h"
#include <tuple>
#include <utility>     // For std::tie
#include <filesystem>  // For std::filesystem

#include "../Optimization/include/NewtonDescent.h"
#include "../src/GeometryDerivatives.h"
#include "igl/boundary_loop.h"
#include "igl/null.h"

#include "make_geometric_shapes/HalfCylinder.h"
#include "make_geometric_shapes/Cylinder.h"
#include "make_geometric_shapes/Sphere.h"
#include "make_geometric_shapes/PunchDisk.h"
#include "spdlog/fmt/bundled/chrono.h"

#include <polyscope/surface_vector_quantity.h>
#include <polyscope/polyscope.h>
#include <polyscope/surface_mesh.h>
#include <polyscope/point_cloud.h>

#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <igl/writePLY.h>
#include <igl/principal_curvature.h>
#include <set>
#include <vector>

#include <CLI/CLI.hpp>

enum class SFFModelType { kS1Sin, kS1Tan, kS2Sin };
enum class MaterialType { kStVK, kNeoHookean };

bool is_include_III = true;
double inner_radius = 0.1;
double middle_radius = 0.4;
double outer_radius = 0.5;
int num_iters = 5000;

std::vector<int> fixed_verts;

double target_height = 0.2;
double shell_thickness = 1e-3;

struct OptimizationLog {
    OptSolver::NewtonSolverLog log;
    double membrane_energy = 0;
    double bending_energy = 0;
    double III_energy = 0;
    double mag_comp_energy = 0;
    double direct_perp_energy = 0;
    double total_energy = 0;

    void print() {
        std::cout << "=== Newton solver log: \n"
                  << "- energy: " << log.energy << "\n"
                  << "- grad_norm: " << log.grad_norm << "\n"
                  << "- x_norm: " << log.x_norm << "\n"
                  << "- f_norm: " << log.f_norm << "\n"
                  << "- newton_dec: " << log.newton_dec << "\n"
                  << "- line_search_step_size: " << log.line_search_step_size << "\n"
                  << "- terminate_iter: " << log.terminate_iter << "\n"
                  << "- is_converged: " << log.is_converged << std::endl;
        std::cout << "=== Total energy: " << total_energy << std::endl;
        std::cout << "- membrane energy: " << membrane_energy << std::endl;
        std::cout << "- bending energy: " << bending_energy << std::endl;
        std::cout << "- III energy: " << III_energy << std::endl;
        std::cout << "- mag comp energy: " << mag_comp_energy << std::endl;
        std::cout << "- direct perp energy: " << direct_perp_energy << std::endl;
    }

    void save(const std::string& filename) {
        std::ofstream file(filename);
        file << "=== Newton solver log: \n"
             << "- energy: " << log.energy << "\n"
             << "- grad_norm: " << log.grad_norm << "\n"
             << "- x_norm: " << log.x_norm << "\n"
             << "- f_norm: " << log.f_norm << "\n"
             << "- newton_dec: " << log.newton_dec << "\n"
             << "- line_search_step_size: " << log.line_search_step_size << "\n"
             << "- terminate_iter: " << log.terminate_iter << "\n"
             << "- is_converged: " << log.is_converged << "\n";
        file << "=== Total energy: " << total_energy << "\n";
        file << "- membrane energy: " << membrane_energy << "\n";
        file << "- bending energy: " << bending_energy << "\n";
        file << "- III energy: " << III_energy << "\n";
        file << "- mag comp energy: " << mag_comp_energy << "\n";
        file << "- direct perp energy: " << direct_perp_energy << "\n";
    }
};

void lameParameters(double youngs, double poisson, double& alpha, double& beta) {
    alpha = youngs * poisson / (1.0 - poisson * poisson);
    beta = youngs / 2.0 / (1.0 + poisson);
}

std::vector<Eigen::Vector3d> get_face_edge_normal_vectors(const Eigen::MatrixXd& cur_pos,
                                                          const LibShell::MeshConnectivity& mesh,
                                                          const Eigen::VectorXd& edge_dofs) {
    std::vector<Eigen::Vector3d> face_edge_normals = {};
    int nfaces = mesh.nFaces();

    for (int i = 0; i < nfaces; i++) {
        std::vector<Eigen::Vector3d> general_edge_normals =
            LibShell::MidedgeAngleGeneralSinFormulation::get_face_edge_normals(mesh, cur_pos, edge_dofs, i);
        for (int j = 0; j < 3; j++) {
            face_edge_normals.push_back(general_edge_normals[j]);
        }
    }
    return face_edge_normals;
}

std::unordered_map<int, double> build_boundary_fixed_vertex_map(const Eigen::MatrixXd& cur_pos,
                                                                const LibShell::MeshConnectivity& mesh) {
    std::unordered_map<int, double> fixed_vertex_dofs;
    for (auto id : fixed_verts) {
        for (int j = 0; j < 3; j++) {
            fixed_vertex_dofs[3 * id + j] = cur_pos(id, j);
        }
    }

    return fixed_vertex_dofs;
}

void optimizeFullDOFs(ShellEnergy& energy,
                      const std::vector<Eigen::Matrix2d>& abars,
                      const LibShell::MeshConnectivity& cur_mesh,
                      Eigen::MatrixXd& cur_pos,
                      Eigen::VectorXd& cur_edge_dofs,
                      std::shared_ptr<LibShell::ExtraEnergyTermsBase> extra_energy_terms = nullptr,
                      const std::unordered_map<int, double>* fixed_vert_dofs = nullptr,
                      const std::unordered_set<int>* fixed_edge_dofs = nullptr,
                      OptimizationLog* opt_log = nullptr) {
    double tol = 1e-5;
    int nposdofs = cur_pos.rows() * 3;
    int nedgedofs = cur_edge_dofs.size();
    int nfulldofs = nposdofs + nedgedofs;

    std::vector<Eigen::Triplet<double>> Pcoeffs;
    int row = 0;
    for (int i = 0; i < nposdofs; i++) {
        if (!fixed_vert_dofs || fixed_vert_dofs->count(i) == 0) {
            Pcoeffs.push_back({row, i, 1.0});
            row++;
        }
    }
    for (int i = 0; i < nedgedofs; i++) {
        if (!fixed_edge_dofs || fixed_edge_dofs->count(i) == 0) {
            Pcoeffs.push_back({row, nposdofs + i, 1.0});
            row++;
        }
    }
    Eigen::SparseMatrix<double> P(row, nfulldofs);
    P.setFromTriplets(Pcoeffs.begin(), Pcoeffs.end());

    std::cout << "P size: " << P.rows() << " " << P.cols() << std::endl;

    Eigen::SparseMatrix<double> PT = P.transpose();

    Eigen::VectorXd fixed_full_dofs_val = Eigen::VectorXd::Zero(nfulldofs);
    if (fixed_vert_dofs) {
        for (auto& id : *fixed_vert_dofs) {
            fixed_full_dofs_val(id.first) = id.second;
        }
    }

    if (fixed_edge_dofs) {
        for (auto& id : *fixed_edge_dofs) {
            fixed_full_dofs_val(nposdofs + id) = cur_edge_dofs(id);
        }
    }

    auto separate_full_dofs = [&](const Eigen::VectorXd& full_dofs) {
        Eigen::MatrixXd converted_pos = cur_pos;
        Eigen::VectorXd converted_edge_dofs = cur_edge_dofs;
        for (int i = 0; i < cur_pos.rows(); i++) {
            for (int j = 0; j < 3; j++) {
                converted_pos(i, j) = full_dofs(i * 3 + j);
            }
        }
        for (int j = 0; j < cur_edge_dofs.size(); j++) {
            converted_edge_dofs(j) = full_dofs(nposdofs + j);
        }
        return std::make_pair(converted_pos, converted_edge_dofs);
    };

    auto combine_2_full_dofs = [&](const Eigen::MatrixXd& pos_to_combine, const Eigen::VectorXd& edge_dofs_to_combine) {
        Eigen::VectorXd full_dofs = Eigen::VectorXd::Zero(nfulldofs);
        for (int i = 0; i < cur_pos.rows(); i++) {
            for (int j = 0; j < 3; j++) {
                full_dofs(i * 3 + j) = pos_to_combine(i, j);
            }
        }
        for (int j = 0; j < cur_edge_dofs.size(); j++) {
            full_dofs(nposdofs + j) = edge_dofs_to_combine(j);
        }
        return full_dofs;
    };

    auto convert_var_2_pos_edge_dofs = [&](const Eigen::VectorXd& var) {
        auto full_dofs = PT * var + fixed_full_dofs_val;
        return separate_full_dofs(full_dofs);
    };

    auto convert_pos_edge_dofs_2_var = [&](const Eigen::MatrixXd& pos_to_convert,
                                           const Eigen::VectorXd& edge_dofs_to_convert) {
        Eigen::VectorXd full_dofs = combine_2_full_dofs(pos_to_convert, edge_dofs_to_convert);
        Eigen::VectorXd reduced_dofs = P * full_dofs;
        return reduced_dofs;
    };

    // energy, gradient, and hessian
    auto obj_func = [&](const Eigen::VectorXd& var, Eigen::VectorXd* grad, Eigen::SparseMatrix<double>* hessian,
                        bool psd_proj) {
        std::vector<Eigen::Triplet<double>> hessian_triplets;
        double total_energy = 0;

        auto [pos, edge_dofs] = convert_var_2_pos_edge_dofs(var);

        double elastic_energy = energy.elasticEnergy(
            pos, edge_dofs, true, true, grad, hessian ? &hessian_triplets : nullptr,
            psd_proj ? LibShell::HessianProjectType::kMaxZero : LibShell::HessianProjectType::kNone);

        total_energy += elastic_energy;

        if (grad) {
            *grad = P * (*grad);
        }

        if (hessian) {
            hessian->resize(edge_dofs.size() + 3 * cur_pos.rows(), edge_dofs.size() + 3 * cur_pos.rows());
            hessian->setFromTriplets(hessian_triplets.begin(), hessian_triplets.end());
            *hessian = P * (*hessian) * PT;
        }

        if (extra_energy_terms || nedgedofs == 4 * cur_mesh.nEdges()) {
            Eigen::VectorXd mag_comp_deriv, direct_perp_deriv, III_deriv;
            std::vector<Eigen::Triplet<double>> mag_comp_triplets, direct_perp_triplets, III_triplets;

            double mag_comp = extra_energy_terms->compute_magnitude_compression_energy(
                edge_dofs, cur_mesh, grad ? &mag_comp_deriv : nullptr, hessian ? &mag_comp_triplets : nullptr,
                psd_proj);
            double direct_perp = extra_energy_terms->compute_vector_perp_tangent_energy(
                pos, edge_dofs, cur_mesh, abars, grad ? &direct_perp_deriv : nullptr,
                hessian ? &direct_perp_triplets : nullptr, psd_proj);
            double III = extra_energy_terms->compute_thirdFundamentalForm_energy(
                pos, edge_dofs, cur_mesh, abars, grad ? &III_deriv : nullptr, hessian ? &III_triplets : nullptr,
                psd_proj);

            total_energy += mag_comp;
            total_energy += direct_perp;

            if (is_include_III) {
                total_energy += III;
            }

            if (grad) {
                Eigen::VectorXd full_mag_comp_deriv = Eigen::VectorXd::Zero(nfulldofs);
                full_mag_comp_deriv.segment(nposdofs, nedgedofs) = mag_comp_deriv;
                *grad += P * (direct_perp_deriv + full_mag_comp_deriv);

                if (is_include_III) {
                    *grad += P * III_deriv;
                }
            }
            if (hessian) {
                Eigen::SparseMatrix<double> mag_comp_hess, direct_perp_hess, III_hess;

                std::vector<Eigen::Triplet<double>> full_mag_comp_triplets;
                for (int i = 0; i < mag_comp_triplets.size(); i++) {
                    full_mag_comp_triplets.push_back(Eigen::Triplet<double>(mag_comp_triplets[i].row() + nposdofs,
                                                                            mag_comp_triplets[i].col() + nposdofs,
                                                                            mag_comp_triplets[i].value()));
                }

                mag_comp_hess.resize(nfulldofs, nfulldofs);
                mag_comp_hess.setFromTriplets(full_mag_comp_triplets.begin(), full_mag_comp_triplets.end());
                mag_comp_hess = P * mag_comp_hess * PT;

                direct_perp_hess.resize(edge_dofs.size() + 3 * cur_pos.rows(), edge_dofs.size() + 3 * cur_pos.rows());
                direct_perp_hess.setFromTriplets(direct_perp_triplets.begin(), direct_perp_triplets.end());
                direct_perp_hess = P * direct_perp_hess * PT;

                III_hess.resize(edge_dofs.size() + 3 * cur_pos.rows(), edge_dofs.size() + 3 * cur_pos.rows());
                III_hess.setFromTriplets(III_triplets.begin(), III_triplets.end());
                III_hess = P * III_hess * PT;

                *hessian += mag_comp_hess;

                *hessian += direct_perp_hess;

                if (is_include_III) {
                    *hessian += III_hess;
                }
            }
        }

        return total_energy;
    };

    auto find_max_step = [&](const Eigen::VectorXd& x, const Eigen::VectorXd& dir) { return 1.0; };

    std::cout << "------------------------ At beginning -------------------------" << std::endl
              << "elastic energy: " << energy.elasticEnergy(cur_pos, cur_edge_dofs, true, true, NULL, NULL) << std::endl
              << "membrane energy: " << energy.elasticEnergy(cur_pos, cur_edge_dofs, true, false, NULL, NULL)
              << std::endl
              << "bending energy: " << energy.elasticEnergy(cur_pos, cur_edge_dofs, false, true, NULL, NULL)
              << std::endl;

    if (extra_energy_terms) {
        double mag_comp =
            extra_energy_terms->compute_magnitude_compression_energy(cur_edge_dofs, cur_mesh, nullptr, nullptr, false);
        double direct_perp = extra_energy_terms->compute_vector_perp_tangent_energy(cur_pos, cur_edge_dofs, cur_mesh,
                                                                                    abars, nullptr, nullptr, false);
        double III_term = extra_energy_terms->compute_thirdFundamentalForm_energy(cur_pos, cur_edge_dofs, cur_mesh,
                                                                                  abars, nullptr, nullptr, false);

        std::cout << "||m^2 - 1||^2: " << mag_comp << std::endl;
        std::cout << "direct perp: " << direct_perp << std::endl;
        std::cout << "III: " << III_term << std::endl;
    }

    Eigen::VectorXd x = convert_pos_edge_dofs_2_var(cur_pos, cur_edge_dofs);
    OptSolver::TestFuncGradHessian(obj_func, x);

    OptSolver::NewtonSolverLog log;

    OptSolver::NewtonSolver(obj_func, find_max_step, x, num_iters, tol, 1e-15, 1e-15, true, true, true, &log);

    if (opt_log) {
        opt_log->log = log;
    }

    std::tie(cur_pos, cur_edge_dofs) = convert_var_2_pos_edge_dofs(x);

    std::cout << "------------------------ At end -------------------------" << std::endl
              << "elastic energy: " << energy.elasticEnergy(cur_pos, cur_edge_dofs, true, true, NULL, NULL) << std::endl
              << "membrane energy: " << energy.elasticEnergy(cur_pos, cur_edge_dofs, true, false, NULL, NULL)
              << std::endl
              << "bending energy: " << energy.elasticEnergy(cur_pos, cur_edge_dofs, false, true, NULL, NULL)
              << std::endl;

    if (opt_log) {
        opt_log->membrane_energy = energy.elasticEnergy(cur_pos, cur_edge_dofs, true, false, NULL, NULL);
        opt_log->bending_energy = energy.elasticEnergy(cur_pos, cur_edge_dofs, false, true, NULL, NULL);
    }

    if (extra_energy_terms) {
        double mag_comp =
            extra_energy_terms->compute_magnitude_compression_energy(cur_edge_dofs, cur_mesh, nullptr, nullptr, false);
        double direct_perp = extra_energy_terms->compute_vector_perp_tangent_energy(cur_pos, cur_edge_dofs, cur_mesh,
                                                                                    abars, nullptr, nullptr, false);
        double III_term = extra_energy_terms->compute_thirdFundamentalForm_energy(cur_pos, cur_edge_dofs, cur_mesh,
                                                                                  abars, nullptr, nullptr, false);
        std::cout << "||m^2 - 1||^2: " << mag_comp << std::endl;
        std::cout << "direct perp: " << direct_perp << std::endl;
        std::cout << "III: " << III_term << std::endl;

        if (opt_log) {
            opt_log->mag_comp_energy = mag_comp;
            opt_log->direct_perp_energy = direct_perp;
            opt_log->III_energy = III_term;
        }
    }

    if (opt_log) {
        opt_log->total_energy = opt_log->membrane_energy + opt_log->bending_energy + opt_log->mag_comp_energy +
                                opt_log->direct_perp_energy + opt_log->III_energy;
    }
}

std::pair<std::shared_ptr<ShellEnergy>, std::shared_ptr<LibShell::ExtraEnergyTermsBase>> initialization(
    const LibShell::MeshConnectivity& rest_mesh,
    const Eigen::MatrixXd& rest_pos,
    const LibShell::MeshConnectivity& cur_mesh,  // we need this since we will stitch to get the current intial mesh
    double thickness,
    double young,
    double poisson,
    Eigen::VectorXd& edge_dofs,
    LibShell::MonolayerRestState& rest_state,
    SFFModelType model_type,
    MaterialType material_type) {
    rest_state.abars.clear();
    rest_state.thicknesses.clear();
    rest_state.bbars.clear();
    rest_state.lameAlpha.clear();
    rest_state.lameAlpha.clear();
    rest_state.thicknesses.resize(rest_mesh.nFaces(), thickness);
    double lame_alpha, lame_beta;
    lameParameters(young, poisson, lame_alpha, lame_beta);
    rest_state.lameAlpha.resize(rest_mesh.nFaces(), lame_alpha);
    rest_state.lameBeta.resize(rest_mesh.nFaces(), lame_beta);

    // initialize first and second fundamental forms to those of input mesh
    LibShell::ElasticShell<LibShell::MidedgeAngleTanFormulation>::firstFundamentalForms(rest_mesh, rest_pos,
                                                                                        rest_state.abars);
    rest_state.bbars = rest_state.abars;
    for (auto& mat : rest_state.bbars) {
        mat.setZero();
    }

    double shear = young / (2.0 * (1.0 + poisson));

    std::shared_ptr<LibShell::ExtraEnergyTermsBase> extra_energy_terms;
    std::shared_ptr<ShellEnergy> dir_energy_model;

    switch (model_type) {
        case SFFModelType::kS1Sin: {
            LibShell::MidedgeAngleSinFormulation::initializeExtraDOFs(edge_dofs, cur_mesh, rest_pos);
            extra_energy_terms = std::make_shared<LibShell::ExtraEnergyTermsSinFormulation>();
            extra_energy_terms->initialization(rest_pos, rest_mesh, young, shear, thickness, poisson, 3);
            if (material_type == MaterialType::kStVK) {
                dir_energy_model = std::make_shared<StVKS1DirectorSinShellEnergy>(cur_mesh, rest_state);
            } else if (material_type == MaterialType::kNeoHookean) {
                dir_energy_model = std::make_shared<NeohookeanS1DirectorSinShellEnergy>(cur_mesh, rest_state);
            }
            break;
        }

        case SFFModelType::kS1Tan: {
            LibShell::MidedgeAngleTanFormulation::initializeExtraDOFs(edge_dofs, cur_mesh, rest_pos);
            extra_energy_terms = std::make_shared<LibShell::ExtraEnergyTermsTanFormulation>();
            extra_energy_terms->initialization(rest_pos, rest_mesh, young, shear, thickness, poisson, 3);
            if (material_type == MaterialType::kStVK) {
                dir_energy_model = std::make_shared<StVKS1DirectorTanShellEnergy>(cur_mesh, rest_state);
            } else if (material_type == MaterialType::kNeoHookean) {
                dir_energy_model = std::make_shared<NeohookeanS1DirectorTanShellEnergy>(cur_mesh, rest_state);
            }
            break;
        }

        case SFFModelType::kS2Sin: {
            LibShell::MidedgeAngleGeneralSinFormulation::initializeExtraDOFs(edge_dofs, cur_mesh, rest_pos);
            extra_energy_terms = std::make_shared<LibShell::ExtraEnergyTermsGeneralSinFormulation>();
            extra_energy_terms->initialization(rest_pos, rest_mesh, young, shear, thickness, poisson, 3);
            if (material_type == MaterialType::kStVK) {
                dir_energy_model = std::make_shared<StVKS2DirectorSinShellEnergy>(cur_mesh, rest_state);
            } else if (material_type == MaterialType::kNeoHookean) {
                dir_energy_model = std::make_shared<NeohookeanS2DirectorSinShellEnergy>(cur_mesh, rest_state);
            }
            break;
        }
    }

    return {dir_energy_model, extra_energy_terms};
}

OptimizationLog processFullModelOneStep(const LibShell::MeshConnectivity& rest_mesh,
                                        const LibShell::MonolayerRestState& rest_state,
                                        const LibShell::MeshConnectivity& cur_mesh,
                                        const std::shared_ptr<ShellEnergy> stvk_dir_energy_model,
                                        const std::shared_ptr<LibShell::ExtraEnergyTermsBase> extra_energy_terms,
                                        Eigen::MatrixXd& cur_pos,
                                        Eigen::VectorXd& cur_edge_dofs,
                                        SFFModelType model_type) {
    std::unordered_map<int, double> cur_bnd_condition;
    cur_bnd_condition = build_boundary_fixed_vertex_map(cur_pos, cur_mesh);

    switch (model_type) {
        case SFFModelType::kS1Sin:
            std::cout << "============= Optimizing edge direction (S1 Sin) =========== " << std::endl;
            break;
        case SFFModelType::kS1Tan:
            std::cout << "============= Optimizing edge direction (S1 Tan) =========== " << std::endl;
            break;
        case SFFModelType::kS2Sin:
            std::cout << "============= Optimizing edge direction (S2 Sin) =========== " << std::endl;
            break;
    }
    OptimizationLog opt_log;
    optimizeFullDOFs(*stvk_dir_energy_model, rest_state.abars, cur_mesh, cur_pos, cur_edge_dofs, extra_energy_terms,
                     cur_bnd_condition.empty() ? nullptr : &cur_bnd_condition, nullptr, &opt_log);
    return opt_log;
}

void update_rendering(polyscope::SurfaceMesh* cur_surface_mesh,
                      polyscope::PointCloud* pt_mesh,
                      const std::shared_ptr<ShellEnergy> stvk_dir_energy_model,
                      const std::shared_ptr<LibShell::ExtraEnergyTermsBase> extra_energy_terms,
                      const LibShell::MeshConnectivity& cur_mesh,
                      const LibShell::MonolayerRestState& rest_state,
                      const Eigen::MatrixXd& cur_pos,
                      const Eigen::VectorXd& cur_edge_dofs,
                      SFFModelType model_type) {
    std::vector<Eigen::Vector3d> face_edge_midpts = {};
    for (int i = 0; i < cur_mesh.nFaces(); i++) {
        for (int j = 0; j < 3; j++) {
            int eid = cur_mesh.faceEdge(i, j);
            Eigen::Vector3d midpt =
                (cur_pos.row(cur_mesh.edgeVertex(eid, 0)) + cur_pos.row(cur_mesh.edgeVertex(eid, 1))) / 2.0;
            face_edge_midpts.push_back(midpt);
        }
    }
    pt_mesh->updatePointPositions(face_edge_midpts);
    cur_surface_mesh->updateVertexPositions(cur_pos);

    // draw the edge normals
    Eigen::VectorXd edge_dofs;
    std::string model_name;
    switch (model_type) {
        case SFFModelType::kS1Sin:
        case SFFModelType::kS1Tan: {
            edge_dofs.resize(2 * cur_mesh.nEdges());
            for (int i = 0; i < cur_mesh.nEdges(); i++) {
                edge_dofs(2 * i) = cur_edge_dofs(i);
                edge_dofs(2 * i + 1) = M_PI_2;
            }
            model_name = "S1 sin";
            break;
        }
        case SFFModelType::kS2Sin: {
            edge_dofs = cur_edge_dofs;
            model_name = "S2 sin";
            break;
        }
        default: {
            return;
        }
    }
    std::vector<Eigen::Vector3d> face_edge_normals = get_face_edge_normal_vectors(cur_pos, cur_mesh, edge_dofs);

    // draw energy terms
    std::vector<double> bending_scalars =
        stvk_dir_energy_model->elasticEnergyPerElement(cur_pos, cur_edge_dofs, false, true);
    std::vector<double> stretching_scalars =
        stvk_dir_energy_model->elasticEnergyPerElement(cur_pos, cur_edge_dofs, true, false);

    auto bending_plot = cur_surface_mesh->addFaceScalarQuantity("bending", bending_scalars);
    bending_plot->setMapRange({*std::min_element(bending_scalars.begin(), bending_scalars.end()),
                               *std::max_element(bending_scalars.begin(), bending_scalars.end())});

    auto stretching_plot = cur_surface_mesh->addFaceScalarQuantity("stretching", stretching_scalars);
    stretching_plot->setMapRange({*std::min_element(stretching_scalars.begin(), stretching_scalars.end()),
                                  *std::max_element(stretching_scalars.begin(), stretching_scalars.end())});

    Eigen::VectorXd bderiv;
    stvk_dir_energy_model->elasticEnergy(cur_pos, cur_edge_dofs, false, true, &bderiv, NULL);
    Eigen::MatrixXd bvertderiv(cur_pos.rows(), 3);
    for (int i = 0; i < cur_pos.rows(); i++) {
        for (int j = 0; j < 3; j++) {
            bvertderiv(i, j) = bderiv[3 * i + j];
        }
    }
    cur_surface_mesh->addVertexVectorQuantity("b. deriv", bvertderiv);

    std::vector<double> perp_scalars, III_scalars;
    for (int i = 0; i < cur_mesh.nFaces(); i++) {
        perp_scalars.push_back(extra_energy_terms->compute_vector_perp_tangent_energy_perface(
            cur_pos, edge_dofs, cur_mesh, rest_state.abars, i, nullptr, nullptr, false));
        III_scalars.push_back(extra_energy_terms->compute_thirdFundamentalForm_energy_perface(
            cur_pos, edge_dofs, cur_mesh, rest_state.abars, i, nullptr, nullptr, false));
    }
    auto scalar_plot = cur_surface_mesh->addFaceScalarQuantity("perp", perp_scalars);
    scalar_plot->setMapRange({*std::min_element(perp_scalars.begin(), perp_scalars.end()),
                              *std::max_element(perp_scalars.begin(), perp_scalars.end())});
    auto III_plot = cur_surface_mesh->addFaceScalarQuantity("III", III_scalars);
    III_plot->setMapRange({*std::min_element(III_scalars.begin(), III_scalars.end()),
                           *std::max_element(III_scalars.begin(), III_scalars.end())});

    auto vec_quantity = pt_mesh->addVectorQuantity(model_name + " Edge Normals", face_edge_normals);
    vec_quantity->setEnabled(true);
}

struct InputArgs {
    double thickness = 1.0;
    double triangle_area = 0.002;
    double poisson = 0.3;
    int sff_model = 0;  // 0 for s1 and 2 for s2
    int material = 1;   // 0 for StVK and 1 for NeoHookean
    bool with_gui = false;
    std::string output_dir = "";
};

SFFModelType parseModelType(int type) {
    if (type == 0)
        return SFFModelType::kS1Sin;
    else if (type == 1)
        return SFFModelType::kS1Tan;
    else if (type == 2)
        return SFFModelType::kS2Sin;
    else {
        assert(!"Illegal model type");
        exit(-1);
    }
}

int main(int argc, char* argv[]) {
    InputArgs args;
    CLI::App app{"Shell Energy Model: Punched Disk Demo"};
    app.add_option("-t,--thickness", args.thickness, "Thickness of the shell");
    app.add_option("-a,--triangle_area", args.triangle_area, "Relative triangle area of the mesh");
    app.add_option("-s,--sff_model", args.sff_model, "SFF model type");
    app.add_option("-m, --material", args.material, "Material type");
    app.add_option("-p, --poisson", args.poisson, "Poisson ratio of the material");
    app.add_option("-o, --output_dir", args.output_dir, "Output directory");
    app.add_flag("-g, --with_gui", args.with_gui, "Whether to show the GUI");
    CLI11_PARSE(app, argc, argv);

    double triangle_area = 0.002;
    if (args.triangle_area > 0) {
        triangle_area = args.triangle_area;
    }

    if (args.thickness > 0) {
        shell_thickness = args.thickness;
    }

    SFFModelType sff_type = parseModelType(args.sff_model);

    MaterialType material_type = args.material == 0 ? MaterialType::kStVK : MaterialType::kNeoHookean;

    double inner_radius = 0.1;
    double middle_radius = 0.4;
    double outer_radius = 0.5;

    // set up material parameters
    double young = 1e7;
    double poisson = 0.3;
    if (args.poisson > 0 && args.poisson < 1) {
        poisson = args.poisson;
    }
    double shear = young / (2.0 * (1.0 + poisson));

    Eigen::MatrixXd flatV, V3d, V3d_init;
    Eigen::MatrixXi F, F3d;

    makePunchedDisk(inner_radius, middle_radius, outer_radius, triangle_area, flatV, F);

    auto set_up_initial_positions = [&]() {
        V3d = flatV;
        F3d = F;

        for (int i = 0; i < flatV.rows(); i++) {
            double r = flatV.row(i).norm();
            // inner radius -> target_height
            // middle radius -> 0
            if (r <= inner_radius) {
                V3d(i, 2) = target_height;
            } else if (r >= middle_radius) {
                V3d(i, 2) = 0;
            } else {
                V3d(i, 2) = -target_height * (r - inner_radius) / (middle_radius - inner_radius) + target_height;
            }
        }

        V3d_init = V3d;
    };

    set_up_initial_positions();

    Eigen::VectorXd edge_dofs;
    LibShell::MonolayerRestState rest_state;
    LibShell::MeshConnectivity rest_mesh, cur_mesh;

    std::shared_ptr<LibShell::ExtraEnergyTermsBase> extra_energy_terms;
    std::shared_ptr<ShellEnergy> dir_energy_model;

    bool reinitialization = false;
    std::string model_name = "";
    std::string material_name = "";

    std::string prefix_name = args.output_dir;
    std::string gui_prefix_name = "";
    std::string results_dir = "";

    auto initialize_all = [&]() {
        rest_mesh = LibShell::MeshConnectivity(F);
        cur_mesh = LibShell::MeshConnectivity(F3d);
        std::tie(dir_energy_model, extra_energy_terms) =
            initialization(rest_mesh, flatV, cur_mesh, shell_thickness, young, poisson, edge_dofs, rest_state, sff_type,
                           material_type);
        reinitialization = false;
        V3d = V3d_init;

        switch (sff_type) {
            case SFFModelType::kS1Sin: {
                model_name = "S1 sin ";
                break;
            }
            case SFFModelType::kS1Tan: {
                model_name = "S1 tan ";
                break;
            }
            case SFFModelType::kS2Sin: {
                model_name = "S2 sin ";
                break;
            }
            default: {
                break;
            }
        }

        switch (material_type) {
            case MaterialType::kStVK: {
                material_name = "StVK";
                break;
            }
            case MaterialType::kNeoHookean: {
                material_name = "NeoHookean";
                break;
            }
            default: {
                break;
            }
        }
        gui_prefix_name = model_name + material_name + " ";
        if (!args.output_dir.empty()) {
            prefix_name = args.output_dir;
            if (!std::filesystem::exists(prefix_name)) {
                try {
                    std::filesystem::create_directories(prefix_name);
                } catch (const std::filesystem::filesystem_error& e) {
                    std::cerr << "Failed to create output directory: " << e.what() << std::endl;
                    std::cerr << "Using current directory instead." << std::endl;
                    prefix_name = ".";
                }
            }
            prefix_name +=
                "/" + material_name + "/V_" + std::to_string(flatV.rows()) + "_h_" + std::to_string(shell_thickness);
        } else {
            prefix_name =
                material_name + "/V_" + std::to_string(flatV.rows()) + "_h_" + std::to_string(shell_thickness);
        }

        prefix_name += "/" + model_name;

        fixed_verts.clear();
        std::vector<std::vector<int>> boundary_loops;
        igl::boundary_loop(F, boundary_loops);
        for (int i = 0; i < boundary_loops.size(); i++) {
            for (int j = 0; j < boundary_loops[i].size(); j++) {
                fixed_verts.push_back(boundary_loops[i][j]);
            }
        }

        constexpr double PI = 3.1415926535898;
        double abs_triangle_area = PI * outer_radius * outer_radius * triangle_area;
        double targetlength = 2.0 * std::sqrt(abs_triangle_area / std::sqrt(3.0));

        int middle_samples = std::max(3, int(2 * PI * middle_radius / targetlength));

        for (int i = 0; i < flatV.rows(); i++) {
            double r = flatV.row(i).norm();
            double middle_radius_ratio = std::cos(PI / middle_samples) * middle_radius;
            if (r >= middle_radius_ratio * (1 - 1e-5)) {
                fixed_verts.push_back(i);
            }
        }

        // make a directory for the results
        results_dir = prefix_name;
        if (!results_dir.empty()) {
            try {
                std::filesystem::create_directories(results_dir);
            } catch (const std::filesystem::filesystem_error& e) {
                std::cerr << "Failed to create directory: " << e.what() << std::endl;
                std::cerr << "Will use current directory instead." << std::endl;
                results_dir = ".";
            }
        }
    };

    if (!args.with_gui) {
        initialize_all();
        // save the initial configuration
        std::ofstream initial_config_file(results_dir + "/initial_config.txt");
        initial_config_file << "shell_thickness: " << shell_thickness << std::endl;
        initial_config_file << "triangle_area: " << triangle_area << std::endl;
        initial_config_file << "sff_type: ";
        switch (sff_type) {
            case SFFModelType::kS1Sin: {
                initial_config_file << "S1 sin";
                break;
            }
            case SFFModelType::kS1Tan: {
                initial_config_file << "S1 tan";
                break;
            }
            case SFFModelType::kS2Sin: {
                initial_config_file << "S2 sin";
                break;
            }
            default: {
                break;
            }
        }
        initial_config_file << std::endl;
        initial_config_file << "material_type: ";
        switch (material_type) {
            case MaterialType::kStVK: {
                initial_config_file << "StVK";
                break;
            }
            case MaterialType::kNeoHookean: {
                initial_config_file << "NeoHookean";
                break;
            }
            default: {
                break;
            }
        }
        initial_config_file << std::endl;
        initial_config_file << "young: " << young << std::endl;
        initial_config_file << "poisson: " << poisson << std::endl;
        initial_config_file.close();

        OptimizationLog opt_log = processFullModelOneStep(rest_mesh, rest_state, cur_mesh, dir_energy_model,
                                                          extra_energy_terms, V3d, edge_dofs, sff_type);
        igl::writeOBJ(results_dir + "/rest_mesh.obj", flatV, F);
        igl::writeOBJ(results_dir + "/converged_mesh.obj", V3d, F3d);

        // save edge dofs as a text file
        std::ofstream edge_dofs_file(results_dir + "/edge_dofs.txt");
        for (int i = 0; i < edge_dofs.size(); i++) {
            edge_dofs_file << edge_dofs(i) << std::endl;
        }
        edge_dofs_file.close();

        opt_log.save(results_dir + "/optimization_log.txt");
    } else {
        polyscope::init();
        polyscope::SurfaceMesh* surface_mesh = nullptr;
        polyscope::SurfaceMesh* init_surface_mesh = nullptr;
        polyscope::PointCloud *pt_mesh = nullptr, *init_pt_mesh = nullptr;

        auto initialize_rendering = [&]() {
            surface_mesh = polyscope::registerSurfaceMesh(gui_prefix_name + "Current mesh", V3d, F3d);
            surface_mesh->setEnabled(true);
            init_surface_mesh = polyscope::registerSurfaceMesh(gui_prefix_name + "Initial mesh", V3d_init, F3d);
            init_surface_mesh->setEnabled(false);

            auto flat_surface_mesh = polyscope::registerSurfaceMesh(gui_prefix_name + "Rest mesh", flatV, F);
            flat_surface_mesh->setEnabled(false);

            std::vector<Eigen::Vector3d> face_edge_midpts = {};
            for (int i = 0; i < cur_mesh.nFaces(); i++) {
                for (int j = 0; j < 3; j++) {
                    int eid = cur_mesh.faceEdge(i, j);
                    Eigen::Vector3d midpt =
                        (V3d.row(cur_mesh.edgeVertex(eid, 0)) + V3d.row(cur_mesh.edgeVertex(eid, 1))) / 2.0;
                    face_edge_midpts.push_back(midpt);
                }
            }

            pt_mesh = polyscope::registerPointCloud(gui_prefix_name + "Face edge midpoints", face_edge_midpts);
            pt_mesh->setEnabled(true);
            init_pt_mesh = polyscope::registerPointCloud(gui_prefix_name + "Initial Face edge midpoints", face_edge_midpts);
            init_pt_mesh->setEnabled(false);

            std::vector<Eigen::Vector3d> fixed_verts_pos;
            for (int i = 0; i < fixed_verts.size(); i++) {
                fixed_verts_pos.push_back(V3d.row(fixed_verts[i]));
            }
            auto fixed_pt_mesh = polyscope::registerPointCloud(gui_prefix_name + "Fixed verts", fixed_verts_pos);
            fixed_pt_mesh->setEnabled(false);
        };

        initialize_all();
        initialize_rendering();

        update_rendering(init_surface_mesh, init_pt_mesh, dir_energy_model, extra_energy_terms, cur_mesh, rest_state,
                         V3d, edge_dofs, sff_type);
        update_rendering(surface_mesh, pt_mesh, dir_energy_model, extra_energy_terms, cur_mesh, rest_state, V3d,
                         edge_dofs, sff_type);

        polyscope::state::userCallback = [&]() {
            if (ImGui::InputDouble("Triangle Area", &triangle_area)) {
                if (triangle_area <= 0) {
                    triangle_area = 0.02;
                }
                reinitialization = true;
            }

            if (ImGui::InputDouble("Thickness", &shell_thickness)) {
                if (shell_thickness <= 0) {
                    shell_thickness = 0.001;
                }
                reinitialization = true;
            }

            if (ImGui::InputDouble("Target Height", &target_height)) {
                if (target_height <= 0) {
                    target_height = 0.2;
                }
                reinitialization = true;
            }

            if (ImGui::InputDouble("Poisson", &poisson)) {
                if (poisson < 0) {
                    poisson = 0.3;
                }
                reinitialization = true;
            }

            if (ImGui::Combo("Material Type", &args.material, "StVK\0NeoHookean\0")) {
                material_type = args.material == 0 ? MaterialType::kStVK : MaterialType::kNeoHookean;
                initialize_all();
                initialize_rendering();

                update_rendering(init_surface_mesh, init_pt_mesh, dir_energy_model, extra_energy_terms, cur_mesh,
                                 rest_state, V3d, edge_dofs, sff_type);
            }

            if (ImGui::Combo("Bending Type", &args.sff_model, "S1 Sin\0S1 Tan\0S2 Sin\0")) {
                sff_type = parseModelType(args.sff_model);
                initialize_all();
                initialize_rendering();

                update_rendering(init_surface_mesh, init_pt_mesh, dir_energy_model, extra_energy_terms, cur_mesh,
                                 rest_state, V3d, edge_dofs, sff_type);
            }

            ImGui::Checkbox("Include III", &is_include_III);

            if (ImGui::Button("Remake Disk", ImVec2(-1, 0))) {
                makePunchedDisk(inner_radius, middle_radius, outer_radius, triangle_area, flatV, F);
                set_up_initial_positions();
                reinitialization = true;
                initialize_all();
                initialize_rendering();
                update_rendering(init_surface_mesh, init_pt_mesh, dir_energy_model, extra_energy_terms, cur_mesh,
                                 rest_state, V3d, edge_dofs, sff_type);
            }

            if (ImGui::InputInt("Number of Newton Iters", &num_iters)) {
                if (num_iters <= 0) {
                    num_iters = 200;
                }
            }
            if (ImGui::Button("Reset", ImVec2(-1, 0))) {
                V3d = V3d_init;
                edge_dofs = edge_dofs;
                update_rendering(surface_mesh, pt_mesh, dir_energy_model, extra_energy_terms, cur_mesh, rest_state, V3d,
                                 edge_dofs, sff_type);
            }

            if (ImGui::Button("Punch", ImVec2(-1, 0))) {
                if (reinitialization) {
                    initialize_all();
                }
                OptimizationLog opt_log = processFullModelOneStep(rest_mesh, rest_state, cur_mesh, dir_energy_model,
                                                                  extra_energy_terms, V3d, edge_dofs, sff_type);
                update_rendering(surface_mesh, pt_mesh, dir_energy_model, extra_energy_terms, cur_mesh, rest_state, V3d,
                                 edge_dofs, sff_type);

                igl::writeOBJ(results_dir + "/converged_mesh.obj", V3d, F3d);

                // save edge dofs as a text file
                std::ofstream edge_dofs_file(results_dir + "/edge_dofs.txt");
                for (int i = 0; i < edge_dofs.size(); i++) {
                    edge_dofs_file << edge_dofs(i) << std::endl;
                }
                edge_dofs_file.close();

                opt_log.save(results_dir + "/optimization_log.txt");
            }
        };

        polyscope::show();
    }
    return 0;
}
