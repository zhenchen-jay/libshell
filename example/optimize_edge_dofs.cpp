#include "StaticSolve.h"
#include "../Optimization/include/NewtonDescent.h"

#include "../include/MeshConnectivity.h"
#include "../include/ElasticShell.h"
#include "../include/MidedgeAngleTanFormulation.h"
#include "../include/MidedgeAngleSinFormulation.h"
#include "../include/MidedgeAverageFormulation.h"
#include "../include/MidedgeAngleThetaFormulation.h"
#include "../include/StVKMaterial.h"
#include "../include/TensionFieldStVKMaterial.h"
#include "../include/NeoHookeanMaterial.h"
#include "../include/RestState.h"

#include <polyscope/surface_mesh.h>
#include <polyscope/point_cloud.h>

#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <igl/triangle/triangulate.h>
#include <igl/boundary_loop.h>

#include <unordered_set>
#include <memory>
#include <filesystem>

#include <CLI/CLI.hpp>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/async.h>
#include <chrono>

int num_steps;
double grad_tol;
double f_tol;
double x_tol;
bool is_swap;

double young;
double thickness;
double poisson;
int matid;
int sffid;
int proj_type;

Eigen::MatrixXd cur_pos;
LibShell::MeshConnectivity mesh;

std::string output_folder = "";

void lame_parameters(double& alpha, double& beta) {
    alpha = young * poisson / (1.0 - poisson * poisson);
    beta = young / 2.0 / (1.0 + poisson);
}

template <class SFF>
double run_simulation(const LibShell::MeshConnectivity& mesh,
                    const Eigen::MatrixXd& rest_pos,
                    Eigen::MatrixXd& cur_pos,
                    const std::unordered_set<int>* fixed_verts,
                    double thickness,
                    double lame_alpha,
                    double lame_beta,
                    int matid,
                    int proj_type) {
    // initialize default edge DOFs (edge director angles)
    Eigen::VectorXd init_edge_DOFs;
    SFF::initializeExtraDOFs(init_edge_DOFs, mesh, cur_pos);
    init_edge_DOFs.setZero();

    // initialize the rest geometry of the shell
    LibShell::MonolayerRestState rest_state;

    // set uniform thicknesses
    rest_state.thicknesses.resize(mesh.nFaces(), thickness);

    // initialize first fundamental forms to those of input mesh
    LibShell::ElasticShell<SFF>::firstFundamentalForms(mesh, rest_pos, rest_state.abars);

    // initialize second fundamental forms to those of input mesh
    rest_state.bbars.resize(mesh.nFaces());
    for (int i = 0; i < mesh.nFaces(); i++) {
        rest_state.bbars[i] = SFF::secondFundamentalForm(mesh, rest_pos, init_edge_DOFs, i, nullptr, nullptr);
        rest_state.bbars[i].setZero();
    }

    rest_state.lameAlpha.resize(mesh.nFaces(), lame_alpha);
    rest_state.lameBeta.resize(mesh.nFaces(), lame_beta);

    std::shared_ptr<LibShell::MaterialModel<SFF>> mat;
    switch (matid) {
        case 0:
            mat = std::make_shared<LibShell::NeoHookeanMaterial<SFF>>();
            break;
        case 1:
            mat = std::make_shared<LibShell::StVKMaterial<SFF>>();
            break;
        case 2:
            mat = std::make_shared<LibShell::TensionFieldStVKMaterial<SFF>>();
            break;
        default:
            assert(false);
    }

    // projection matrix
    Eigen::SparseMatrix<double> P;
    std::vector<Eigen::Triplet<double>> Pcoeffs;
    int nedges = mesh.nEdges();
    int nedgedofs = SFF::numExtraDOFs;
    // we only allow fixed vertices in the current implementation
    Eigen::VectorXd fixed_dofs(3 * cur_pos.rows());
    fixed_dofs.setZero();
    int nfree = 0;
    for (int i = 0; i < cur_pos.rows(); i++) {
        if (!fixed_verts || !fixed_verts->count(i)) {
            Pcoeffs.push_back({nfree, 3 * i, 1.0});
            Pcoeffs.push_back({nfree + 1, 3 * i + 1, 1.0});
            Pcoeffs.push_back({nfree + 2, 3 * i + 2, 1.0});
            nfree += 3;
        } else {
            fixed_dofs.segment<3>(3 * i) = cur_pos.row(i).transpose();
        }
    }
    for (int i = 0; i < nedges * nedgedofs; i++) {
        Pcoeffs.push_back(Eigen::Triplet<double>(nfree, 3 * cur_pos.rows() + i, 1.0));
        nfree++;
    }

    P.resize(nfree, 3 * cur_pos.rows() + nedges * nedgedofs);
    P.setFromTriplets(Pcoeffs.begin(), Pcoeffs.end());

    int totalDOFs = 3 * cur_pos.rows() + nedges * nedgedofs;

    // project the current position
    auto pos_edgedofs_to_variable = [&](const Eigen::MatrixXd& pos, const Eigen::VectorXd& edge_DOFs) {
        Eigen::VectorXd var(nfree);
        int n = 0;
        for (int i = 0; i < pos.rows(); i++) {
            if (!fixed_verts || !fixed_verts->count(i)) {
                var.segment<3>(n) = pos.row(i).transpose();
                n += 3;
            }
        }
        var.tail(nedges * nedgedofs) = edge_DOFs;
        return var;
    };

    auto variable_to_pos_edgedofs = [&](const Eigen::VectorXd& var) {
        Eigen::MatrixXd pos(cur_pos.rows(), 3);
        int n = 0;
        for (int i = 0; i < cur_pos.rows(); i++) {
            if (!fixed_verts || !fixed_verts->count(i)) {
                pos.row(i) = var.segment<3>(n).transpose();
                n += 3;
            } else {
                pos.row(i) = fixed_dofs.segment<3>(3 * i).transpose();
            }
        }
        Eigen::VectorXd edge_DOFs = var.tail(nedges * nedgedofs);
        return std::pair<Eigen::MatrixXd, Eigen::VectorXd>{pos, edge_DOFs};
    };

    // energy, gradient, and hessian
    auto obj_func = [&](const Eigen::VectorXd& var, Eigen::VectorXd* grad, Eigen::SparseMatrix<double>* hessian,
                        bool psd_proj) {
        Eigen::MatrixXd pos;
        Eigen::VectorXd edge_DOFs;
        std::vector<Eigen::Triplet<double>> hessian_triplets;
        std::tie(pos, edge_DOFs) = variable_to_pos_edgedofs(var);

        double energy =
            LibShell::ElasticShell<SFF>::elasticEnergy(mesh, pos, edge_DOFs, *mat, rest_state, psd_proj ? proj_type : 0,
                                                       grad, hessian ? &hessian_triplets : nullptr);

        if (grad) {
            if (fixed_verts) {
                *grad = P * (*grad);
            }
        }

        if (hessian) {
            hessian->resize(totalDOFs, totalDOFs);
            hessian->setFromTriplets(hessian_triplets.begin(), hessian_triplets.end());
            if (fixed_verts) {
                *hessian = P * (*hessian) * P.transpose();
            }
        }

        return energy;
    };

    auto find_max_step = [&](const Eigen::VectorXd& x, const Eigen::VectorXd& dir) { return 1.0; };

    Eigen::VectorXd x0 = pos_edgedofs_to_variable(cur_pos, init_edge_DOFs);

    if (output_folder != "") {
        auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(output_folder + "/log.txt", true);
        spdlog::flush_every(std::chrono::seconds(1));

        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();

        auto multi_sink_logger =
            std::make_shared<spdlog::logger>("Newton Solver", spdlog::sinks_init_list{file_sink, console_sink});

        spdlog::set_default_logger(multi_sink_logger);
    }

    double init_energy = obj_func(x0, nullptr, nullptr, false);

    OptSolver::TestFuncGradHessian(obj_func, x0);

    OptSolver::NewtonSolver(obj_func, find_max_step, x0, num_steps, grad_tol, x_tol, f_tol, proj_type != 0, true,
                            is_swap);

    std::tie(cur_pos, init_edge_DOFs) = variable_to_pos_edgedofs(x0);

    double energy = obj_func(x0, nullptr, nullptr, false);

    std::cout << "Edge Dofs: " << init_edge_DOFs.transpose() << std::endl;

    spdlog::info("Initial Energy: {}, Optimized Energy: {}", init_energy, energy);

    return obj_func(x0, nullptr, nullptr, false);
}

static void gererated_foleded_mesh(int N, int M, double fold_theta, Eigen::MatrixXd& rest_V, Eigen::MatrixXi& rest_F, Eigen::MatrixXd& fold_V) {
    double constexpr R = 1;
    double constexpr H = 5;

    std::vector<Eigen::Vector3d> rest_pos, fold_pos;
    std::vector<Eigen::Vector3i> rest_faces;

    for (int i = 0; i <= 2 * N; i++) {
        for (int j = 0; j <= M; j++) {
            double x = (i - N) * M_PI * R / N;
            double y = j * H / M;
            rest_pos.push_back(Eigen::Vector3d{x, y, 0});

            if (i <= N) {
                fold_pos.push_back(Eigen::Vector3d(x, y, 0));
            } else {
                fold_pos.push_back(Eigen::Vector3d(x * std::cos(fold_theta), y, x * std::sin(fold_theta)));
            }

            // T theta = i * M_PI / N;
            // cylinder_pos.push_back(Vector<T, dim>{-R * std::sin(theta), y, R + R * std::cos(theta)});
        }
    }

    for (int i = 0; i < 2 * N; i++) {
        for (int j = 0; j < M; j++) {
            int k = i * (M + 1) + j;
            rest_faces.push_back(Eigen::Vector3i{k, k + 1, k + M + 1});
            rest_faces.push_back(Eigen::Vector3i{k + 1, k + M + 2, k + M + 1});
        }
    }

    rest_V.resize(rest_pos.size(), 3);
    for (int vid = 0; vid < rest_pos.size(); vid++) {
        rest_V.row(vid) = rest_pos[vid].transpose();
    }

    fold_V.resize(fold_pos.size(), 3);
    for (int vid = 0; vid < fold_pos.size(); vid++) {
        fold_V.row(vid) = fold_pos[vid].transpose();
    }

    rest_F.resize(rest_faces.size(), 3);
    for (int fid = 0; fid < rest_faces.size(); fid++) {
        rest_F.row(fid) = rest_faces[fid].transpose();
    } 
}

int main(int argc, char* argv[]) {
    CLI::App app("Static Simulation for a Stretched Sheet");
    double triangle_area;
    bool no_gui;
    int N = 50;
    int M = 50;


    // optimization parameters
    app.add_option("--num-steps", num_steps, "Number of iteration")->default_val(30);
    app.add_option("--grad-tol", grad_tol, "Gradient tolerance")->default_val(1e-6);
    app.add_option("--f-tol", f_tol, "Function tolerance")->default_val(0);
    app.add_option("--x-tol", x_tol, "Variable tolerance")->default_val(0);

    // material parameters
    app.add_option("--young", young, "Young's Modulus")->default_val(1e9);
    app.add_option("--thickness", thickness, "Thickness")->default_val(1e-4);
    app.add_option("--poisson", poisson, "Poisson's Ratio")->default_val(0.5);
    app.add_option("--material", matid, "Material Model")->default_val(1);
    app.add_option("--sff", sffid,
                   "Second Fundamental Form Formula, 0: midedge tan, 1: midedge sin, 2: midedge average, 3: midedge theta")
        ->default_val(2);
    app.add_option("--projection", proj_type, "Hessian Projection Type, 0 : no projection, 1: max(H, 0), 2: Abs(H)")
        ->default_val(1);
    app.add_flag("--swap", is_swap, "Swap to Actual Hessian when close to optimum");


    // sampling parameters
    app.add_option("-N", N, "Sampling points in x direction")->default_val(1);
    app.add_option("-M", M, "Sampling points in y direction")->default_val(1);

    // folded angle
    double fold_theta;
    app.add_option("--fold-theta", fold_theta, "Folded angle")->default_val(0.9999 * M_PI);

    app.add_option("ouput,-o,--output", output_folder, "Output folder");
    app.add_flag("--no-gui", no_gui, "Without gui")->default_val(false);
    CLI11_PARSE(app, argc, argv);

    // make output folder
    if (output_folder != "") {
        std::filesystem::create_directories(output_folder);
    }

    // generate mesh
    Eigen::MatrixXd orig_V, rest_V;
    Eigen::MatrixXi F;

    gererated_foleded_mesh(N, M, fold_theta, rest_V, F, orig_V);

    std::unordered_set<int> fixed_verts;
    for (int i = 0; i < rest_V.rows(); i++) {
        fixed_verts.insert(i);
    }

    // set up mesh connectivity
    mesh = LibShell::MeshConnectivity(F);

    // initial position
    cur_pos= orig_V;

    double energy = 0;

    if (no_gui) {
        double lame_alpha, lame_beta;
        lame_parameters(lame_alpha, lame_beta);

        switch (sffid) {
            case 0:
                energy = run_simulation<LibShell::MidedgeAngleTanFormulation>(
                    mesh, rest_V, cur_pos, &fixed_verts, thickness, lame_alpha, lame_beta, matid, proj_type);
                break;
            case 1:
                energy = run_simulation<LibShell::MidedgeAngleSinFormulation>(
                    mesh, rest_V, cur_pos, &fixed_verts, thickness, lame_alpha, lame_beta, matid, proj_type);
                break;
            case 2:
                energy = run_simulation<LibShell::MidedgeAverageFormulation>(
                    mesh, rest_V, cur_pos, &fixed_verts, thickness, lame_alpha, lame_beta, matid, proj_type);
                break;
            case 3:
                energy = run_simulation<LibShell::MidedgeAngleThetaFormulation>(
                    mesh, rest_V, cur_pos, &fixed_verts, thickness, lame_alpha, lame_beta, matid, proj_type);
                break;
            default:
                assert(false);
        }
        if (output_folder != "") {
            igl::writeOBJ(output_folder + "/rest.obj", rest_V, F);
            igl::writeOBJ(output_folder + "/orig.obj", orig_V, F);
            igl::writeOBJ(output_folder + "/deformed.obj", cur_pos, F);
        }
        return EXIT_SUCCESS;
    }

    polyscope::init();

    // Register a surface mesh structure
    auto surface_mesh = polyscope::registerSurfaceMesh("Rest mesh", rest_V, F);
    surface_mesh->setEnabled(false);

    auto cur_surface_mesh = polyscope::registerSurfaceMesh("Current mesh", cur_pos, F);

    if (output_folder != "") {
        igl::writeOBJ(output_folder + "/rest.obj", rest_V, F);
        igl::writeOBJ(output_folder + "/orig.obj", orig_V, F);
    }

    polyscope::state::userCallback = [&]() {
        if (ImGui::Button("Reset", ImVec2(-1, 0))) {
            cur_pos = orig_V;
            cur_surface_mesh->updateVertexPositions(cur_pos);
        }

        if (ImGui::CollapsingHeader("Parameters", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::InputDouble("Thickness", &thickness);
            ImGui::InputDouble("Poisson's Ration", &poisson);
            ImGui::Combo("Material Model", &matid, "NeoHookean\0StVK\0\0");
            ImGui::Combo("Second Fundamental Form", &sffid, "TanTheta\0SinTheta\0Average\0Theta\0\0");
        }

        if(ImGui::CollapsingHeader("Mesh Parameters", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::InputInt("N", &N);
            ImGui::InputInt("M", &M);
            ImGui::InputDouble("Folded Angle", &fold_theta);

            if(ImGui::Button("Regenerate Mesh")) {
                gererated_foleded_mesh(N, M, fold_theta, rest_V, F, orig_V);

                // get the left and right boundary vertices
                fixed_verts.clear();

                for (int i = 0; i < rest_V.rows(); i++) {
                    fixed_verts.insert(i);
                }

                // set up mesh connectivity
                mesh = LibShell::MeshConnectivity(F);

                // initial position
                cur_pos = orig_V;

                surface_mesh = polyscope::registerSurfaceMesh("Rest mesh", rest_V, F);
                surface_mesh->setEnabled(false);
                cur_surface_mesh = polyscope::registerSurfaceMesh("Current mesh", cur_pos, F);
            }
        }

        if (ImGui::CollapsingHeader("Optimization", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Combo("Hessian Projection", &proj_type, "No Projection\0Max Zero\0Abs\0\0");
            ImGui::InputInt("Num Steps", &num_steps);
            ImGui::InputDouble("Gradient Tol", &grad_tol);
            ImGui::InputDouble("Function Tol", &f_tol);
            ImGui::InputDouble("Variable Tol", &x_tol);
            ImGui::Checkbox("Swap to Actual Hessian when close to optimum", &is_swap);

            if (ImGui::Button("Optimize Some Step", ImVec2(-1, 0))) {
                double lame_alpha, lame_beta;
                lame_parameters(lame_alpha, lame_beta);

                switch (sffid) {
                    case 0:
                        energy = run_simulation<LibShell::MidedgeAngleTanFormulation>(
                            mesh, rest_V, cur_pos, &fixed_verts, thickness, lame_alpha, lame_beta, matid, proj_type);
                        break;
                    case 1:
                        energy = run_simulation<LibShell::MidedgeAngleSinFormulation>(
                            mesh, rest_V, cur_pos, &fixed_verts, thickness, lame_alpha, lame_beta, matid, proj_type);
                        break;
                    case 2:
                        energy = run_simulation<LibShell::MidedgeAverageFormulation>(
                            mesh, rest_V, cur_pos, &fixed_verts, thickness, lame_alpha, lame_beta, matid, proj_type);
                        break;
                    case 3:
                        energy = run_simulation<LibShell::MidedgeAngleThetaFormulation>(
                            mesh, rest_V, cur_pos, &fixed_verts, thickness, lame_alpha, lame_beta, matid, proj_type);
                        break;
                    default: 
                        assert(false);
                }
                cur_surface_mesh->updateVertexPositions(cur_pos);
                if (output_folder != "") {
                    igl::writeOBJ(output_folder + "/deformed.obj", cur_pos, F);
                }
            }
        }
    };

    // View the point cloud and mesh we just registered in the 3D UI
    polyscope::show();
}