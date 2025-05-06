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
#include <utility>  // For std::tie

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

MaterialType parseMaterialType(int type) {
    if (type == 0)
        return MaterialType::kStVK;
    else if (type == 1)
        return MaterialType::kNeoHookean;
}

bool readEdgeDofs(const std::string& filename, Eigen::VectorXd& edge_dofs) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open edge dofs file: " << filename << std::endl;
        return false;
    }
    std::string line;
    std::vector<double> edge_dofs_vec;
    while (std::getline(file, line)) {
        edge_dofs_vec.push_back(std::stod(line));
    }
    edge_dofs.resize(edge_dofs_vec.size());
    for (int i = 0; i < edge_dofs_vec.size(); i++) {
        edge_dofs(i) = edge_dofs_vec[i];
    }
    return true;
}

/*
=== Newton solver log:
- energy: 8.84755
- grad_norm: 1.5956e-08
- x_norm: 3.514
- f_norm: 8.84755
- newton_dec: 0.000173003
- line_search_step_size: 1
- terminate_iter: 16
- is_converged: 1
=== Total energy: 8.84755
- membrane energy: 8.52411
- bending energy: 6.21679e-06
- III energy: 0
- mag comp energy: 0
- direct perp energy: 0.323432
*/
bool readOptimizationLog(const std::string& filename, OptimizationLog& log) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open optimization log file: " << filename << std::endl;
        return false;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.find("=== Newton solver log:") != std::string::npos) {
            continue;
        } else if (line.find("- energy:") != std::string::npos) {
            log.log.energy = std::stod(line.substr(line.find(":") + 1));
        } else if (line.find("- grad_norm:") != std::string::npos) {
            log.log.grad_norm = std::stod(line.substr(line.find(":") + 1));
        } else if (line.find("- x_norm:") != std::string::npos) {
            log.log.x_norm = std::stod(line.substr(line.find(":") + 1));
        } else if (line.find("- f_norm:") != std::string::npos) {
            log.log.f_norm = std::stod(line.substr(line.find(":") + 1));
        } else if (line.find("- newton_dec:") != std::string::npos) {
            log.log.newton_dec = std::stod(line.substr(line.find(":") + 1));
        } else if (line.find("- line_search_step_size:") != std::string::npos) {
            log.log.line_search_step_size = std::stod(line.substr(line.find(":") + 1));
        } else if (line.find("- terminate_iter:") != std::string::npos) {
            log.log.terminate_iter = std::stoi(line.substr(line.find(":") + 1));
        } else if (line.find("- is_converged:") != std::string::npos) {
            log.log.is_converged = std::stoi(line.substr(line.find(":") + 1));
        } else if (line.find("=== Total energy:") != std::string::npos) {
            log.total_energy = std::stod(line.substr(line.find(":") + 1));
        } else if (line.find("- membrane energy:") != std::string::npos) {
            log.membrane_energy = std::stod(line.substr(line.find(":") + 1));
        } else if (line.find("- bending energy:") != std::string::npos) {
            log.bending_energy = std::stod(line.substr(line.find(":") + 1));
        } else if (line.find("- III energy:") != std::string::npos) {
            log.III_energy = std::stod(line.substr(line.find(":") + 1));
        } else if (line.find("- mag comp energy:") != std::string::npos) {
            log.mag_comp_energy = std::stod(line.substr(line.find(":") + 1));
        } else if (line.find("- direct perp energy:") != std::string::npos) {
            log.direct_perp_energy = std::stod(line.substr(line.find(":") + 1));
        }
    }

    return true;
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

struct InputArgs {
    std::string results_dir;
};

int main(int argc, char* argv[]) {
    InputArgs args;
    CLI::App app{"Shell Model Visualization"};
    app.add_option("-r, --results_dir", args.results_dir, "Results directory");
    CLI11_PARSE(app, argc, argv);

    std::string config_file_path = args.results_dir + "/initial_config.txt";
    std::string optimization_log_file_path = args.results_dir + "/optimization_log.txt";
    std::string rest_mesh_file_path = args.results_dir + "/rest_mesh.obj";
    std::string converged_mesh_file_path = args.results_dir + "/converged_mesh.obj";
    std::string edge_dofs_file_path = args.results_dir + "/edge_dofs.txt";

    std::ifstream config_file(config_file_path);
    if (!config_file.is_open()) {
        std::cerr << "Failed to open config file: " << config_file_path << std::endl;
        return 1;
    }
    /*
    Config file format (in txt):
    shell_thickness: 0.001
    triangle_area: 0.002
    sff_type: S1 sin
    material_type: StVK
    young: 1e7
    poisson: 0.3
    */
    double shell_thickness = 1e-3;
    double triangle_area = 0.002;
    SFFModelType sff_type = SFFModelType::kS1Sin;
    MaterialType material_type = MaterialType::kStVK;
    double young = 1e7;
    double poisson = 0.3;

    std::string line;
    while (std::getline(config_file, line)) {
        std::istringstream iss(line);
        std::string key;

        // First extract the key (everything before the colon)
        std::getline(iss, key, ':');
        key = key.substr(0, key.find_last_not_of(" \t") + 1);  // Trim trailing whitespace

        // Then extract the value (everything after the colon)
        std::string value;
        std::getline(iss, value);
        value = value.substr(value.find_first_not_of(" \t"));  // Trim leading whitespace

        std::cout << "key: " << key << ", value: " << value << std::endl;
        // print the line
        std::cout << "line: " << line << std::endl;

        if (key == "shell_thickness") {
            shell_thickness = std::stod(value);
        } else if (key == "triangle_area") {
            triangle_area = std::stod(value);
        } else if (key == "sff_type") {
            if (value == "S1 sin") {
                sff_type = SFFModelType::kS1Sin;
            } else if (value == "S1 tan") {
                sff_type = SFFModelType::kS1Tan;
            } else if (value == "S2 sin") {
                sff_type = SFFModelType::kS2Sin;
            }
        } else if (key == "material_type") {
            if (value == "StVK") {
                material_type = MaterialType::kStVK;
            } else if (value == "NeoHookean") {
                material_type = MaterialType::kNeoHookean;
            }
        } else if (key == "young") {
            young = std::stod(value);
        } else if (key == "poisson") {
            poisson = std::stod(value);
        }
    }
    double shear = young / (2.0 * (1.0 + poisson));

    Eigen::MatrixXd flatV, V3d;
    Eigen::MatrixXi F, F3d;

    if (!igl::readOBJ(rest_mesh_file_path, flatV, F)) {
        std::cerr << "Failed to read rest mesh file: " << rest_mesh_file_path << std::endl;
        return 1;
    }

    if (!igl::readOBJ(converged_mesh_file_path, V3d, F3d)) {
        std::cerr << "Failed to read converged mesh file: " << converged_mesh_file_path << std::endl;
        return 1;
    }

    Eigen::VectorXd edge_dofs;
    if (!readEdgeDofs(edge_dofs_file_path, edge_dofs)) {
        std::cerr << "Failed to read edge dofs file: " << edge_dofs_file_path << std::endl;
        return 1;
    }

    OptimizationLog opt_log;
    if (!readOptimizationLog(optimization_log_file_path, opt_log)) {
        std::cerr << "Failed to read optimization log file: " << optimization_log_file_path << std::endl;
        return 1;
    }

    LibShell::MonolayerRestState rest_state;
    LibShell::MeshConnectivity rest_mesh, cur_mesh;

    std::shared_ptr<LibShell::ExtraEnergyTermsBase> extra_energy_terms;
    std::shared_ptr<ShellEnergy> dir_energy_model;

    bool reinitialization = false;
    std::string model_name = "";
    std::string material_name = "";
    std::string results_dir = "";
    std::string prefix_name = "";

    auto initialize_all = [&]() {
        rest_mesh = LibShell::MeshConnectivity(F);
        cur_mesh = LibShell::MeshConnectivity(F3d);
        auto result = initialization(rest_mesh, flatV, cur_mesh, shell_thickness, young, poisson, edge_dofs, rest_state,
                                     sff_type, material_type);
        dir_energy_model = result.first;
        extra_energy_terms = result.second;
        reinitialization = false;

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
        prefix_name = model_name + " " + material_name;
    };

    polyscope::init();
    polyscope::SurfaceMesh* surface_mesh = nullptr;
    polyscope::SurfaceMesh* init_surface_mesh = nullptr;
    polyscope::PointCloud *pt_mesh = nullptr, *init_pt_mesh = nullptr;

    auto initialize_rendering = [&]() {
        surface_mesh = polyscope::registerSurfaceMesh("Current mesh", V3d, F3d);
        surface_mesh->setEnabled(true);

        std::vector<Eigen::Vector3d> face_edge_midpts = {};
        for (int i = 0; i < cur_mesh.nFaces(); i++) {
            for (int j = 0; j < 3; j++) {
                int eid = cur_mesh.faceEdge(i, j);
                Eigen::Vector3d midpt =
                    (V3d.row(cur_mesh.edgeVertex(eid, 0)) + V3d.row(cur_mesh.edgeVertex(eid, 1))) / 2.0;
                face_edge_midpts.push_back(midpt);
            }
        }

        pt_mesh = polyscope::registerPointCloud(prefix_name + "Face edge midpoints", face_edge_midpts);
        pt_mesh->setEnabled(true);
    };

    initialize_all();
    initialize_rendering();

    update_rendering(surface_mesh, pt_mesh, dir_energy_model, extra_energy_terms, cur_mesh, rest_state, V3d, edge_dofs,
                     sff_type);

    polyscope::state::userCallback = [&]() {
        // change this to text (so that it can't be edited)
        if (ImGui::CollapsingHeader("Configuration")) {
            ImGui::Text("Triangle Area: %f", triangle_area);
            ImGui::Text("Thickness: %f", shell_thickness);
            ImGui::Text("Poisson: %f", poisson);
            ImGui::Text("Material Type: %s", material_name.c_str());
            ImGui::Text("Bending Type: %s", model_name.c_str());
        }
        // add the extra section for the energy
        if (ImGui::CollapsingHeader("Energy")) {
            ImGui::Text("Energy: %f", opt_log.total_energy);
            ImGui::Text("Membrane Energy: %f", opt_log.membrane_energy);
            ImGui::Text("Bending Energy: %f", opt_log.bending_energy);
            ImGui::Text("III Energy: %f", opt_log.III_energy);
            ImGui::Text("Mag Comp Energy: %f", opt_log.mag_comp_energy);
            ImGui::Text("Direct Perp Energy: %f", opt_log.direct_perp_energy);
        }
        if (ImGui::CollapsingHeader("Newton Solver Log")) {
            ImGui::Text("Energy: %f", opt_log.log.energy);
            ImGui::Text("Grad Norm: %f", opt_log.log.grad_norm);
            ImGui::Text("X Norm: %f", opt_log.log.x_norm);
            ImGui::Text("F Norm: %f", opt_log.log.f_norm);
            ImGui::Text("Newton Dec: %f", opt_log.log.newton_dec);
            ImGui::Text("Line Search Step Size: %f", opt_log.log.line_search_step_size);
            ImGui::Text("Terminate Iter: %d", opt_log.log.terminate_iter);
            ImGui::Text("Is Converged: %d", opt_log.log.is_converged);
        }

        if (ImGui::Button("Draw Energies", ImVec2(-1, 0))) {
            update_rendering(surface_mesh, pt_mesh, dir_energy_model, extra_energy_terms, cur_mesh, rest_state, V3d,
                             edge_dofs, sff_type);
        }
    };

    polyscope::show();
    return 0;
}
