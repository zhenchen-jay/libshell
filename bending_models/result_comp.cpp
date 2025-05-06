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
#include <igl/doublearea.h>
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

struct Config {
    double shell_thickness;
    double triangle_area;
    SFFModelType sff_type;
    MaterialType material_type;
    double young;
    double poisson;
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

/*
Config file format (in txt):
shell_thickness: 0.001
triangle_area: 0.002
sff_type: S1 sin
material_type: StVK
*/
bool readConfig(const std::string& filename, Config& config) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open config file: " << filename << std::endl;
        return false;
    }
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string key;

        // First extract the key (everything before the colon)
        std::getline(iss, key, ':');
        key = key.substr(0, key.find_last_not_of(" \t") + 1);  // Trim trailing whitespace

        // Then extract the value (everything after the colon)
        std::string value;
        std::getline(iss, value);
        value = value.substr(value.find_first_not_of(" \t"));  // Trim leading whitespace

        if (key == "shell_thickness") {
            config.shell_thickness = std::stod(value);
        } else if (key == "triangle_area") {
            config.triangle_area = std::stod(value);
        } else if (key == "sff_type") {
            if (value == "S1 sin") {
                config.sff_type = SFFModelType::kS1Sin;
            } else if (value == "S1 tan") {
                config.sff_type = SFFModelType::kS1Tan;
            } else if (value == "S2 sin") {
                config.sff_type = SFFModelType::kS2Sin;
            }
        } else if (key == "material_type") {
            if (value == "StVK") {
                config.material_type = MaterialType::kStVK;
            } else if (value == "NeoHookean") {
                config.material_type = MaterialType::kNeoHookean;
            }
        } else if (key == "young") {
            config.young = std::stod(value);
        } else if (key == "poisson") {
            config.poisson = std::stod(value);
        }
    }
    return true;
}

struct CompResult {
    double surface_dist = 0;
    double edge_vec_dist = 0;
};

/*
Compute the distance between two meshes (V1, F) and (V2, F) in Finite element space (linear elements). Each vertex
distance is computed by the Euclidean distance between V1 and V2 times by the vertex area in the rest mesh.
*/
double computeSurfaceDist(const Eigen::MatrixXd& V1,
                          const Eigen::MatrixXd& V2,
                          const Eigen::MatrixXd& restV,
                          const Eigen::MatrixXi& F,
                          Eigen::VectorXd& dist) {
    // sanity check
    if (V1.rows() != V2.rows()) {
        std::cerr << "V1 and V2 must have the same number of rows" << std::endl;
        exit(1);
    }
    if (V1.cols() != 3 || V2.cols() != 3 || restV.cols() != 3) {
        std::cerr << "V1, V2, and restV must have 3 columns" << std::endl;
        exit(1);
    }
    // compute the area of each vertex in the rest mesh
    Eigen::VectorXd double_area;
    igl::doublearea(restV, F, double_area);
    Eigen::VectorXd vertex_area(restV.rows());
    dist.resize(restV.rows());
    dist.setZero();
    for (size_t fid = 0; fid < F.rows(); fid++) {
        for (int j = 0; j < 3; j++) {
            int vid = F(fid, j);
            vertex_area(vid) += double_area(fid) / 6.0;
            dist(vid) += (V1.row(vid) - V2.row(vid)).norm() * double_area(fid) / 6.0;
        }
    }
    // sum up the distance
    return dist.sum();
}

/*
Compute the distance between two face edge vectors (face_edge_vecs1, F) and (face_edge_vecs2, F) in Finite element space
(linear elements). Each edge vector distance is computed by the norm of the difference between the edge vectors in the
two meshes timed by face edge area.
*/
double computeEdgeVecDist(const Eigen::MatrixXd& rest_V,
                          const Eigen::MatrixXi& F,
                          const std::vector<Eigen::Vector3d>& face_edge_vecs1,
                          const std::vector<Eigen::Vector3d>& face_edge_vecs2,
                          Eigen::VectorXd& dist) {
    // sanity check
    if (face_edge_vecs1.size() != 3 * F.rows() || face_edge_vecs2.size() != 3 * F.rows()) {
        std::cerr << "face_edge_vecs1 and face_edge_vecs2 must have the same size as F" << std::endl;
        exit(1);
    }

    LibShell::MeshConnectivity rest_mesh(F);
    Eigen::VectorXd double_area;
    igl::doublearea(rest_V, F, double_area);
    dist.resize(3 * F.rows());
    dist.setZero();

    for (size_t fid = 0; fid < F.rows(); fid++) {
        double face_edge_area = double_area(fid) / 6.0;
        for (int j = 0; j < 3; j++) {
            int eid = rest_mesh.faceEdge(fid, j);
            dist(3 * fid + j) += (face_edge_vecs1[3 * fid + j] - face_edge_vecs2[3 * fid + j]).norm() * face_edge_area;
        }
    }
    return dist.sum();
}

struct InputArgs {
    std::string results_dir;
    bool with_gui = false;
};

int main(int argc, char* argv[]) {
    InputArgs args;
    CLI::App app{"Shell Model Comparison"};
    app.add_option("-r, --results_dir", args.results_dir, "Results directory");
    app.add_flag("-g, --with_gui", args.with_gui, "Whether to show the gui");
    CLI11_PARSE(app, argc, argv);

    // under the results dir, there are two sub folders: S1 sin and S2 sin
    // under each sub folder, there are four files:
    // 1. initial_config.txt
    // 2. optimization_log.txt
    // 3. rest_mesh.obj
    // 4. converged_mesh.obj
    // 5. edge_dofs.txt
    // we want to compare the energy of the two models
    // We need read the current mesh and current edge dofs from the each sub folder, and compare the distance of the two
    // meshes

    std::vector<std::string> model_names = {"S1 sin ", "S2 sin "};

    // the first index is for S1 sin, the second index is for S2 sin
    std::vector<Eigen::MatrixXd> flatV(2), V3d(2);
    std::vector<Eigen::MatrixXi> F(2), F3d(2);
    std::vector<Eigen::VectorXd> edge_dofs(2);
    std::vector<Config> configs(2);
    std::vector<OptimizationLog> opt_logs(2);
    std::vector<std::vector<Eigen::Vector3d>> face_edge_vecs(2);

    for (int i = 0; i < 2; i++) {
        auto& model_name = model_names[i];
        std::string config_file_path = args.results_dir + "/" + model_name + "/initial_config.txt";
        std::string optimization_log_file_path = args.results_dir + "/" + model_name + "/optimization_log.txt";
        std::string rest_mesh_file_path = args.results_dir + "/" + model_name + "/rest_mesh.obj";
        std::string converged_mesh_file_path = args.results_dir + "/" + model_name + "/converged_mesh.obj";
        std::string edge_dofs_file_path = args.results_dir + "/" + model_name + "/edge_dofs.txt";
        if (!readConfig(config_file_path, configs[i])) {
            std::cerr << "Failed to read config file: " << config_file_path << std::endl;
            return 1;
        }
        if (!igl::readOBJ(rest_mesh_file_path, flatV[i], F[i])) {
            std::cerr << "Failed to read rest mesh file: " << rest_mesh_file_path << std::endl;
            return 1;
        }
        if (!igl::readOBJ(converged_mesh_file_path, V3d[i], F3d[i])) {
            std::cerr << "Failed to read converged mesh file: " << converged_mesh_file_path << std::endl;
            return 1;
        }
        if (!readEdgeDofs(edge_dofs_file_path, edge_dofs[i])) {
            std::cerr << "Failed to read edge dofs file: " << edge_dofs_file_path << std::endl;
            return 1;
        }
        if (!readOptimizationLog(optimization_log_file_path, opt_logs[i])) {
            std::cerr << "Failed to read optimization log file: " << optimization_log_file_path << std::endl;
            return 1;
        }

        // compute the face edge vectors
        LibShell::MeshConnectivity cur_mesh(F3d[i]);
        Eigen::VectorXd cur_edge_dofs = edge_dofs[i];
        if (edge_dofs[i].size() == cur_mesh.nEdges()) {
            // we need to extend the edge dofs to general version
            cur_edge_dofs.resize(2 * cur_mesh.nEdges());
            for (int j = 0; j < cur_mesh.nEdges(); j++) {
                cur_edge_dofs(2 * j) = edge_dofs[i](j);
                cur_edge_dofs(2 * j + 1) = M_PI_2;
            }
        }
        face_edge_vecs[i] = get_face_edge_normal_vectors(V3d[i], cur_mesh, cur_edge_dofs);
    }

    // compute the distance between the two meshes
    Eigen::VectorXd surface_dist, edge_vec_dist;
    CompResult comp_result;
    comp_result.surface_dist = computeSurfaceDist(V3d[0], V3d[1], flatV[0], F[0], surface_dist);
    comp_result.edge_vec_dist = computeEdgeVecDist(flatV[0], F[0], face_edge_vecs[0], face_edge_vecs[1], edge_vec_dist);

    std::cout << "surface_dist: " << comp_result.surface_dist << std::endl;
    std::cout << "edge_vec_dist: " << comp_result.edge_vec_dist << std::endl;
    std::cout << "S1 sin energy: " << opt_logs[0].total_energy << ", is converged: " << opt_logs[0].log.is_converged
              << std::endl;
    std::cout << "S2 sin energy: " << opt_logs[1].total_energy << ", is converged: " << opt_logs[1].log.is_converged
              << std::endl;

    // save this to a file
    std::ofstream out_file(args.results_dir + "/comp_result.txt");
    out_file << "surface_dist: " << comp_result.surface_dist << std::endl;
    out_file << "edge_vec_dist: " << comp_result.edge_vec_dist << std::endl;
    out_file << "S1 sin energy: " << opt_logs[0].total_energy << ", is converged: " << opt_logs[0].log.is_converged
             << std::endl;
    out_file << "S2 sin energy: " << opt_logs[1].total_energy << ", is converged: " << opt_logs[1].log.is_converged
             << std::endl;
    out_file.close();

    if (args.with_gui) {
        // visualize the difference on the rest mesh
        polyscope::init();
        polyscope::SurfaceMesh* surface_mesh = nullptr;

        surface_mesh = polyscope::registerSurfaceMesh("rest mesh", flatV[0], F[0]);
        surface_mesh->setEnabled(true);

        surface_mesh->addVertexScalarQuantity("surface_dist", surface_dist);

        LibShell::MeshConnectivity cur_mesh(F[0]);

        std::vector<Eigen::Vector3d> face_edge_midpts(3 * F[0].rows());
        for (int i = 0; i < F[0].rows(); i++) {
            for (int j = 0; j < 3; j++) {
                int eid = cur_mesh.faceEdge(i, j);
                face_edge_midpts[3 * i + j] =
                    (flatV[0].row(cur_mesh.edgeVertex(eid, 0)) + flatV[0].row(cur_mesh.edgeVertex(eid, 1))) / 2.0;
            }
        }
        auto pt_mesh = polyscope::registerPointCloud("face edge midpts", face_edge_midpts);
        pt_mesh->setEnabled(true);
        pt_mesh->setPointRadius(0);

        pt_mesh->addVectorQuantity("S1 sin vec", face_edge_vecs[0]);
        pt_mesh->addVectorQuantity("S2 sin vec", face_edge_vecs[1]);

        // we should average the edge_vec_dist to the face distance for visualization
        Eigen::VectorXd face_edge_vec_dist(F[0].rows());
        face_edge_vec_dist.setZero();
        for (int i = 0; i < F[0].rows(); i++) {
            for (int j = 0; j < 3; j++) {
                face_edge_vec_dist(i) += edge_vec_dist(3 * i + j);
            }
        }
        surface_mesh->addFaceScalarQuantity("face_edge_vec_dist", face_edge_vec_dist);

        // add S1 sin and S2 sin to the gui
        for (int i = 0; i < 2; i++) {
            polyscope::registerSurfaceMesh(model_names[i], V3d[i], F[i]);
            std::vector<Eigen::Vector3d> face_edge_midpts(3 * F[i].rows());
            for (int j = 0; j < F[i].rows(); j++) {
                for (int k = 0; k < 3; k++) {
                    int eid = cur_mesh.faceEdge(j, k);
                    face_edge_midpts[3 * j + k] =
                        (V3d[i].row(cur_mesh.edgeVertex(eid, 0)) + V3d[i].row(cur_mesh.edgeVertex(eid, 1))) / 2.0;
                }
            }
            auto pt_mesh_3d = polyscope::registerPointCloud(model_names[i] + " face edge midpts", face_edge_midpts);
            pt_mesh_3d->setEnabled(true);
            pt_mesh_3d->setPointRadius(0);

            auto vec_3d = pt_mesh_3d->addVectorQuantity(model_names[i] + " vec", face_edge_vecs[i]);
            vec_3d->setEnabled(true);
        }
        polyscope::show();
    }
    return 0;
}
