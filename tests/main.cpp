#include <Eigen/Core>
#include <iostream>
#include <map>
#include <cmath>
#include "../include/MeshConnectivity.h"
#include "../include/ElasticShell.h"
#include "../include/MidedgeAngleTanFormulation.h"
#include "../include/MidedgeAngleSinFormulation.h"
#include "../include/MidedgeAverageFormulation.h"
#include "../include/MidedgeAngleThetaFormulation.h"
#include "../include/MidedgeAngleCompressiveFormulation.h"
#include "../include/StVKMaterial.h"
#include "../include/BilayerStVKMaterial.h"
#include "../include/TensionFieldStVKMaterial.h"
#include "../include/NeoHookeanMaterial.h"
#include "../include/RestState.h"
#include "findiff.h"
#include <random>

std::default_random_engine rng;

const int nummats = 4;
const int numsff = 5;    


template<class SFF>
static void testStretchingFiniteDifferences(
    const LibShell::MeshConnectivity &mesh,
    const Eigen::MatrixXd &curPos,
    const LibShell::MaterialModel<SFF> &mat,
    const LibShell::RestState &restState,
    bool verbose,
    FiniteDifferenceLog &log);

template<class SFF>
static void testBendingFiniteDifferences(
    const LibShell::MeshConnectivity &mesh,
    const Eigen::MatrixXd &curPos,
    const Eigen::VectorXd &edgeDOFs,
    const LibShell::MaterialModel<SFF> &mat,
    const LibShell::RestState &restState,
    bool verbose,
    FiniteDifferenceLog &globalLog,
    FiniteDifferenceLog &localLog);


void makeSquareMesh(int dim, Eigen::MatrixXd &V, Eigen::MatrixXi &F)
{
    V.resize(dim*dim, 3);
    F.resize(2 * (dim - 1) * (dim - 1), 3);
    int vrow = 0;
    int frow = 0;
    for (int i = 0; i < dim; i++)
    {
        for (int j = 0; j < dim; j++)
        {
            double y = -(i * 1.0 + (dim - i - 1) * -1.0) / double(dim - 1);
            double x = (j * 1.0 + (dim - j - 1) * -1.0) / double(dim - 1);

            V(vrow, 0) = x;
            V(vrow, 1) = y;
            V(vrow, 2) = 0;
            vrow++;

            if (i != 0 && j != 0)
            {
                int iprev = i - 1;
                int jprev = j - 1;
                F(frow, 0) = iprev * dim + jprev;
                F(frow, 1) = iprev * dim + j;
                F(frow, 2) = i * dim + j;
                frow++;
                F(frow, 0) = iprev * dim + jprev;
                F(frow, 1) = i * dim + j;
                F(frow, 2) = i * dim + jprev;
                frow++;
            }
        }
    }
}

void printDiffLog(const std::map<int, double> &difflog)
{
    for (auto it : difflog)
    {
        std::cout << it.first << "\t" << it.second << std::endl;
    }
}

template <class SFF>
void differenceTest(const LibShell::MeshConnectivity &mesh,
    const Eigen::MatrixXd &restPos,
    int matid,
    bool verbose)
{
    Eigen::MatrixXd curPos = restPos;
    curPos.setRandom();
    const double PI = 3.14159263535898;

    Eigen::VectorXd edgeDOFs;
    SFF::initializeExtraDOFs(edgeDOFs, mesh, curPos);
    int nedgeDOFs = (int)edgeDOFs.size();
    std::uniform_real_distribution<double> angdist(-PI / 2, PI / 2);
    for (int i = 0; i < nedgeDOFs; i++)
    {
        edgeDOFs[i] = angdist(rng);
    }
    std::vector<Eigen::Matrix2d> abar1;
    LibShell::ElasticShell<SFF>::firstFundamentalForms(mesh, curPos, abar1);

    std::vector<Eigen::Matrix2d> bbar1;
    LibShell::ElasticShell<SFF>::secondFundamentalForms(mesh, curPos, edgeDOFs, bbar1);

    std::uniform_real_distribution<double> logThicknessDist(-6, 0);
    int nfaces = mesh.nFaces();
    std::vector<double> thicknesses1(nfaces);
    std::vector<double> thicknesses2(nfaces);

    // set abar2, bbar2 slightly different than abar1, bbar1 for testing bilayers
    std::vector<Eigen::Matrix2d> abar2;
    std::vector<Eigen::Matrix2d> bbar2;
    for (int i = 0; i < nfaces; i++)
    {
        abar2.push_back(0.9 * abar1[i]);
        bbar2.push_back(0.9 * bbar1[i]);
    }

    for (int i = 0; i < nfaces; i++)
    {
        thicknesses1[i] = std::pow(10.0, logThicknessDist(rng));
        thicknesses2[i] = std::pow(10.0, logThicknessDist(rng));
    }

    std::uniform_real_distribution<double> loglamedist(-1, 1);

    for (int lameiter1 = 0; lameiter1 < 2; lameiter1++)
    {
        double lameAlpha1 = 0;
        double lameBeta1 = 0;
        (lameiter1 == 1 ? lameAlpha1 : lameBeta1) = std::pow(10.0, loglamedist(rng));
        std::vector<double> lameAlpha1v;
        lameAlpha1v.resize(nfaces, lameAlpha1);
        std::vector<double> lameBeta1v;
        lameBeta1v.resize(nfaces, lameBeta1);

        for (int lameiter2 = 0; lameiter2 < 2; lameiter2++)
        {
            double lameAlpha2 = 0;
            double lameBeta2 = 0;
            (lameiter2 == 1 ? lameAlpha2 : lameBeta2) = std::pow(10.0, loglamedist(rng));
            std::vector<double> lameAlpha2v;
            lameAlpha2v.resize(nfaces, lameAlpha2);
            std::vector<double> lameBeta2v;
            lameBeta2v.resize(nfaces, lameBeta2);

            bool skip = false;

            LibShell::MaterialModel<SFF>* mat;
            LibShell::RestState* restState;

            switch (matid)
            {
            case 0:
            {
                if (lameiter1 != lameiter2)
                {
                    skip = true;
                }
                else
                {
                    std::cout << "NeoHookeanMaterial, alpha = " << lameAlpha1 << ", beta = " << lameBeta1 << std::endl;
                    mat = new LibShell::NeoHookeanMaterial<SFF>();
                    LibShell::MonolayerRestState* rs = new LibShell::MonolayerRestState;
                    rs->thicknesses = thicknesses1;
                    rs->abars = abar1;
                    rs->bbars = bbar1;
                    rs->lameAlpha = lameAlpha1v;
                    rs->lameBeta = lameBeta1v;
                    restState = rs;
                }
                break;
            }
            case 1:
            {
                if (lameiter1 != lameiter2)
                {
                    skip = true;
                }
                else
                {
                    std::cout << "StVKMaterial, alpha = " << lameAlpha1 << ", beta = " << lameBeta1 << std::endl;
                    mat = new LibShell::StVKMaterial<SFF>();
                    LibShell::MonolayerRestState* rs = new LibShell::MonolayerRestState;
                    rs->thicknesses = thicknesses1;
                    rs->abars = abar1;
                    rs->bbars = bbar1;
                    rs->lameAlpha = lameAlpha1v;
                    rs->lameBeta = lameBeta1v;
                    restState = rs;
                }
                break;
            }
            case 2:
            {
                if (lameiter1 != lameiter2)
                {
                    skip = true;
                }
                else
                {
                    std::cout << "TensionFieldStVKMaterial, alpha = " << lameAlpha1 << ", beta = " << lameBeta1 << std::endl;
                    mat = new LibShell::TensionFieldStVKMaterial<SFF>();
                    LibShell::MonolayerRestState* rs = new LibShell::MonolayerRestState;
                    rs->thicknesses = thicknesses1;
                    rs->abars = abar1;
                    rs->bbars = bbar1;
                    rs->lameAlpha = lameAlpha1v;
                    rs->lameBeta = lameBeta1v;
                    restState = rs;
                }
                break;
            }
            case 3:
            {
                std::cout << "BilayerStVKMaterial, alpha1 = " << lameAlpha1 << ", beta1 = " << lameBeta1 << ", alpha2 = " << lameAlpha2 << ", beta2 = " << lameBeta2 << std::endl;
                mat = new LibShell::BilayerStVKMaterial<SFF>();
                LibShell::BilayerRestState* rs = new LibShell::BilayerRestState;
                rs->layers[0].thicknesses = thicknesses1;
                rs->layers[1].thicknesses = thicknesses2;
                rs->layers[0].abars = abar1;
                rs->layers[1].abars = abar2;
                rs->layers[0].bbars = bbar1;
                rs->layers[1].bbars = bbar2;
                rs->layers[0].lameAlpha = lameAlpha1v;
                rs->layers[0].lameBeta = lameBeta1v;
                rs->layers[1].lameAlpha = lameAlpha2v;
                rs->layers[1].lameBeta = lameBeta2v;
                restState = rs;
                break;
            }
            default:
                assert(false);
            }

            if (skip)
                continue;

            curPos.setRandom();
            for (int i = 0; i < nedgeDOFs; i++)
            {
                edgeDOFs[i] = angdist(rng);
            }

            /*FiniteDifferenceLog localStretchingLog;
            testStretchingFiniteDifferences(mesh, curPos, *mat, *restState, verbose, localStretchingLog);
            std::cout << "Stretching (stencil):" << std::endl;
            localStretchingLog.printStats();*/

            FiniteDifferenceLog globalBendingLog;
            FiniteDifferenceLog localBendingLog;
            testBendingFiniteDifferences(mesh, curPos, edgeDOFs, *mat, *restState, verbose, globalBendingLog, localBendingLog);
            std::cout << "Bending (global):" << std::endl;
            globalBendingLog.printStats();
            std::cout << "Bending (stencil):" << std::endl;
            localBendingLog.printStats();
            std::cout << std::endl;

            delete mat;
            delete restState;
        }
    }
}

template<class SFF> 
double bilayerTest(const LibShell::MeshConnectivity& mesh,
    const Eigen::MatrixXd& restPos,
    const Eigen::VectorXd& thicknesses,    
    double lameAlpha, double lameBeta)
{    
    Eigen::MatrixXd curPos = restPos;
    curPos.setRandom();
    Eigen::VectorXd edgeDOFs;
    SFF::initializeExtraDOFs(edgeDOFs, mesh, curPos);
    int nedgeDOFs = (int)edgeDOFs.size();


    LibShell::MonolayerRestState monoRestState;
    monoRestState.thicknesses.resize(mesh.nFaces());
    monoRestState.lameAlpha.resize(mesh.nFaces());
    monoRestState.lameBeta.resize(mesh.nFaces());
    for (int i = 0; i < mesh.nFaces(); i++)
    {
        monoRestState.thicknesses[i] = thicknesses[i];
        monoRestState.lameAlpha[i] = lameAlpha;
        monoRestState.lameBeta[i] = lameBeta;
    }

    LibShell::BilayerRestState biRestState;
    biRestState.layers[0].thicknesses.resize(mesh.nFaces());
    biRestState.layers[1].thicknesses.resize(mesh.nFaces());
    biRestState.layers[0].lameAlpha.resize(mesh.nFaces());
    biRestState.layers[0].lameBeta.resize(mesh.nFaces());
    biRestState.layers[1].lameAlpha.resize(mesh.nFaces());
    biRestState.layers[1].lameBeta.resize(mesh.nFaces());
    for (int i = 0; i < mesh.nFaces(); i++)
    {
        biRestState.layers[0].thicknesses[i] = thicknesses[i];
        biRestState.layers[1].thicknesses[i] = thicknesses[i];
        biRestState.layers[0].lameAlpha[i] = lameAlpha;
        biRestState.layers[0].lameBeta[i] = lameBeta;
        biRestState.layers[1].lameAlpha[i] = lameAlpha;
        biRestState.layers[1].lameBeta[i] = lameBeta;
    }

    LibShell::ElasticShell<SFF>::firstFundamentalForms(mesh, restPos, monoRestState.abars);
    LibShell::ElasticShell<SFF>::secondFundamentalForms(mesh, restPos, edgeDOFs, monoRestState.bbars);

    LibShell::ElasticShell<SFF>::firstFundamentalForms(mesh, restPos, biRestState.layers[0].abars);
    LibShell::ElasticShell<SFF>::firstFundamentalForms(mesh, restPos, biRestState.layers[1].abars);
    LibShell::ElasticShell<SFF>::secondFundamentalForms(mesh, restPos, edgeDOFs, biRestState.layers[0].bbars);
    LibShell::ElasticShell<SFF>::secondFundamentalForms(mesh, restPos, edgeDOFs, biRestState.layers[1].bbars);
    
    LibShell::StVKMaterial<SFF> monomat;
    LibShell::BilayerStVKMaterial<SFF> bimat;

    double energy1 = LibShell::ElasticShell<SFF>::elasticEnergy(mesh, curPos, edgeDOFs, monomat, monoRestState, 0, NULL, NULL);
    double energy2 = LibShell::ElasticShell<SFF>::elasticEnergy(mesh, curPos, edgeDOFs, bimat, biRestState, 0, NULL, NULL);
    
    return std::fabs(energy1 - energy2);
}

template<class SFF> 
void getHessian(const LibShell::MeshConnectivity &mesh, 
    const Eigen::MatrixXd &curPos, 
    const Eigen::VectorXd &thicknesses, 
    int matid,
    double lameAlpha, double lameBeta,
    Eigen::SparseMatrix<double> &H)
{
    Eigen::VectorXd edgeDOFs;
    SFF::initializeExtraDOFs(edgeDOFs, mesh, curPos);
    int nedgeDOFs = (int)edgeDOFs.size();

    LibShell::MonolayerRestState restState;
    restState.thicknesses.resize(mesh.nFaces());
    for (int i = 0; i < mesh.nFaces(); i++)
        restState.thicknesses[i] = thicknesses[i];

    restState.lameAlpha.resize(mesh.nFaces(), lameAlpha);
    restState.lameBeta.resize(mesh.nFaces(), lameBeta);

    LibShell::ElasticShell<SFF>::firstFundamentalForms(mesh, curPos, restState.abars);
    LibShell::ElasticShell<SFF>::secondFundamentalForms(mesh, curPos, edgeDOFs, restState.bbars);

    std::vector<Eigen::Triplet<double> > hessian;

    LibShell::MaterialModel<SFF> *mat;
    switch (matid)
    {
    case 0:
        mat = new LibShell::NeoHookeanMaterial<SFF>();
        break;
    case 1:
        mat = new LibShell::StVKMaterial<SFF>();
        break;
    case 2:
        mat = new LibShell::TensionFieldStVKMaterial<SFF>();
        break;
    case 3:
        H.resize(0, 0);
        return;
    default:
        assert(false);
    }

    LibShell::ElasticShell<SFF>::elasticEnergy(mesh, curPos, edgeDOFs, *mat, restState, 0, NULL, &hessian);

    int nverts = curPos.rows();
    int nedges = mesh.nEdges();
    int dim = 3 * nverts + nedgeDOFs;
    H.resize(dim, dim);
    H.setFromTriplets(hessian.begin(), hessian.end());

    delete mat;
}


void consistencyTests(const LibShell::MeshConnectivity &mesh, const Eigen::MatrixXd &restPos)
{
    std::uniform_real_distribution<double> logThicknessDist(-6, 0);
    int nfaces = mesh.nFaces();
    Eigen::VectorXd thicknesses(nfaces);
    for (int i = 0; i < nfaces; i++)
    {
        thicknesses[i] = std::pow(10.0, logThicknessDist(rng));
    }

    std::uniform_real_distribution<double> loglamedist(-1, 1);

    for (int lameiter = 0; lameiter < 2; lameiter++)
    {
        double lameAlpha = 0;
        double lameBeta = 0;
        (lameiter == 1 ? lameAlpha : lameBeta) = std::pow(10.0, loglamedist(rng));
        std::cout << "Testing with alpha = " << lameAlpha << ", beta = " << lameBeta << std::endl;
        
        int numoptions = nummats * numsff;
        std::vector<Eigen::SparseMatrix<double> > hessians(numoptions);
        for (int i = 0; i < nummats; i++)
        {
            for (int j = 4; j < numsff; j++)
            {
                switch (j)
                {
                case 0:
                    getHessian<LibShell::MidedgeAngleTanFormulation>(mesh, restPos, thicknesses, i, lameAlpha, lameBeta, hessians[i*numsff + j]);
                    break;
                case 1:
                    getHessian<LibShell::MidedgeAngleSinFormulation>(mesh, restPos, thicknesses, i, lameAlpha, lameBeta, hessians[i*numsff + j]);
                    break;
                case 2:
                    getHessian<LibShell::MidedgeAverageFormulation>(mesh, restPos, thicknesses, i, lameAlpha, lameBeta, hessians[i*numsff + j]);
                    break;
                case 3:
                    getHessian<LibShell::MidedgeAngleThetaFormulation>(mesh, restPos, thicknesses, i, lameAlpha, lameBeta,
                                                                     hessians[i * numsff + j]);
                    break;
                case 4:
                    getHessian<LibShell::MidedgeAngleCompressiveFormulation>(mesh, restPos, thicknesses, i, lameAlpha,
                                                                             lameBeta, hessians[i * numsff + j]);
                    break;
                default:
                    assert(false);
                }                
            }
        }
        for (int i = 0; i < nummats; i++)
        {
            for (int j = 4; j < numsff; j++)
            {
                for (int k = 0; k < nummats; k++)
                {
                    for (int l = 4; l < numsff; l++)
                    {
                        // ignore TensionField material
                        if (i == 2 || k == 2)
                            continue;
                        // ignore BilayerStVK material
                        if (i == 3 || k == 3)
                            continue;
                        int idx1 = i * numsff + j;
                        int idx2 = k * numsff + l;
                        if (idx2 <= idx1)
                            continue;
                        std::string matnames[] = { "Neohk", "StVK" };
                        std::string sffnames[] = { "Tan", "Sin", "Avg", "Theta" };
                        std::cout << "(" << matnames[i] << ", " << sffnames[j] << ") vs (" << matnames[k] << ", " << sffnames[l] << "): ";
                        double diff = 0;
                        int nverts = restPos.rows();
                        for (int m = 0; m < 3 * nverts; m++)
                        {
                            for (int n = m; n < 3 * nverts; n++)
                            {
                                diff += std::fabs(hessians[idx1].coeff(m, n) - hessians[idx2].coeff(m, n));
                            }
                        }
                        std::cout << diff << std::endl;
                    }                    
                }
            }
        }


        // bilayer consistency tests
        std::cout << "Bilayer consistency tests: " << std::endl;
        for (int j = 4; j < numsff; j++)
        {
            double diff = 0;
            switch (j)
            {
            case 0:
                diff = bilayerTest<LibShell::MidedgeAngleTanFormulation>(mesh, restPos, thicknesses, lameAlpha, lameBeta);
                break;
            case 1:
                diff = bilayerTest<LibShell::MidedgeAngleSinFormulation>(mesh, restPos, thicknesses, lameAlpha, lameBeta);
                break;
            case 2:
                diff = bilayerTest<LibShell::MidedgeAverageFormulation>(mesh, restPos, thicknesses, lameAlpha, lameBeta);
                break;
            case 3:
                diff =
                    bilayerTest<LibShell::MidedgeAngleThetaFormulation>(mesh, restPos, thicknesses, lameAlpha, lameBeta);
                break;
            case 4: 
                diff = bilayerTest<LibShell::MidedgeAngleCompressiveFormulation>(mesh, restPos, thicknesses, lameAlpha,
                                                                                 lameBeta);
                break;
            default:
                assert(false);
            }
            std::string sffnames[] = {"Tan", "Sin", "Avg", "Theta", "Compressive-tan"};
            std::cout << "  - " << sffnames[j] << ": " << diff << std::endl;
        }
    }
}


int main()
{
    int dim = 2;
    bool verbose = true;
    bool testderivatives = true;
    bool testconsistency = false;
    
    
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    makeSquareMesh(dim, V, F);

    Eigen::MatrixXd restV = V;

    if (dim == 2) {
        V(2, 2) = 0.5;
        V(1, 2) = 0.5;
    }

    LibShell::MeshConnectivity mesh(F);

    Eigen::VectorXd edgeDOFs;
    LibShell::MidedgeAngleCompressiveFormulation::initializeExtraDOFs(edgeDOFs, mesh, V);

    constexpr int nedgedofs = LibShell::MidedgeAngleCompressiveFormulation::numExtraDOFs;

    Eigen::Matrix<double, 4, 18 + 3 * nedgedofs> bderiv;
    std::vector<Eigen::Matrix<double, 18 + 3 * nedgedofs, 18 + 3 * nedgedofs>> bhess;
    Eigen::Matrix2d b = LibShell::MidedgeAngleCompressiveFormulation::secondFundamentalForm(
        mesh, V, edgeDOFs, 0, &bderiv, &bhess);

    Eigen::VectorXd tan_edgeDOFs;
    LibShell::MidedgeAngleTanFormulation::initializeExtraDOFs(tan_edgeDOFs, mesh, V);

    constexpr int ntanedgedofs = LibShell::MidedgeAngleTanFormulation::numExtraDOFs;

    Eigen::Matrix<double, 4, 18 + 3 * ntanedgedofs> btanderiv;
    std::vector<Eigen::Matrix<double, 18 + 3 * ntanedgedofs, 18 + 3 * ntanedgedofs>> btanhess;
    Eigen::Matrix2d btan =
        LibShell::MidedgeAngleTanFormulation::secondFundamentalForm(mesh, V, tan_edgeDOFs, 0, &btanderiv, &btanhess);

    std::cout << "compression formula: \n";
    std::cout << "b: \n" << b << std::endl;
    std::cout << "bderiv: \n" << bderiv << std::endl;

    int faceid = 0;

   auto test_energy_derivatives = [&V, &nedgedofs, &mesh](
        const Eigen::VectorXd &x, const Eigen::VectorXd &perturb,
        const std::function<double(const Eigen::VectorXd &, Eigen::VectorXd *, Eigen::MatrixXd *)> func) {
            Eigen::VectorXd deriv;
            Eigen::MatrixXd hess;
            double f = func(x, &deriv, &hess);

            for (int i = 4; i < 10; i++) {
                double eps = std::pow(0.1, i);
                auto x_pert = x + eps * perturb;

                Eigen::VectorXd deriv_pert;
                double f_pert = func(x_pert, &deriv_pert, nullptr);

                std::cout << "eps: " << eps << std::endl;
                std::cout << "f-g check: "
                          << "f: " << f << ", f_pert: " << f_pert << ", diff: " << (f_pert - f) / eps - perturb.dot(deriv)
                          << std::endl;
                std::cout << "g-h check: " << ((deriv_pert - deriv) / eps - hess * perturb).norm() << std::endl;
            }
    };

   auto pos_edge_dofs_2_variable = [&mesh, &faceid](const Eigen::MatrixXd &cur_pos, const Eigen::VectorXd &cur_edge_dofs) {
       int ndofs_per_edge = cur_edge_dofs.size() / mesh.nEdges();

       int total_dofs = 18 + 3 * ndofs_per_edge;

       Eigen::VectorXd x(total_dofs);
       x.setZero();
       for (int i = 0; i < 3; i++) {
           int edge_id = mesh.faceEdge(faceid, i);
           int opp_vid = mesh.vertexOppositeFaceEdge(faceid, i);
           if (opp_vid != -1) {
               x.segment<3>(9 + 3 * i) = cur_pos.row(opp_vid);
           }

           for (int j = 0; j < ndofs_per_edge; j++) {
               x(18 + ndofs_per_edge * i + j) = cur_edge_dofs(ndofs_per_edge * edge_id + j);
           }
       }
       for (int i = 0; i < 3; i++) {
           x.segment<3>(3 * i) = cur_pos.row(mesh.faceVertex(faceid, i));
       }
       return x;
   };

   auto variable_2_pos_edge_dofs = [&mesh, &faceid](const Eigen::VectorXd &x, Eigen::MatrixXd &cur_pos,
                                       Eigen::VectorXd &cur_edge_dofs) {
       int ndofs_per_edge = cur_edge_dofs.size() / mesh.nEdges();

       int total_dofs = 18 + 3 * ndofs_per_edge;

       for (int i = 0; i < 3; i++) {
           int edge_id = mesh.faceEdge(faceid, i);
           int opp_vid = mesh.vertexOppositeFaceEdge(faceid, i);
           if (opp_vid != -1) {
               cur_pos.row(opp_vid) = x.segment<3>(9 + 3 * i);
           }

           for (int j = 0; j < ndofs_per_edge; j++) {
               cur_edge_dofs(ndofs_per_edge * edge_id + j) = x(18 + ndofs_per_edge * i + j);
           }
       }

       for (int i = 0; i < 3; i++) {
           cur_pos.row(mesh.faceVertex(faceid, i)) = x.segment<3>(3 * i);
       }
   };

    auto II_func = [&](const Eigen::VectorXd& x, Eigen::VectorXd* deriv, Eigen::MatrixXd* hess) {
        auto test_V = V;
        constexpr int test_nedgedofs = LibShell::MidedgeAngleCompressiveFormulation::numExtraDOFs;
        auto test_edge_dofs = edgeDOFs;

        variable_2_pos_edge_dofs(x, test_V, test_edge_dofs);

        Eigen::Matrix<double, 4, 18 + 3 * test_nedgedofs> test_bderiv;
        std::vector<Eigen::Matrix<double, 18 + 3 * test_nedgedofs, 18 + 3 * test_nedgedofs>> test_bhess;
        Eigen::Matrix2d test_b =
            LibShell::MidedgeAngleCompressiveFormulation::secondFundamentalForm(mesh, test_V, test_edge_dofs, faceid, deriv ? &test_bderiv : nullptr, hess ? &test_bhess : nullptr);
        if (deriv) {
            *deriv = (test_bderiv.row(0) + test_bderiv.row(1) + test_bderiv.row(3)).transpose();
        }
           
        if (hess) {
            *hess = test_bhess[0] + test_bhess[1] + test_bhess[3];
        }

        double energy = 0;
        energy = test_b(0, 0) + test_b(0, 1) + test_b(1, 1);
         
        return energy;
    };

    LibShell::StVKMaterial<LibShell::MidedgeAngleCompressiveFormulation> monomat;
    LibShell::MonolayerRestState monoRestState;


    

    Eigen::VectorXd x = pos_edge_dofs_2_variable(V, edgeDOFs);
    auto pert = x;
    pert.setRandom();
    pert.normalize();
    //for (int i = 0; i < 3; i++) {
    //    pert(18 + 3 * i + 1) = 0;
    //    pert(18 + 3 * i + 2) = 0;
    //}
    test_energy_derivatives(x, pert, II_func);

    /*Eigen::VectorXd tan_x = pos_edge_dofs_2_variable(V, tan_edgeDOFs);
    Eigen::VectorXd tan_pert = tan_x;
    tan_pert.segment(0, 18) = pert.segment(0, 18);
    for (int i = 0; i < 3; i++) {
        tan_pert(18 + i) = pert(18 + 3 * i);
    }

    test_energy_derivatives(tan_x, tan_pert, tan_II_func);*/


    std::default_random_engine generator;

    if (testderivatives)
    {
        std::cout << "Running finite difference tests" << std::endl;;
        for (int i = 1; i < 2; i++)
        {
            for (int j = 4; j < 5; j++)
            {
                std::cout << "Starting trial: ";
                switch (j)
                {
                case 0:
                    std::cout << "MidedgeAngleTanFormulation, ";
                    differenceTest<LibShell::MidedgeAngleTanFormulation>(mesh, V, i, verbose);
                    break;
                case 1:
                    std::cout << "MidedgeAngleSinFormulation, ";
                    differenceTest<LibShell::MidedgeAngleSinFormulation>(mesh, V, i, verbose);
                    break;
                case 2:
                    std::cout << "MidedgeAverageFormulation, ";
                    differenceTest<LibShell::MidedgeAverageFormulation>(mesh, V, i, verbose);
                    break;
                case 3:
                    std::cout << "MidedgeAngleThetaFormulation, ";
                    differenceTest<LibShell::MidedgeAngleThetaFormulation>(mesh, V, i, verbose);
                    break;
                case 4:
                    std::cout << "MidedgeAngleCompressiveFormulation, ";
                    differenceTest<LibShell::MidedgeAngleCompressiveFormulation>(mesh, V, i, verbose);
                    break;
                default:
                    assert(false);
                }
            }
        }
        std::cout << "Finite difference tests done" << std::endl;
    }
    if (testconsistency)
    {
        std::cout << "Running consistency tests" << std::endl;;
        consistencyTests(mesh, V);
        std::cout << "Consistency tests done" << std::endl;
    }
}


template <class SFF>
void testStretchingFiniteDifferences(
    const LibShell::MeshConnectivity &mesh,
    const Eigen::MatrixXd &curPos,
    const LibShell::MaterialModel<SFF> &mat,
    const LibShell::RestState &restState,
    bool verbose,
    FiniteDifferenceLog &log)
{
    log.clear();
    int nfaces = mesh.nFaces();
    int nedges = mesh.nEdges();
    int nverts = (int)curPos.rows();

    Eigen::MatrixXd testpos = curPos;
    testpos.setRandom();

    std::vector<int> epsilons = {-2, -3, -4, -5, -6};
    
    for (auto epsilon : epsilons)
    {
        double pert = std::pow(10.0, epsilon);

        for (int face = 0; face < nfaces; face++)
        {

            Eigen::Matrix<double, 1, 9> deriv;
            Eigen::Matrix<double, 9, 9> hess;
            double result = mat.stretchingEnergy(mesh, testpos, restState, face, &deriv, &hess);

            for (int j = 0; j < 3; j++)
            {
                for (int k = 0; k < 3; k++)
                {
                    Eigen::MatrixXd fwdpertpos = testpos;
                    Eigen::MatrixXd backpertpos = testpos;
                    fwdpertpos(mesh.faceVertex(face, j), k) += pert;
                    backpertpos(mesh.faceVertex(face, j), k) -= pert;

                    Eigen::Matrix<double, 1, 9> fwdpertderiv;
                    Eigen::Matrix<double, 1, 9> backpertderiv;
                    double fwdnewresult = mat.stretchingEnergy(mesh, fwdpertpos, restState, face, &fwdpertderiv, NULL);
                    double backnewresult = mat.stretchingEnergy(mesh, backpertpos, restState, face, &backpertderiv, NULL);
                    double findiff = (fwdnewresult - backnewresult) / 2.0 / pert;
                    if(verbose) std::cout << '(' << j << ", " << k << ") " << findiff << " " << deriv(0, 3 * j + k) << std::endl;
                    log.addEntry(pert, deriv(0, 3 * j + k), findiff);
                    Eigen::Matrix<double, 1, 9> derivdiff = (fwdpertderiv - backpertderiv) / 2.0 / pert;
                    if (verbose)
                    {
                        std::cout << derivdiff << std::endl;
                        std::cout << "//" << std::endl;
                        std::cout << hess.row(3 * j + k) << std::endl << std::endl;
                    }
                    for (int l = 0; l < 9; l++)
                    {
                        log.addEntry(pert, hess(3 * j + k, l), derivdiff(0, l));
                    }                    
                }
            }
        }        
    }
}

template <class SFF>
void testBendingFiniteDifferences(
    const LibShell::MeshConnectivity &mesh,
    const Eigen::MatrixXd &curPos,
    const Eigen::VectorXd &edgeDOFs,
    const LibShell::MaterialModel<SFF> &mat,
    const LibShell::RestState &restState,
    bool verbose,
    FiniteDifferenceLog &globalLog,
    FiniteDifferenceLog &localLog)
{
    globalLog.clear();
    localLog.clear();

    int nfaces = mesh.nFaces();
    int nedges = mesh.nEdges();
    int nverts = (int)curPos.rows();

    Eigen::MatrixXd testpos = curPos;
    testpos.setRandom();
    Eigen::VectorXd testedge = edgeDOFs;
    testedge.setRandom();


    // for global testing
    Eigen::MatrixXd posPert(testpos.rows(), testpos.cols());
    posPert.setRandom();

    
    Eigen::VectorXd edgePert(testedge.size());
    edgePert.setRandom();

    constexpr int nedgedofs = SFF::numExtraDOFs;

    std::vector<int> epsilons = {-2, -3, -4, -5, -6};
    for (auto epsilon : epsilons)
    {

        double pert = std::pow(10.0, epsilon);

        // global testing
        {
            Eigen::MatrixXd fwdpertpos = testpos + pert * posPert;
            Eigen::MatrixXd backpertpos = testpos - pert * posPert;
            Eigen::VectorXd fwdpertedge = testedge + pert * edgePert;
            Eigen::VectorXd backpertedge = testedge - pert * edgePert;

            Eigen::VectorXd pertVec(3 * nverts + nedgedofs * nedges);
            for (int i = 0; i < nverts; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    pertVec[3 * i + j] = posPert(i, j);
                }
            }
            for (int i = 0; i < nedgedofs * nedges; i++)
            {
                pertVec[3 * nverts + i] = edgePert[i];
            }

            Eigen::VectorXd deriv;
            std::vector<Eigen::Triplet<double> > hess;
            double result = LibShell::ElasticShell<SFF>::elasticEnergy(mesh, testpos, testedge, mat, restState,
                LibShell::ElasticShell<SFF>::EnergyTerm::ET_BENDING, 0, &deriv, &hess);

            Eigen::VectorXd fwdderiv;
            Eigen::VectorXd backderiv;

            double fwdnewresult = LibShell::ElasticShell<SFF>::elasticEnergy(mesh, fwdpertpos, fwdpertedge, mat, restState, 
                LibShell::ElasticShell<SFF>::EnergyTerm::ET_BENDING, 0, &fwdderiv, NULL);
            double backnewresult = LibShell::ElasticShell<SFF>::elasticEnergy(mesh, backpertpos, backpertedge, mat, restState, 
                LibShell::ElasticShell<SFF>::EnergyTerm::ET_BENDING, 0, &backderiv, NULL);

            double findiff = (fwdnewresult - backnewresult) / 2.0 / pert;
            double direcderiv = deriv.dot(pertVec);
            if (verbose) std::cout << "g " << findiff << " " << direcderiv << std::endl;
            globalLog.addEntry(pert, findiff, direcderiv);
            
            Eigen::VectorXd diffderiv = (fwdderiv - backderiv) / 2.0 / pert;
            Eigen::SparseMatrix<double> H(3 * nverts + nedgedofs * nedges, 3 * nverts + nedgedofs * nedges);
            H.setFromTriplets(hess.begin(), hess.end());
            Eigen::VectorXd derivderiv = H * pertVec;
            if (verbose)
            {
                std::cout << diffderiv.transpose() << std::endl;
                std::cout << "//" << std::endl;
                std::cout << derivderiv.transpose() << std::endl;
            }
            for (int i = 0; i < 3 * nverts + nedgedofs * nedges; i++)
            {
                globalLog.addEntry(pert, diffderiv[i], derivderiv[i]);
            }
        }

        for (int face = 0; face < nfaces; face++)
        {
            Eigen::Matrix<double, 1, 18 + 3 * nedgedofs> deriv;
            Eigen::Matrix<double, 18 + 3 * nedgedofs, 18 + 3 * nedgedofs> hess;
            double result = mat.bendingEnergy(mesh, testpos, testedge, restState, face, &deriv, &hess);

            for (int j = 0; j < 3; j++)
            {
                for (int k = 0; k < 3; k++)
                {
                    Eigen::MatrixXd fwdpertpos = testpos;
                    Eigen::MatrixXd backpertpos = testpos;
                    fwdpertpos(mesh.faceVertex(face, j), k) += pert;
                    backpertpos(mesh.faceVertex(face, j), k) -= pert;
                    Eigen::Matrix<double, 1, 18 + 3 * nedgedofs> fwdpertderiv;
                    Eigen::Matrix<double, 1, 18 + 3 * nedgedofs> backpertderiv;
                    double fwdnewresult = mat.bendingEnergy(mesh, fwdpertpos, testedge, restState, face, &fwdpertderiv, NULL);
                    double backnewresult = mat.bendingEnergy(mesh, backpertpos, testedge, restState, face, &backpertderiv, NULL);
                    double findiff = (fwdnewresult - backnewresult) / 2.0 / pert;
                    if(verbose) std::cout << '(' << j << ", " << k << ") " << findiff << " " << deriv(0, 3 * j + k) << std::endl;
                    localLog.addEntry(pert, deriv(0, 3 * j + k), findiff);

                    Eigen::MatrixXd derivdiff = (fwdpertderiv - backpertderiv) / 2.0 / pert;
                    if (verbose)
                    {
                        std::cout << derivdiff << std::endl;
                        std::cout << "//" << std::endl;
                        std::cout << hess.row(3 * j + k) << std::endl << std::endl;
                    }
                    for (int l = 0; l < 18 + 3 * nedgedofs; l++)
                    {
                        localLog.addEntry(pert, hess(3 * j + k, l), derivdiff(0, l));
                    }
                }
    
                int oppidx = mesh.vertexOppositeFaceEdge(face, j);
                if (oppidx != -1)
                {
                    for (int k = 0; k < 3; k++)
                    {
                        Eigen::MatrixXd fwdpertpos = testpos;
                        Eigen::MatrixXd backpertpos = testpos;
                        fwdpertpos(oppidx, k) += pert;
                        backpertpos(oppidx, k) -= pert;
                        Eigen::Matrix<double, 1, 18 + 3 * nedgedofs> fwdpertderiv;
                        Eigen::Matrix<double, 1, 18 + 3 * nedgedofs> backpertderiv;
                        double fwdnewresult = mat.bendingEnergy(mesh, fwdpertpos, testedge, restState, face, &fwdpertderiv, NULL);
                        double backnewresult = mat.bendingEnergy(mesh, backpertpos, testedge, restState, face, &backpertderiv, NULL);
                        double findiff = (fwdnewresult - backnewresult) / 2.0 / pert;
                        Eigen::MatrixXd derivdiff = (fwdpertderiv - backpertderiv) / 2.0 / pert;
                        if (verbose)
                        {
                            std::cout << "opp (" << j << ", " << k << ") " << findiff << " " << deriv(0, 9 + 3 * j + k) << std::endl;
                            std::cout << derivdiff << std::endl;
                            std::cout << "//" << std::endl;
                            std::cout << hess.row(9 + 3 * j + k) << std::endl << std::endl;
                        }
                        localLog.addEntry(pert, deriv(0, 9 + 3 * j + k), findiff);
                        for (int l = 0; l < 18 + 3 * nedgedofs; l++)
                        {
                            localLog.addEntry(pert, hess(9 + 3 * j + k, l), derivdiff(0, l));
                        }                        
                    }
                }
                
                for (int k = 0; k < nedgedofs; k++)
                {
                    Eigen::VectorXd fwdpertedge = testedge;
                    Eigen::VectorXd backpertedge = testedge;
                    fwdpertedge[nedgedofs * mesh.faceEdge(face, j) + k] += pert;
                    backpertedge[nedgedofs * mesh.faceEdge(face, j) + k] -= pert;
                    Eigen::Matrix<double, 1, 18 + 3 * nedgedofs> fwdpertderiv;
                    Eigen::Matrix<double, 1, 18 + 3 * nedgedofs> backpertderiv;
                    double fwdnewresult = mat.bendingEnergy(mesh, testpos, fwdpertedge, restState, face, &fwdpertderiv, NULL);
                    double backnewresult = mat.bendingEnergy(mesh, testpos, backpertedge, restState, face, &backpertderiv, NULL);
                    double findiff = (fwdnewresult - backnewresult) / 2.0 / pert;
                    if(verbose) std::cout << findiff << " " << deriv(0, 18 + nedgedofs * j + k) << std::endl;
                    localLog.addEntry(pert, deriv(0, 18 + nedgedofs * j + k), findiff);
                    
                    Eigen::MatrixXd derivdiff = (fwdpertderiv - backpertderiv) / 2.0 / pert;
                    if (verbose)
                    {
                        std::cout << derivdiff << std::endl;
                        std::cout << "//" << std::endl;
                        std::cout << hess.row(18 + nedgedofs * j + k) << std::endl << std::endl;
                    }
                    for (int l = 0; l < 18 + 3 * nedgedofs; l++)
                    {
                        localLog.addEntry(pert, hess(18 + nedgedofs * j + k, l), derivdiff(0, l));
                    }
                }
            }
        }
    }
}
