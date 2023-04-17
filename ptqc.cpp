#include "QuEST.h"
#include <cstdlib>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/QR>
#include <eigen3/Eigen/Eigenvalues>
#include <iostream>
#include <random>
#include <vector>
// #include <Spectra/GenEigsSolver.h>

using mat     = Eigen::Matrix<std::complex<qreal>, Eigen::Dynamic, Eigen::Dynamic>;
using mat6c   = Eigen::Matrix<std::complex<qreal>, 6, 6>;
using complex = std::complex<qreal>;
using qint    = unsigned long long;

void printComplexMatrixN(const ComplexMatrixN& mat) {
    int dim = std::pow(2, mat.numQubits);
    for (qint i = 0; i < dim; i++) {
        for (qint j = 0; j < dim; j++) {
            std::cout << mat.real[i][j] << "+" << mat.imag[i][j] << "i, ";
        }
        std::cout << '\n';
    }
}

std::random_device rd;
std::mt19937 gen(rd());
std::normal_distribution<qreal> dist(0., 1.);
std::uniform_real_distribution<double> uniform(0, 1);

std::complex<qreal> gen_complex() {
    qreal real = dist(gen);
    qreal imag = dist(gen);
    // std::complex<qreal> b{real, imag};
    return {real, imag};
}
mat6c gen_mat() {
    mat6c mat;
    qreal norm = 1 / std::sqrt(2);
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            mat(i, j) = gen_complex();
        }
    }
    mat = mat * norm;
    return mat;
}

mat6c haar_random() {
    auto A      = gen_mat();
    auto qr     = A.householderQr();
    auto q      = qr.householderQ();
    auto lambda = qr.matrixQR().diagonal();
    mat lambda1 = lambda.cwiseQuotient(lambda.cwiseAbs()).asDiagonal();
    mat U       = q * lambda1;
    // std::cout << U.transpose().conjugate() * U << std::endl;
    return U;
}

ComplexMatrixN haar_random_8() {
    auto A       = createComplexMatrixN(3);
    auto U6      = haar_random();
    A.real[0][0] = 1.;
    A.real[7][7] = 1.;
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            auto val             = U6(i, j);
            A.real[i + 1][j + 1] = val.real();
            A.imag[i + 1][j + 1] = val.imag();
        }
    }
    // prqintComplexMatrixN(A);
    return A;
}

void printDensityMatrix(Qureg qureg) {
    auto d = 1 << qureg.numQubitsRepresented;
    for (qint i = 0; i < d; i++) {
        for (qint j = 0; j < d; j++) {
            auto val = getDensityAmp(qureg, i, j);
            std::printf("%.3f ", val.real);
        }
        std::printf("\n");
    }
}

void printDensityMatrix(const mat &dm) {
    auto d = dm.rows();
    for (qint i = 0; i < d; i++) {
        for (qint j = 0; j < d; j++) {
            auto val = dm(i, j);
            std::printf("%.3f ", val.real());
        }
        std::printf("\n");
    }
}

void printDensityMatrixEasy(const mat &dm) {
    auto d = dm.rows();
    for (qint i = 0; i < d; i++) {
        for (qint j = 0; j < d; j++) {
            auto val = dm(i, j);
            if (std::abs(val) > 1E-6) {
                
                std::printf("(%lld, %lld): %.6f + %.6f i\n", i, j, val.real(), val.imag());
            }
        }
        // std::printf("\n");
    }
}

void applyHaarRandomUnitary(Qureg qureg, int* targets) {
    // applyMatrixN(qureg, targets, 3, haar_random_8());
    multiQubitUnitary(qureg, targets, 3, haar_random_8());
}

void applyZZPM(Qureg qureg, int target1, int target2) {
    ComplexMatrix4 a{0, 0};
    a.real[0][0] = 1;
    a.real[3][3] = 1;
    ComplexMatrix4 b{0, 0};
    b.real[0][2] = 1;
    b.real[3][1] = 1;
    ComplexMatrix4 ops[2]{a, b};
    mixTwoQubitKrausMap(qureg, target1, target2, ops, 2);
}

void applyHRULayer(Qureg qureg, int t) {
    int mod = t % 3;
    while (mod + 2 < qureg.numQubitsRepresented) {
        int targets[3]{mod, mod + 1, mod + 2};
        applyHaarRandomUnitary(qureg, targets);
        mod += 3;
    }
}

void applyZZPMLayer(Qureg qureg, qint t, double p) {
    std::bernoulli_distribution bdist(p);
    int mod = 0;
    std::uniform_int_distribution<int> ui(0, 2);
    while (mod + 1 < qureg.numQubitsRepresented) {
        if (bdist(gen)) {
            applyZZPM(qureg, mod, mod + 1);
        }
        auto space = ui(gen);
        mod += (2 + space);
    }
}

qreal calcEntrophy(Qureg qureg, mat& rho, qint dim) {
    for (qint i = 0; i < dim; i++) {
        for (qint j = 0; j < dim; j++) {
            auto _c = getDensityAmp(qureg, i, j);
            complex c{_c.real, _c.imag};
            rho(i, j) = c;
        }
    }
    // std::cout << "density matrix: " << std::endl;
    // std::cout << rho << std::endl;
    Eigen::ComplexEigenSolver<mat> solver;
    solver.compute(rho);
    auto evs      = solver.eigenvalues();
    qreal entropy = 0.;
    for (auto _ev : evs) {
        auto ev = _ev.real();
        // std::cout << ev << " ";
        if (ev > 0 && ev < 1) {
            entropy -= ev * std::log2(ev);
        }
    }
    // std::cout << std::endl;
    return entropy;
}

int main(int argc, char** argv) {
    int n    = std::atoi(argv[1]);
    double p = std::atof(argv[2]);
    int T    = std::atoi(argv[3]);
    qint dim = 1 << n;

    QuESTEnv env = createQuESTEnv();
    Qureg qubits = createDensityQureg(n, env);

    // generate initial density matrix: random
    mat rho(dim, dim);
    for (qint i = 0; i < dim; i++) {
        for (qint j = 0; j < dim; j++) {
            rho(i, j) = gen_complex();
        }
    }
    rho = rho * rho.conjugate().transpose();
    rho = rho / rho.trace();

    qreal* reals = new qreal[dim * dim];
    qreal* imags = new qreal[dim * dim];
    for (qint i = 0; i < dim; i++) {
        for (qint j = 0; j < dim; j++) {
            auto c             = rho(i, j);
            reals[i * dim + j] = c.real();
            imags[i * dim + j] = c.imag();
        }
    }
    initStateFromAmps(qubits, reals, imags);
    delete[] reals;
    delete[] imags;

    auto ents = std::vector<qreal>();

    ents.emplace_back(calcEntrophy(qubits, rho, dim));
    for (int t = 0; t < T; t++) {
        applyHRULayer(qubits, t);
        applyZZPMLayer(qubits, t, p);
        if ((t + 1) % 100 == 0) {
        // if (t > 5000) {
            // mat oldRho = rho;
            auto ent = calcEntrophy(qubits, rho, dim);
            // double oldEnt = ents.back();
            ents.emplace_back(ent);
            if (ent < 1.05) {
                break;
            }
            // if (std::abs(oldEnt - ent) > 0.5) {
            //     std::printf("old: %.3f, new: %.3f\n", oldEnt, ent);

            //     std::printf("old: \n");
            //     printDensityMatrixEasy(oldRho);
            //     Eigen::ComplexEigenSolver<mat> solver;
            //     solver.compute(oldRho);
            //     std::cout << solver.eigenvalues() << std::endl;

            //     std::printf("new: \n");
            //     printDensityMatrixEasy(rho);
            //     Eigen::ComplexEigenSolver<mat> solver2;
            //     solver2.compute(rho);
            //     std::cout << solver2.eigenvalues() << std::endl;
            // }
        }
    }

    for (auto ent : ents) {
        std::cout << ent << std::endl;
    }

    destroyQureg(qubits, env);
    destroyQuESTEnv(env);
    return 0;
}
