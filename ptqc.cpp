#include "QuEST.h"
#include <cstdlib>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/QR>
#include <iostream>
#include <random>

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
std::normal_distribution<qreal> dist(0., 0.5);
std::uniform_real_distribution<double> uniform(0, 1);

std::complex<qreal> gen_complex() {
    qreal real = dist(gen);
    qreal imag = dist(gen);
    // std::complex<qreal> b{real, imag};
    return {real, imag};
}
mat6c gen_mat() {
    mat6c mat;
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            mat(i, j) = gen_complex();
        }
    }
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
    int mod = t % 3;
    std::uniform_int_distribution<int> ui(0, 2);
    while (mod + 2 < qureg.numQubitsRepresented) {
        auto pgen = uniform(gen);
        if (pgen < p) {
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
    // std::cout << rho << std::endl;
    auto evs        = rho.eigenvalues();
    qreal entropy = 0.;
    // std::cout << evs << std::endl;
    for (auto ev : evs) {
        entropy -= ev.real() * std::log(ev.real());
    }
    // std::cout << entropy << std::endl;
    return entropy;
    
}

int main(int argc, char** argv) {
    // ComplexMatrix4 op1;
    // std::cout << "sdf";
    int n    = std::atoi(argv[1]);
    double p = std::atoi(argv[2]);
    auto T   = std::atoi(argv[3]);
    qint dim = 1 << n;

    QuESTEnv env = createQuESTEnv();

    // haar_random_8();

    Qureg qubits = createDensityQureg(n, env);
    // qreal reals[n * n]{0};
    std::normal_distribution<qreal> dist1(0., 1.);
    mat mycoeffs(dim, dim);
    for (qint i = 0; i < dim; i++) {
        for (qint j = 0; j < dim; j++) {
            mycoeffs(i, j) = gen_complex();
        }
    }
    mycoeffs = mycoeffs * mycoeffs.conjugate().transpose();
    mycoeffs = mycoeffs / mycoeffs.trace();

    qreal* reals = new qreal[dim * dim];
    qreal* imags = new qreal[dim * dim];
    for (qint i = 0; i < dim; i++) {
        for (qint j = 0; j < dim; j++) {
            auto c             = mycoeffs(i, j);
            reals[i * dim + j] = c.real();
            imags[i * dim + j] = c.imag();
        }
    }
    initStateFromAmps(qubits, reals, imags);
    delete[] reals;
    delete[] imags;

    for (int t = 0; t < T; t++) {
        applyHRULayer(qubits, t);
        applyZZPMLayer(qubits, t, p);
        std::cout << calcEntrophy(qubits, mycoeffs, dim);
    }

    // std::cout << calcEntrophy(qubits, mycoeffs, dim);

    destroyQureg(qubits, env);
    destroyQuESTEnv(env);
    return 0;
}
