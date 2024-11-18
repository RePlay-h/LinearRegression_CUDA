
#include "linear_regression_cuda.hpp"
#include "train_test_split.hpp"

#include <random>
#include <iostream>

#define N 1'000'000

#define A 25 
#define B 50 
#define MIN_X -10 
#define MAX_X 10 
#define NOISE_LEVEL 50


using namespace std;

void genDataset(float *x, float *y) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> disX(MIN_X, MAX_X);
    normal_distribution<> disNoise(0, NOISE_LEVEL);

    for (int i = 0; i < N; ++i) {
        x[i] = static_cast<float>(disX(gen));
        y[i] = A * x[i] + B + static_cast<float>(disNoise(gen));
    }
}

int main() {
    float x[N], y[N];

    float x_[5] = {1.004312, 2.33242, 12.213213, 4.231312, 5.21123};
    float *y_ = new float[5];

    genDataset(x, y);

    auto[x_train, y_train, x_test, y_test] = trainTestSplit(x, y, 0.7, 101, N);

    size_t train_size = 0.7 * N;
    size_t test_size = N - train_size;

    LinearModel model(x_train, y_train, x_test, y_test, train_size, test_size);

    model.fit();

    model.predict(x_, y_, 5);

    auto[slope, intercept] = model.getCoefficients();

    std::cout << "\nRMSE: " << model.RMSE() << '\n';
    std::cout << "\nCoefficients: " << slope << ' ' << intercept << '\n';
    std::cout << "\nObservations: \n";
    std::cout << "\n  X: ";

    for(size_t i = 0; i < 5; ++i) 
        std::cout << x_[i] << ' ';
    std::cout << '\n';

    std::cout << "  Y: ";
     for(size_t i = 0; i < 5; ++i) 
        std::cout << y_[i] << ' ';
    std::cout << '\n'; 
}