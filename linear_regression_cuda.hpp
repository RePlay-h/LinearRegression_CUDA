
#ifndef LINEAR_REGRESSION_CUDA_HPP
#define LINEAR_REGRESSION_CUDA_HPP

#include <utility>
#include <cstring>
#include <memory>


// Linear regression model
class LinearModel {
public:
    explicit LinearModel(
        float *x_train,
        float *y_train,
        float *x_test,
        float *y_test,
        const std::size_t train_size,
        const std::size_t test_size
    );

    ~LinearModel();

    void fit();

    float RMSE();

    void predict(float *x, float *dst, const std::size_t n);

    inline std::pair<float, float> getCoefficients() {
        return std::make_pair(slope, intercept);
    }


private:

    // GPU and CPU memory
/*     float *h_x_train, *h_y_train;
    float *h_x_test, *h_y_test; */
    float *d_x_train, *d_y_train;
    float *d_x_test, *d_y_test;
    float *h_predictions, *d_predictions;

    // model variables
    std::size_t train_size, test_size;
    std::size_t n;

    // CUDA variables
    int block_size;
    int grid_size_train, grid_size_test;
    int shared_mem_size;

    // model's coefficients
    float slope, intercept;

    // flag to check weather you can get a RMSE-metric
    bool is_trained;
};

#endif // LINEAR_REGRESSION_CUDA_HPP