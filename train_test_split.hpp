

#include <random>
#include <cstring>
#include <algorithm>
#include <tuple>


std::tuple<float*, float*, float*, float*> trainTestSplit(const float *x, const float *y, const float per_of_train, const int seed, const std::size_t n) {

    std::size_t train_size = n * per_of_train;

    // result of function: tuple of train and test data
    std::tuple<float*, float*, float*, float*> train_test_data = {
        new float[train_size], // train_x
        new float[train_size], // train_y
        new float[n - train_size], // test_x
        new float[n - train_size] // test_y
    };

    // pointer on elements of tuple for more comfortable work
    float *x_train = std::get<0>(train_test_data);
    float *y_train = std::get<1>(train_test_data);
    float *x_test = std::get<2>(train_test_data);
    float *y_test = std::get<3>(train_test_data);

    // generator of random indexes
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> dist(0, train_size);

    size_t *ind = new size_t[n];

    // insert random indexes
    for(size_t i = 0; i < n; ++i)
        ind[i] = i;

    // shuffle random indexes
    std::shuffle(ind, ind + n, gen);

     for(size_t i = 0; i < train_size; ++i) {
        x_train[i] = x[ind[i]];
        y_train[i] = y[ind[i]];
    }

    for(size_t i = train_size; i < n; ++i) {
        x_test[i-train_size] = x[ind[i]];
        y_test[i-train_size] = y[ind[i]];
    }

    delete[] ind; 

    return train_test_data;

}
