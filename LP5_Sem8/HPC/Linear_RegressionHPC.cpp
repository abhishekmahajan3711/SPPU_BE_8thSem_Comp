#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <omp.h>
#define N 1000000
#define TEST_SIZE 100000
using namespace std;
using namespace std::chrono;
// Sequential y=ax+b
void linear_regression_sequential(float* x, float* y, int n, float& as, float& bs) {
    float sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0;
    for (int i = 0; i < n; ++i) {
        sum_x += x[i];
        sum_y += y[i];
        sum_xy += x[i] * y[i];
        sum_xx += x[i] * x[i];
    }
    as = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
    bs = (sum_y - as * sum_x) / n;
}
// Parallel 
void linear_regression_openmp(float* x, float* y, int n, float& ap, float& bp) {
    float sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0;
    #pragma omp parallel for reduction(+:sum_x, sum_y, sum_xy, sum_xx)
    for (int i = 0; i < n; ++i) {
        sum_x += x[i];
        sum_y += y[i];
        sum_xy += x[i] * y[i];
        sum_xx += x[i] * x[i];
    }
    ap = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
    bp = (sum_y - ap * sum_x) / n;
}

// MSE 
float compute_mse(float* x_test, float* y_test, int size, float w, float b) {
    float mse = 0;
    #pragma omp parallel for reduction(+:mse)
    for (int i = 0; i < size; ++i) {
        float y_pred = w * x_test[i] + b;
        mse += (y_pred - y_test[i]) * (y_pred - y_test[i]);
    }
    return mse / size;
}

int main() {
    float* x = new float[N];
    float* y = new float[N];
    srand(time(0));
    // Generate data for y = 5x + some small noise
    for (int i = 0; i < N; ++i) {
        x[i] = static_cast<float>(rand()) / RAND_MAX * 100.0f;
        y[i] = 5.0f * x[i] + static_cast<float>(rand()) / RAND_MAX; // Linear data
    }

    float as, bs, ap, bp;
    // Sequential 
    auto start_seq = high_resolution_clock::now();
    linear_regression_sequential(x, y, N, as, bs);
    auto end_seq = high_resolution_clock::now();
    double time_seq = duration<double>(end_seq - start_seq).count();
    // Parallel 
    auto start_omp = high_resolution_clock::now();
    linear_regression_openmp(x, y, N, ap, bp);
    auto end_omp = high_resolution_clock::now();
    double time_omp = duration<double>(end_omp - start_omp).count();

    float* x_test = new float[TEST_SIZE];
    float* y_test = new float[TEST_SIZE];

    for (int i = 0; i < TEST_SIZE; ++i) {
        x_test[i] = static_cast<float>(rand()) / RAND_MAX * 100.0f;
        y_test[i] = 5.0f * x_test[i] + static_cast<float>(rand()) / RAND_MAX;
    }
    float mse_seq = compute_mse(x_test, y_test, TEST_SIZE, as, bs);
    float mse_omp = compute_mse(x_test, y_test, TEST_SIZE, ap, bp);
    double speedup = time_seq / time_omp;
    
    cout << fixed;
    cout << "Sequential Time : " << time_seq << endl;
    cout << "Parallel Time   : " << time_omp << endl;
    cout << "Speedup factor  : " << speedup << endl;
   
    cout<<endl<<endl;
    cout<<"Predicted Equation "<<endl;
    cout<<"For Sequential : "<<"y = "<<as<<"x + "<<bs<<endl;
    cout<<"For Parallel   : "<<"y = "<<ap<<"x + "<<bp;

    cout<<endl<<endl;
    cout << "\nComparison :" << endl;
    cout << "X\t\tReal Y    \tSequential Y\tParallel Y" << endl;
    for (int i = 0; i < 5; ++i) {
        float y_pred_seq = as * x_test[i] + bs;
        float y_pred_omp = ap * x_test[i] + bp;
        cout << x_test[i] << "\t" << y_test[i] << "\t"
             << y_pred_seq << "\t" << y_pred_omp << endl;
    }
    cout<<endl<<endl;
    cout<<"MSE"<<endl;
    cout << "For Sequential : " << mse_seq << endl;
    cout << "For Parallel   : " << mse_omp << endl;

    delete[] x;
    delete[] y;
    delete[] x_test;
    delete[] y_test;

    return 0;
}
