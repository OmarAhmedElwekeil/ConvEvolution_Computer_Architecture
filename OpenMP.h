#ifndef OpenMP_H
#define OpenMP_H

#include <omp.h>
#include <iostream>
#include <vector>

using namespace std;

typedef vector<vector<vector<float>>> Volume;

Volume conv3dUnrolledOpenMP(const Volume& input, const Volume& kernel) {
    int num;
    cout << "Enter the number of threads tou wanna use: " << endl;
    cin >> num;
    omp_set_num_threads(num);

    #pragma omp master
    {
        cout << "With OpenMP using " << num << " threads" << endl;
    }

    int D = input.size();
    int H = input[0].size();
    int W = input[0][0].size();

    int kD = kernel.size();
    int kH = kernel[0].size();
    int kW = kernel[0][0].size();

    int outD = D - kD + 1;
    int outH = H - kH + 1;
    int outW = W - kW + 1;

    Volume output(outD, vector<vector<float>>(outH, vector<float>(outW, 0.0f)));

    #pragma omp parallel for collapse(3)
    for (int d = 0; d < outD; ++d) {
        for (int i = 0; i < outH; ++i) {
            for (int j = 0; j < outW; ++j) {
                float sum = 0.0f;

                for (int kd = 0; kd < kD; ++kd) {
                    for (int kh = 0; kh < kH; ++kh) {
                        int kw = 0;

                        for (; kw + 3 < kW; kw += 4) {
                            sum += input[d + kd][i + kh][j + kw]     * kernel[kd][kh][kw];
                            sum += input[d + kd][i + kh][j + kw + 1] * kernel[kd][kh][kw + 1];
                            sum += input[d + kd][i + kh][j + kw + 2] * kernel[kd][kh][kw + 2];
                            sum += input[d + kd][i + kh][j + kw + 3] * kernel[kd][kh][kw + 3];
                        }

                        for (; kw < kW; ++kw) {
                            sum += input[d + kd][i + kh][j + kw] * kernel[kd][kh][kw];
                        }
                    }
                }

                output[d][i][j] = sum;
            }
        }
    }

    return output;
}

#endif
