#ifndef SIMD_H
#define SIMD_H

#include <vector>
#include <immintrin.h>  // For AVX intrinsics

using namespace std;

typedef vector<vector<vector<float>>> Volume;

Volume conv3dUnrolledSIMD(const Volume& input, const Volume& kernel) {
    cout << "With SIMD" << endl;

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

    for (int d = 0; d < outD; ++d) {
        for (int i = 0; i < outH; ++i) {
            for (int j = 0; j < outW; ++j) {
                float sum = 0.0f;

                for (int kd = 0; kd < kD; ++kd) {
                    for (int kh = 0; kh < kH; ++kh) {
                        int kw = 0;

                        for (; kw + 7 < kW; kw += 8) {
                            __m256 input_vec = _mm256_set_ps(
                                input[d + kd][i + kh][j + kw + 7],
                                input[d + kd][i + kh][j + kw + 6],
                                input[d + kd][i + kh][j + kw + 5],
                                input[d + kd][i + kh][j + kw + 4],
                                input[d + kd][i + kh][j + kw + 3],
                                input[d + kd][i + kh][j + kw + 2],
                                input[d + kd][i + kh][j + kw + 1],
                                input[d + kd][i + kh][j + kw + 0]
                            );

                            __m256 kernel_vec = _mm256_set_ps(
                                kernel[kd][kh][kw + 7],
                                kernel[kd][kh][kw + 6],
                                kernel[kd][kh][kw + 5],
                                kernel[kd][kh][kw + 4],
                                kernel[kd][kh][kw + 3],
                                kernel[kd][kh][kw + 2],
                                kernel[kd][kh][kw + 1],
                                kernel[kd][kh][kw + 0]
                            );

                            __m256 mul = _mm256_mul_ps(input_vec, kernel_vec);
                            __m256 hsum = _mm256_hadd_ps(mul, mul);
                            hsum = _mm256_hadd_ps(hsum, hsum);
                            float partial[8];
                            _mm256_storeu_ps(partial, hsum);
                            sum += partial[0] + partial[4];
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
