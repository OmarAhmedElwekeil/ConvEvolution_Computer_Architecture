#ifndef NAIIVE_H
#define NAIIVE_H

Volume conv3d(const Volume& input, const Volume& kernel) {
    cout << "With naiive" << endl;

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
                        for (int kw = 0; kw < kW; ++kw) {
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
