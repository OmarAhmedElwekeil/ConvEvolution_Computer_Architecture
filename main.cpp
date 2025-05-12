#include <iostream>
#include <vector>
#include <chrono>
#include "loopunrolling.h"
#include "SIMD.h"
#include "OpenMP.h"
#include "OpenMPAndSIMD.h"
#include "naiive.h"

using namespace std;

typedef vector<vector<vector<float>>> Volume;

void printSlice(const Volume& vol, int depth) {
    cout << "Slice at depth " << depth << ":\n";
    for (const auto& row : vol[depth]) {
        for (float val : row)
            cout << val << " ";
        cout << endl;
    }
}

int main() {
    const int inputSize = 20;
    const int kernelSize = 10;

    Volume input(inputSize, vector<vector<float>>(inputSize, vector<float>(inputSize)));
    float val = 1.0f;
    for (int d = 0; d < inputSize; ++d)
        for (int i = 0; i < inputSize; ++i)
            for (int j = 0; j < inputSize; ++j)
                input[d][i][j] = val++;

    Volume kernel(kernelSize, vector<vector<float>>(kernelSize, vector<float>(kernelSize, 1.0f)));

    auto start = chrono::high_resolution_clock::now();
    int number;
    cout << "Enter number to choose the technique: (1 naiive)  (2 unrolled)  (3 SIMD) (4 OpenMP)  (5 both)" << endl;
    cin >> number;
    Volume output;
    if (number ==1) output = conv3d(input, kernel);
    else if (number == 2)  output = conv3dUnrolled(input, kernel);
    else if (number == 3)  output = conv3dUnrolledSIMD(input, kernel);
    else if (number == 4)  output = conv3dUnrolledOpenMP(input, kernel);
    else  output = conv3dOpenSIMD(input, kernel);
    auto end = chrono::high_resolution_clock::now();

    chrono::duration<double> duration = end - start;
    cout << "Convolution completed in " << duration.count() * 1000 << " ms" << endl;

    printSlice(output, 0);
}
