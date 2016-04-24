
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void generateCheckerboard(float3 *pos, float3 *norm) {
    int i = threadIdx.z * (blockDim.x * blockDim.y) + threadIdx.y * (blockDim.x) + threadIdx.x;

    bool solid = true;
    if (threadIdx.x % 2 == threadIdx.z % 2) {
        solid = threadIdx.y % 2;
    }
    else {
        solid = !(threadIdx.y % 2);
    }
    float length = 1.0f / blockDim.x;
    if (!solid) {
        length = 0;
    }
    float half = 0.5f * blockDim.x;
    float3 rel = { (threadIdx.x - (half - 0.5)) * 2.0 * length,
                   (threadIdx.y - (half - 0.5)) * 2.0 * length,
                   (threadIdx.z - (half - 0.5)) * 2.0 * length, };

    pos[i * 36] = { -length, -length, -length };
    pos[i * 36 + 1] = { -length, -length, length };
    pos[i * 36 + 2] = { -length, length, length };
    pos[i * 36 + 3] = { length, length, -length };
    pos[i * 36 + 4] = { -length, -length, -length };
    pos[i * 36 + 5] = { -length, length, -length };
    pos[i * 36 + 6] = { length, -length, length };
    pos[i * 36 + 7] = { -length, -length, -length };
    pos[i * 36 + 8] = { length, -length, -length };
    pos[i * 36 + 9] = { length, length, -length };
    pos[i * 36 + 10] = { length, -length, -length };
    pos[i * 36 + 11] = { -length, -length, -length };
    pos[i * 36 + 12] = { -length, -length, -length };
    pos[i * 36 + 13] = { -length, length, length };
    pos[i * 36 + 14] = { -length, length, -length };
    pos[i * 36 + 15] = { length, -length, length };
    pos[i * 36 + 16] = { -length, -length, length };
    pos[i * 36 + 17] = { -length, -length, -length };
    pos[i * 36 + 18] = { -length, length, length };
    pos[i * 36 + 19] = { -length, -length, length };
    pos[i * 36 + 20] = { length, -length, length };
    pos[i * 36 + 21] = { length, length, length };
    pos[i * 36 + 22] = { length, -length, -length };
    pos[i * 36 + 23] = { length, length, -length };
    pos[i * 36 + 24] = { length, -length, -length };
    pos[i * 36 + 25] = { length, length, length };
    pos[i * 36 + 26] = { length, -length, length };
    pos[i * 36 + 27] = { length, length, length };
    pos[i * 36 + 28] = { length, length, -length };
    pos[i * 36 + 29] = { -length, length, -length };
    pos[i * 36 + 30] = { length, length, length };
    pos[i * 36 + 31] = { -length, length, -length };
    pos[i * 36 + 32] = { -length, length, length };
    pos[i * 36 + 33] = { length, length, length };
    pos[i * 36 + 34] = { -length, length, length };
    pos[i * 36 + 35] = { length, -length, length };
    for (int j = 0; j < 36; j++) {
        pos[i * 36 + j].x += rel.x;
        pos[i * 36 + j].y += rel.y;
        pos[i * 36 + j].z += rel.z;
    }
    norm[i * 36] = { -1, 0, 0 };
    norm[i * 36 + 1] = { -1, 0, 0 };
    norm[i * 36 + 2] = { -1, 0, 0 };
    norm[i * 36 + 3] = { 0, 0, -1 };
    norm[i * 36 + 4] = { 0, 0, -1 };
    norm[i * 36 + 5] = { 0, 0, -1 };
    norm[i * 36 + 6] = { 0, -1, 0 };
    norm[i * 36 + 7] = { 0, -1, 0 };
    norm[i * 36 + 8] = { 0, -1, 0 };
    norm[i * 36 + 9] = { 0, 0, -1 };
    norm[i * 36 + 10] = { 0, 0, -1 };
    norm[i * 36 + 11] = { 0, 0, -1 };
    norm[i * 36 + 12] = { -1, 0, 0 };
    norm[i * 36 + 13] = { -1, 0, 0 };
    norm[i * 36 + 14] = { -1, 0, 0 };
    norm[i * 36 + 15] = { 0, -1, 0 };
    norm[i * 36 + 16] = { 0, -1, 0 };
    norm[i * 36 + 17] = { 0, -1, 0 };
    norm[i * 36 + 18] = { 0, 0, 1 };
    norm[i * 36 + 19] = { 0, 0, 1 };
    norm[i * 36 + 20] = { 0, 0, 1 };
    norm[i * 36 + 21] = { 1, 0, 0 };
    norm[i * 36 + 22] = { 1, 0, 0 };
    norm[i * 36 + 23] = { 1, 0, 0 };
    norm[i * 36 + 24] = { 1, 0, 0 };
    norm[i * 36 + 25] = { 1, 0, 0 };
    norm[i * 36 + 26] = { 1, 0, 0 };
    norm[i * 36 + 27] = { 0, 1, 0 };
    norm[i * 36 + 28] = { 0, 1, 0 };
    norm[i * 36 + 29] = { 0, 1, 0 };
    norm[i * 36 + 30] = { 0, 1, 0 };
    norm[i * 36 + 31] = { 0, 1, 0 };
    norm[i * 36 + 32] = { 0, 1, 0 };
    norm[i * 36 + 33] = { 0, 0, 1 };
    norm[i * 36 + 34] = { 0, 0, 1 };
    norm[i * 36 + 35] = { 0, 0, 1 };
}

void launchGenerateCheckerboard(dim3 grid, dim3 block, float3 *pos, float3 *norm) {
    generateCheckerboard <<<grid, block>>>(pos, norm);
}