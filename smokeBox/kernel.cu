
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void generateCheckerboard(float3 *pos, float3 *norm, int granularity) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    // Detect out of bounds
    if (x >= granularity || y >= granularity || z >= granularity) {
        return;
    }

    // Determine if this thread's block is solid
    int solid = 1 - (threadIdx.y % 2);
    if (threadIdx.x % 2 == threadIdx.z % 2) {
        solid = threadIdx.y % 2;
    }

    float length = (float) solid / granularity;
    float half = 0.5f * granularity;
    float3 rel = { (x - (half - 0.5f)) * 2.0f * length,
                   (y - (half - 0.5f)) * 2.0f * length,
                   (z - (half - 0.5f)) * 2.0f * length, };

    int base = 36 * (z * granularity * granularity + y * granularity + x);
    pos[base] = { -length, -length, -length };
    pos[base + 1] = { -length, -length, length };
    pos[base + 2] = { -length, length, length };
    pos[base + 3] = { length, length, -length };
    pos[base + 4] = { -length, -length, -length };
    pos[base + 5] = { -length, length, -length };
    pos[base + 6] = { length, -length, length };
    pos[base + 7] = { -length, -length, -length };
    pos[base + 8] = { length, -length, -length };
    pos[base + 9] = { length, length, -length };
    pos[base + 10] = { length, -length, -length };
    pos[base + 11] = { -length, -length, -length };
    pos[base + 12] = { -length, -length, -length };
    pos[base + 13] = { -length, length, length };
    pos[base + 14] = { -length, length, -length };
    pos[base + 15] = { length, -length, length };
    pos[base + 16] = { -length, -length, length };
    pos[base + 17] = { -length, -length, -length };
    pos[base + 18] = { -length, length, length };
    pos[base + 19] = { -length, -length, length };
    pos[base + 20] = { length, -length, length };
    pos[base + 21] = { length, length, length };
    pos[base + 22] = { length, -length, -length };
    pos[base + 23] = { length, length, -length };
    pos[base + 24] = { length, -length, -length };
    pos[base + 25] = { length, length, length };
    pos[base + 26] = { length, -length, length };
    pos[base + 27] = { length, length, length };
    pos[base + 28] = { length, length, -length };
    pos[base + 29] = { -length, length, -length };
    pos[base + 30] = { length, length, length };
    pos[base + 31] = { -length, length, -length };
    pos[base + 32] = { -length, length, length };
    pos[base + 33] = { length, length, length };
    pos[base + 34] = { -length, length, length };
    pos[base + 35] = { length, -length, length };
    for (int j = 0; j < 36; j++) {
        pos[base + j].x += rel.x;
        pos[base + j].y += rel.y;
        pos[base + j].z += rel.z;
    }
    norm[base] = { -1, 0, 0 };
    norm[base + 1] = { -1, 0, 0 };
    norm[base + 2] = { -1, 0, 0 };
    norm[base + 3] = { 0, 0, -1 };
    norm[base + 4] = { 0, 0, -1 };
    norm[base + 5] = { 0, 0, -1 };
    norm[base + 6] = { 0, -1, 0 };
    norm[base + 7] = { 0, -1, 0 };
    norm[base + 8] = { 0, -1, 0 };
    norm[base + 9] = { 0, 0, -1 };
    norm[base + 10] = { 0, 0, -1 };
    norm[base + 11] = { 0, 0, -1 };
    norm[base + 12] = { -1, 0, 0 };
    norm[base + 13] = { -1, 0, 0 };
    norm[base + 14] = { -1, 0, 0 };
    norm[base + 15] = { 0, -1, 0 };
    norm[base + 16] = { 0, -1, 0 };
    norm[base + 17] = { 0, -1, 0 };
    norm[base + 18] = { 0, 0, 1 };
    norm[base + 19] = { 0, 0, 1 };
    norm[base + 20] = { 0, 0, 1 };
    norm[base + 21] = { 1, 0, 0 };
    norm[base + 22] = { 1, 0, 0 };
    norm[base + 23] = { 1, 0, 0 };
    norm[base + 24] = { 1, 0, 0 };
    norm[base + 25] = { 1, 0, 0 };
    norm[base + 26] = { 1, 0, 0 };
    norm[base + 27] = { 0, 1, 0 };
    norm[base + 28] = { 0, 1, 0 };
    norm[base + 29] = { 0, 1, 0 };
    norm[base + 30] = { 0, 1, 0 };
    norm[base + 31] = { 0, 1, 0 };
    norm[base + 32] = { 0, 1, 0 };
    norm[base + 33] = { 0, 0, 1 };
    norm[base + 34] = { 0, 0, 1 };
    norm[base + 35] = { 0, 0, 1 };
}

void launchGenerateCheckerboard(dim3 grid, dim3 block, float3 *pos, float3 *norm, int granularity) {
    generateCheckerboard <<<grid, block>>>(pos, norm, granularity);
}