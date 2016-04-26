
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/scan.h>
#include <thrust/device_vector.h>

#include <stdio.h>

__global__ void voxelOccupancy(int* occupancy, int granularity) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    // Detect out of bounds
    if (x >= granularity || y >= granularity || z >= granularity) {
        return;
    }

    // Determine if this thread's block is solid
    int base = z * granularity * granularity + y * granularity + x;
    int solid = 1 - (threadIdx.y % 2);
    if (threadIdx.x % 2 == threadIdx.z % 2) {
        solid = threadIdx.y % 2;
    }
    occupancy[base] = solid;
}

__global__ void compactVoxels(int* compact, int* occupancy, int* scanned, int granularity) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    // Detect out of bounds
    if (x >= granularity || y >= granularity || z >= granularity) {
        return;
    }

    // Resolve elements of the compacted array
    int base = z * granularity * granularity + y * granularity + x;
    if (occupancy[base] == 1) {
        compact[scanned[base]] = base;
    }
}

__global__ void generateVoxels(int* dCompact, float3* pos, float3* norm, int granularity, int maxVert) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    int i = z * granularity * granularity + y * granularity + x;

    // Detect out of bounds
    if (x >= granularity || y >= granularity || z >= granularity || i > maxVert) {
        return;
    }

    // Compute relative spatial locations of this voxel
    int spatialIndex = dCompact[i];
    int relz = spatialIndex / (granularity * granularity);
    int rely = (spatialIndex - relz * (granularity * granularity)) / granularity;
    int relx = spatialIndex - relz * (granularity * granularity) - rely * granularity;

    float length = 1.0f / granularity;
    float half = 0.5f * granularity;
    float3 rel = { (relx - (half - 0.5f)) * 2.0f * length,
                   (rely - (half - 0.5f)) * 2.0f * length,
                   (relz - (half - 0.5f)) * 2.0f * length, };

    // Hardcoded cube data (should be replaced using instancing)
    int base = 36 * i;
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
    for (int j = 0; j < 36; j++) {
        pos[base + j].x += rel.x;
        pos[base + j].y += rel.y;
        pos[base + j].z += rel.z;
    }
}

void launchVoxelOccupancy(dim3 grid, dim3 block, int *occupancy, int granularity) {
    voxelOccupancy<<<grid, block>>>(occupancy, granularity);
}

void launchOccupancyScan(int numVoxels, int* scanned, int* occupancy) {
    thrust::exclusive_scan(thrust::device_ptr<int>(occupancy),
                           thrust::device_ptr<int>(occupancy + numVoxels),
                           thrust::device_ptr<int>(scanned));
}

void launchCompactVoxels(dim3 grid, dim3 block, int* dCompact, int* dOccupancy, int* dScanned, int granularity) {
    compactVoxels<<<grid, block>>>(dCompact, dOccupancy, dScanned, granularity);
}

void launchGenerateVoxels(dim3 grid, dim3 block, int* dCompact, float3* dVoxels, float3* dNormals, int granularity, int maxVert) {
    generateVoxels<<<grid, block>>>(dCompact, dVoxels, dNormals, granularity, maxVert);
}