
__global__ void read_index(uint* indices, float4 *newVel, float3 vel, float* data) {
    uint index = threadIdx.x * blockDim.x + blockIdx.x;
    uint originalIndex = indices[index];
    newVel[originalIndex] = make_float4(vel, data[index]);
}
