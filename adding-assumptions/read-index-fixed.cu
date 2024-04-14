
__global__ void read_index(uint* indices, float4 *newVel, float3 vel, float* data) {
    uint index = threadIdx.x * blockDim.x + blockIdx.x;
    uint originalIndex = indices[index];
    __assume(__distinct_int(originalIndex));
    newVel[originalIndex] = make_float4(vel, data[index]);
}
