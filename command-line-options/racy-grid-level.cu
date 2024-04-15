__global__ void saxpy(int n, float a, float *x, float *y)
{
  int i = threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}
