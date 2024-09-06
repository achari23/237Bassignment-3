__kernel void matrixMultiply(
    __global const float *A, __global const float *B, __global float *C,
    const unsigned int numARows, const unsigned int numAColumns,
    const unsigned int numBRows, const unsigned int numBColumns,
    const unsigned int numCRows, const unsigned int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  
  int g_row = get_global_id(0); // rows of result
  int g_col = get_global_id(1); // columns of result

  float value = 0.0f;
  for(int i = 0; i < numBRows; i++) {
    value += A[g_row*numAColumns + i] * B[i*numBColumns + g_col];
  }
  C[g_row*numBColumns + g_col] = value;
}
