__kernel void matrixMultiply(
    __global const float *A, __global const float *B, __global float *C,
    const unsigned int numARows, const unsigned int numAColumns,
    const unsigned int numBRows, const unsigned int numBColumns,
    const unsigned int numCRows, const unsigned int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  
  int g_row = get_global_id(1); // rows of result
  int g_col = get_global_id(0); // columns of result

  float value = 0.0;
  for(int i = 0; i < numBRows; i++) {
    value += A[i*numARows + g_row] * B[g_col*numBRows + i];
  }
  C[g_col*numARows + g_row] = value;
}

