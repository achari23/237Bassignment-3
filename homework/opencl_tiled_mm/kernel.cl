#define BLOCKSIZE 4
__kernel void matrixMultiply(
    __global const float *A, __global const float *B, __global float *C,
    const unsigned int numARows, const unsigned int numAColumns,
    const unsigned int numBRows, const unsigned int numBColumns,
    const unsigned int numCRows, const unsigned int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  int l_row = get_local_id(0);
  int l_col = get_local_id(1);

  int g_row = BLOCKSIZE*get_global_id(0) + l_row; 
  int g_col = BLOCKSIZE*get_global_id(1) + l_col; 
  __local Atile[BLOCKSIZE][BLOCKSIZE];
  __local Btile[BLOCKSIZE][BLOCKSIZE];
  float value = 0.0f;
  int num_tiles = numBRows/BLOCKSIZE;

  for (int t=0; t<num_tiles; t++) {
    int tiled_row = BLOCKSIZE*t + l_row;
    int tiled_col = BLOCKSIZE*t + l_col;
    Atile[l_col][l_row] = A[g_row*numAColumns + tiled_col];
    Btile[l_col][l_row] = B[tiled_row*numBColumns + g_col];

    barrier(CLK_LOCAL_MEM_FENCE);
        
    for (int k=0; k<BLOCKSIZE; k++) {
      value += Atile[k][l_row] * Btile[l_col][k];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    C[g_row*numBColumns + g_col] = value;
  }


}