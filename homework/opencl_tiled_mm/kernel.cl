#define TILE_WIDTH 1
__kernel void matrixMultiply(
    __global const float *A, __global const float *B, __global float *C,
    const unsigned int numARows, const unsigned int numAColumns,
    const unsigned int numBRows, const unsigned int numBColumns,
    const unsigned int numCRows, const unsigned int numCColumns) {
  //@@ Insert code to implement matrix multiplication here

  __local float Atile[TILE_WIDTH][TILE_WIDTH];
  __local float Btile[TILE_WIDTH][TILE_WIDTH];
 
  int g_row = get_group_id(0);
  int g_col = get_group_id(1);
  int l_row = get_local_id(0);
  int l_col = get_local_id(1);
  int row = g_col * TILE_WIDTH + l_col;
  int col = g_row * TILE_WIDTH + l_row;
  float result = 0;
  for (int m = 0; m < numBRows / TILE_WIDTH; m++) {
    Atile[l_col][l_row] = A[(row * numBRows) + (m * TILE_WIDTH) + l_row];
    Btile[l_col][l_row] = B[(((m * TILE_WIDTH) + l_col) * numBRows) + col];
    barrier(CLK_LOCAL_MEM_FENCE); 
    for (int t = 0; t < TILE_WIDTH; t++) {
      result += Atile[l_col][t] * Btile[t][l_row];
    }
    barrier(CLK_LOCAL_MEM_FENCE); 
  }
  C[(row * numBRows) + col] = result;
}

