/*
 * Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:   
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and 
 * international Copyright laws.  
 *
 * This software and the information contained herein is PROPRIETARY and 
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and 
 * conditions of a Non-Disclosure Agreement.  Any reproduction or 
 * disclosure to any third party without the express written consent of 
 * NVIDIA is prohibited.     
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
 * OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
 * OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
 * OR PERFORMANCE OF THIS SOURCE CODE.  
 *
 * U.S. Government End Users.  This source code is a "commercial item" as 
 * that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
 * "commercial computer software" and "commercial computer software 
 * documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
 * and is provided to the U.S. Government only as a commercial end item.  
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
 * source code with only those rights set forth herein.
 */

/* Matrix multiplication: C = A * B.
 * Device code.
 */

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include <stdio.h>
#include "matrixmul.h"

#define TILE_WIDTH 32

#if 0
#define DEBUG(x...) printf(x)
#else
#define DEBUG(x...) ;
#endif


// see if i, j is a valid index of a matrix of dim m x n.
#define IS_VALID(m, n, i, j) ((i) < (m) && (j) < (n))

////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionality
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
// Matrix multiplication kernel thread specification
__global__ void MatrixMulKernel(Matrix M, Matrix N, Matrix P)
{
    // block indices
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // row and column we are working on within the matrix
    int row = by*TILE_WIDTH + ty;
    int col = bx*TILE_WIDTH + tx;

    int n_phases = (M.width + TILE_WIDTH - 1) / TILE_WIDTH;

    // now, for each phase we need to do two things:
    // 1. move relevant data into shared memory
    // 2. compute partial results

    // allocate shared memory
    // each block gets TILE_WIDTH*TILE_WIDTH threads
    __shared__ float M_entries[TILE_WIDTH][TILE_WIDTH];
    __shared__ float N_entries[TILE_WIDTH][TILE_WIDTH];

    // running sum of partial dot products
    float output = 0;

    for (int phase = 0; phase < n_phases; phase++) {
        DEBUG("phase %d\n", phase);
        // move data into shared memory
        int M_col_offset = phase*TILE_WIDTH+tx;
        if (IS_VALID(M.height, M.width, row, M_col_offset)) {
            DEBUG("valid: %d, %d\n", row, M_col_offset);
            M_entries[ty][tx] = M.elements[row*M.width + M_col_offset];
        } else {
            M_entries[ty][tx] = 0;
        }
        int N_row_offset = phase*TILE_WIDTH+ty;
        if (IS_VALID(N.height, N.width, N_row_offset, col)) {
            DEBUG("valid: %d, %d\n", N_row_offset, col);
            N_entries[ty][tx] = N.elements[N_row_offset*N.width + col];
        } else {
            N_entries[ty][tx] = 0;
        }
        // wait for all threads to move data into shared memory before continuing
        __syncthreads();

        // calculate partial dot product/result, standard algorithm...
        for (int k = 0; k < TILE_WIDTH; k++) {
            output += M_entries[ty][k] * N_entries[k][tx];
        }
        __syncthreads();
    }
    
    // only set output if we are accessing valid memory
    if (IS_VALID(M.height, N.width, row, col))
        P.elements[row*N.width+col] = output;
}

#endif // #ifndef _MATRIXMUL_KERNEL_H_
