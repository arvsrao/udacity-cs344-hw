//Udacity HW 4
//Radix Sorting

#include "reference_calc.cpp"
#include "utils.h"

/* Red Eye Removal
 ===============

 For this assignment we are implementing red eye removal.  This is
 accomplished by first creating a score for every pixel that tells us how
 likely it is to be a red eye pixel.  We have already done this for you - you
 are receiving the scores and need to sort them in ascending order so that we
 know which pixels to alter to remove the red eye.

 Note: ascending order == smallest to largest

 Each score is associated with a position, when you sort the scores, you must
 also move the positions accordingly.

 Implementing Parallel Radix Sort with CUDA
 ==========================================

 The basic idea is to construct a histogram on each pass of how many of each
 "digit" there are.   Then we scan this histogram so that we know where to put
 the output of each digit.  For example, the first 1 must come after all the
 0s so we have to know how many 0s there are to be able to start moving 1s
 into the correct position.

 1) Histogram of the number of occurrences of each digit
 2) Exclusive Prefix Sum of Histogram
 3) Determine relative offset of each digit
 For example [0 0 1 1 0 0 1]
 ->  [0 1 0 1 2 3 2]
 4) Combine the results of steps 2 & 3 to determine the final
 output location for each element and move it there

 LSB Radix sort is an out-of-place sort and you will need to ping-pong values
 between the input and output buffers we have provided.  Make sure the final
 sorted results end up in the output buffer!  Hint: You may need to do a copy
 at the end.

 */

__global__
void binMask(unsigned int* const src, 
            unsigned char* const output, 
            const unsigned int mask, 
            const size_t N) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // index on the src data.
    if ( idx >= N ) return; 

    output[idx] = ( src[idx] & mask ) == mask;
}

__global__
void deepCopy(unsigned int * const input, 
              unsigned int * const output, 
              const size_t N) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x; // index on the src data.
    if ( idx >= N ) return; 

    output[idx] = input[idx];
}

__global__
void moveKernel(unsigned int* const inputVals,
                unsigned int* const inputPos, 
                unsigned int* const outputVals,
                unsigned int* const outputPos,  
                unsigned int* const sums,
                unsigned int* const buffer, 
                unsigned int* const scannedSums,
                unsigned char* const mask, 
                const size_t N) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x; // index on the src data.
    if ( idx >= N ) return; 

    __shared__ unsigned int hist; 
    if ( threadIdx.x == 0 ) hist = scannedSums[gridDim.x-1] + sums[gridDim.x-1];
    __syncthreads();

    int newIdx = ( mask[idx] ) ? buffer[idx + N] + scannedSums[blockIdx.x + gridDim.x] + hist : buffer[idx] + scannedSums[blockIdx.x];    
    outputVals[newIdx] = inputVals[idx];
    outputPos[newIdx] = inputPos[idx];
}

__global__
void intermediateScan(unsigned int * const input, unsigned int * const output, const size_t zero, const size_t N ) {

    __shared__ unsigned int temp[1024];
    unsigned int level, next_level, j;
    
    temp[threadIdx.x] = ( threadIdx.x > 0 && threadIdx.x < N ) ? input[threadIdx.x - 1 + zero * N] : 0;
    __syncthreads();

    // block scan; must be local to the block.
    j = 0;
    for ( int offset = 1; offset < blockDim.x; offset *= 2) {
        level = (j % 2) * blockDim.x;
        next_level = ((j + 1) % 2) * blockDim.x;
        if ( offset <= threadIdx.x ) // where 'original' input is.
            temp[threadIdx.x + next_level] = temp[threadIdx.x + level] + temp[threadIdx.x - offset + level];
        else
            temp[threadIdx.x + next_level] = temp[threadIdx.x + level]; //copy over elements with idx < offset
        j++;
        __syncthreads();
    }

    if ( threadIdx.x >= N ) return;
    output[threadIdx.x + zero * N] = temp[threadIdx.x + (j % 2) * blockDim.x];
}

__global__
void localScan(unsigned char * const input,
               unsigned int * const buffer, 
               unsigned int * const sums, 
               const size_t zero, 
               const size_t N ) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x; // index on the src data.
    __shared__ unsigned int temp[1024];
    __shared__ unsigned int lastValue;

    unsigned int level, next_level, j;

    temp[threadIdx.x] = ( threadIdx.x > 0 && idx < N ) ? (unsigned int) zero==input[idx - 1] : 0;
    __syncthreads();

    if ( threadIdx.x == blockDim.x - 1 )
        lastValue = ( idx < N ) ? (unsigned int) zero==input[idx] : (unsigned int) zero==input[N - 1];
    __syncthreads();

    // block scan; must be local to the block.
    j = 0;
    for ( int offset = 1; offset < blockDim.x; offset *= 2) {
        level = (j % 2) * blockDim.x;
        next_level = ((j + 1) % 2) * blockDim.x;
        if ( offset <= threadIdx.x ) // where 'original' input is.
            temp[threadIdx.x + next_level] = temp[threadIdx.x + level] + temp[threadIdx.x - offset + level];
        else
            temp[threadIdx.x + next_level] = temp[threadIdx.x + level]; //copy over elements with idx < offset
        j++;
        __syncthreads();
    }

    if ( idx >= N ) return;
    buffer[idx + zero * N] = temp[threadIdx.x + (j % 2) * blockDim.x];

    if ( threadIdx.x != 0 ) return;
    sums[blockIdx.x + zero * gridDim.x] = temp[blockDim.x - 1 + (j % 2) * blockDim.x] + lastValue; 
}

/*
 * general idea is compact (filter ) the src array with predicates 1 -> 0 and 1-> 0 in combination
 * with exclusive sum scan to get the scatter addresses.
 */
void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems) {

    const size_t numBins = 2;
    const size_t numBitsInInt = 8 * sizeof(unsigned int);
    const size_t xlength = 512;
    const size_t xDim = ceil(numElems / (float) xlength );
     
    // d_hist is the count of zero's for each bit precision.
    // d_sums array is an auxillary scan array for intermediate scan results. 
    unsigned int *d_sums, *d_scannedSums, *d_buffer;
    unsigned char* d_mask;

    unsigned int *vals_src = d_inputVals;
    unsigned int *pos_src  = d_inputPos;
    unsigned int *vals_dst = d_outputVals;
    unsigned int *pos_dst  = d_outputPos;

    checkCudaErrors(cudaMalloc( &d_mask, sizeof(unsigned char) * numElems ));
    checkCudaErrors(cudaMalloc( &d_sums, sizeof(unsigned int) * 2 * xDim ));
    checkCudaErrors(cudaMalloc( &d_scannedSums, sizeof(unsigned int) * 2 * xDim ));
    checkCudaErrors(cudaMalloc( &d_buffer, sizeof(unsigned int) * 2 * numElems ));
       
    // histogram values on the columns are invariant to how numbers are sorted.
    // so its totally valid for use to precompute the histograms
    
    for ( int i = 0; i < numBitsInInt; i++) {

        checkCudaErrors(cudaMemset( d_sums, 0, sizeof(unsigned int) * 2 * xDim ));
        checkCudaErrors(cudaMemset( d_scannedSums, 0, sizeof(unsigned int) * 2 * xDim ));
        checkCudaErrors(cudaMemset( d_buffer, 0, sizeof(unsigned int) * 2 * numElems ));
        checkCudaErrors(cudaMemset( d_mask, 0, sizeof(unsigned char) * numElems ));

        binMask<<< xDim, xlength >>>( vals_src, d_mask, 1 << i, numElems);

        for( unsigned int j = 0 ; j < numBins; j++ ) {
            localScan<<< xDim, xlength >>>(d_mask, d_buffer, d_sums, j, numElems);
            intermediateScan<<< 1 , xlength >>>(d_sums, d_scannedSums, j, xDim);
        }
        moveKernel<<< xDim, xlength >>>(vals_src, pos_src, vals_dst, pos_dst, d_sums, d_buffer, d_scannedSums, d_mask, numElems);       
        
        std::swap(vals_dst, vals_src);
        std::swap(pos_dst, pos_src);
    }

    deepCopy<<< xDim, xlength >>>(d_inputVals, d_outputVals, numElems);
    deepCopy<<< xDim, xlength >>>(d_inputPos, d_outputPos, numElems); 

    checkCudaErrors(cudaFree(d_mask));
    checkCudaErrors(cudaFree(d_sums));
    checkCudaErrors(cudaFree(d_scannedSums));
    checkCudaErrors(cudaFree(d_buffer));
}