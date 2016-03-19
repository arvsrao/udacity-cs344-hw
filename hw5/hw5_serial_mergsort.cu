/* Udacity HW5
   Histogramming for Speed

   The goal of this assignment is to compute a histogram
   as fast as possible.  We have simplified the problem as much as
   possible to allow you to focus solely on the histogramming algorithm.

   The input values that you need to histogram are already the exact
   bins that need to be updated.  This is unlike in HW3 where you needed
   to compute the range of the data and then do:
   bin = (val - valMin) / valRange to determine the bin.

   Here the bin is just:
   bin = val

   so the serial histogram calculation looks like:
   for (i = 0; i < numElems; ++i)
     histo[val[i]]++;

   That's it!  Your job is to make it run as fast as possible!

   The values are normally distributed - you may take
   advantage of this fact in your implementation.
*/

#include "utils.h"
#include "reference.cpp"

const int xlength = 1024;

__device__
void merge(int lo,
           int mid,  
           int hi, 
           unsigned short * const vals,
           unsigned short * const vals_aux,
           unsigned short * const cids,
           unsigned short * const cids_aux) {

  int i = lo; 
  int j = mid + 1;

  for ( int k = lo; k <= hi ; k++) {
    if ( i > mid ) {
      cids[k] = cids_aux[j++];
      vals[k] = vals_aux[j-1];
    }
    else if ( j > hi ) {
      cids[k] = cids_aux[i++];
      vals[k] = vals_aux[i-1];
    }
    else if ( cids_aux[j] < cids_aux[i] ) {
      cids[k] = cids_aux[j++];
      vals[k] = vals_aux[j-1];
    }
    else {
      cids[k] = cids_aux[i++];
      vals[k] = vals_aux[i-1];
    }
  }
}

__global__
void sum(unsigned int* const input,
         unsigned int* const output,
         const unsigned int M) {

  extern __shared__ unsigned int temp[];
  
  int idx = blockDim.x * blockIdx.x + threadIdx.x; // idx = blockIdx.x
  int idy = blockDim.y * blockIdx.y + threadIdx.y;

  if ( idy >= xlength ) return; 

  temp[threadIdx.x] = ( idx < M ) ? input[idx + idy * M] : 0; 
  __syncthreads();   

  for ( short sz = 1; sz < blockDim.x; sz += sz ) {
    if ( threadIdx.x % (sz + sz) == 0 ) temp[threadIdx.x] += temp[threadIdx.x + sz];
    __syncthreads();
  }

  if ( threadIdx.x == 0 ) output[blockIdx.x + blockIdx.y * gridDim.x] = temp[0];
}

__global__
void yourHisto(const unsigned int* const input, //INPUT
               unsigned int* const histo,      //OUPUT
               int numVals) {
  //TODO fill in this kernel to calculate the histogram as quickly as possible
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  short lo = threadIdx.x;

  if ( idx >= numVals ) return; 

  __shared__ unsigned short vals[xlength];
  __shared__ unsigned short vals_aux[xlength];
  __shared__ unsigned short cids[xlength];
  __shared__ unsigned short cids_aux[xlength];
  __shared__ unsigned char ends[xlength];
  __shared__ unsigned short localHist[xlength];

  vals[lo] = input[idx];
  cids[lo] = input[idx] / 10;
  localHist[lo] = 0;
  __syncthreads();

  // merge sort in the kernal
  for( short sz = 1; sz < blockDim.x; sz += sz ) {

    cids_aux[lo] = cids[lo]; vals_aux[lo] = vals[lo];
    __syncthreads();

    if ( lo % (2*sz) == 0 ) 
      merge(lo, lo + sz - 1, lo + sz + sz - 1, vals,vals_aux, cids, cids_aux);
    __syncthreads();
  }

  ends[lo] = ( lo > 0 ) ? cids[lo] != cids[lo-1] : 1;
  __syncthreads();

  if ( ends[lo] ) { 
    localHist[vals[lo]]++;
    for ( short j = lo+1; ends[j] == 0 && j < blockDim.x; j++ ) localHist[vals[j]]++;
  }
  __syncthreads();

  histo[blockIdx.x + lo * gridDim.x] = localHist[lo];
}

void computeHistogram(const unsigned int* const d_vals, //INPUT
                      unsigned int* const d_histo,      //OUTPUT
                      const unsigned int numBins,
                      const unsigned int numElems) {

  const unsigned int xDim = numElems / xlength;
  const unsigned int xSumsDim = ceil(xDim/ (float) 128);
  unsigned int * d_localHist, *d_sums, * your_histo; //, *your_histo;
     
  checkCudaErrors(cudaMalloc( &d_localHist, sizeof(unsigned int) * numElems ));
  checkCudaErrors(cudaMemset( d_localHist, 0, sizeof(unsigned int) * numElems ));

  checkCudaErrors(cudaMalloc( &d_sums, sizeof(unsigned int) * xlength * xSumsDim ));
  checkCudaErrors(cudaMemset( d_sums, 0, sizeof(unsigned int) * xlength * xSumsDim ));
  
  dim3 grid1( xSumsDim, xlength ); // xSumsDim = 40
  dim3 grid2(1, xlength); 

  yourHisto<<< xDim, xlength >>>(d_vals, d_localHist, numElems);
  sum<<< grid1, 128, sizeof(unsigned int) * 128 >>>(d_localHist, d_sums, xDim);
  sum<<< grid2, 128, sizeof(unsigned int) * 128 >>>(d_sums, d_histo, xSumsDim );

  checkCudaErrors(cudaFree(d_localHist));
  checkCudaErrors(cudaFree(d_sums));
}   