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
const int sumLength = 256;

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

  for ( int k = lo; k < hi ; k++) {
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
void serialHist(const unsigned int* const input,
         unsigned int* const output,
         const unsigned int N) {

  __shared__ unsigned int bins[xlength];
  __shared__ unsigned int hist[xlength];

  int idx = blockDim.x * blockIdx.x + threadIdx.x;

  bins[idx] = input[idx]; hist[idx] = 0;
  __syncthreads();

  atomicAdd(&hist[bins[idx]], 1);
  __syncthreads();

  output[idx] = hist[idx];  
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
  for( short j = 2; j <= blockDim.x; j *= 2 ) {

    cids_aux[lo] = cids[lo]; vals_aux[lo] = vals[lo]; // copy back.
    __syncthreads();

    if ( lo % j == 0 ) 
      merge(lo, lo + j/2 - 1, lo + j - 1, vals, vals_aux, cids, cids_aux);
    __syncthreads();
  }

  ends[lo] = ( lo > 0 ) ? cids[lo] != cids[lo-1] : 0;
  __syncthreads();

  if ( ends[lo] ) { 
    localHist[vals[lo]]++;
    for ( short j = lo+1; ends[j] == 0 && j < blockDim.x; j++ ) localHist[vals[j]]++;
  }
  __syncthreads();

  //histo[blockIdx.x + threadIdx.x * gridDim.x] = localHist[threadIdx.x];
  histo[threadIdx.x] = localHist[threadIdx.x];
}

void computeHistogram(const unsigned int* const d_vals, //INPUT
                      unsigned int* const d_histo,      //OUTPUT
                      const unsigned int numBins,
                      const unsigned int numElems) {

  const unsigned int xDim = numElems / xlength;
  const unsigned int xSumsDim = ceil(xDim/sumLength);
  unsigned int * d_localHist, *h_seqHist, *h_vals;
     
  checkCudaErrors(cudaMalloc( &d_localHist, sizeof(unsigned int) * 2 * numBins ));
  checkCudaErrors(cudaMemset( d_localHist, 0, sizeof(unsigned int) * 2 * numBins ));
  
  dim3 grid( xSumsDim, xlength ); // xSumsDim = 40
  dim3 blocksPerGrid(1, xlength); 

  //yourHisto<<< 1, xlength >>>(d_vals, d_localHist, numBins);
  serialHist<<<1, xlength >>>(d_vals, d_localHist, numBins);

  h_seqHist = new unsigned int[2*numBins];
  h_vals = new unsigned int[numElems];

  checkCudaErrors(cudaMemcpy(h_seqHist, d_localHist , sizeof(unsigned int) * 2 * numBins, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_vals, d_vals , sizeof(unsigned int) * numElems, cudaMemcpyDeviceToHost));
  
  for(int i=0; i < numBins; i++) h_seqHist[h_vals[i] + numBins]++;
    
  short bigger=0;
  short smaller=0;
  short equal =0;
    
  for(short k = 0; k < 1024; k++ ) {
      if ( h_seqHist[k] != 0 ) {
         // std::cout<< "bin: " << k << ", count: " << h_seqHist[k] << " " << h_seqHist[k+xlength] << "\n";
          if ( h_seqHist[k] > h_seqHist[k+xlength] )
              bigger++;
          else if ( h_seqHist[k] < h_seqHist[k+xlength] )
              smaller++;
          else
              equal++;
      }
  }
    
  std::cout<< "smaller: " << smaller << " equal: " << equal << " bigger: " << bigger << "\n"; 
  delete[] h_seqHist;
  checkCudaErrors(cudaFree(d_localHist));
}   