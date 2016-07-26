#include "utils.h"
#include "reference.cpp"

const int xlength = 1024;

__device__
void swap(short x, short y, unsigned short * input) {

  short temp = 0;
  if ( input[y] < input[x] ) {
    temp = input[x];
    input[x] = input[y];
    input[y] = temp;
  }
}

__global__
void bubbleSort(const unsigned int* const data, //INPUT
                unsigned int * const output,
                int numVals) {

  short idx = blockDim.x * blockIdx.x + threadIdx.x;
  short localIdx = threadIdx.x; 

  extern __shared__ unsigned short cids[];

  if ( idx >= numVals ) return;

  cids[localIdx] = data[idx] / 10;
  __syncthreads();

  if ( localIdx % 2 == 0 ) return;

// block sort -- parallel version of bubble sort presort arrays of size 32
  for( short k = 0; k < blockDim.x/2; k++) {
    swap(localIdx - 1, localIdx, cids);
    __syncthreads();

    if ( localIdx % blockDim.x != blockDim.x - 1)
      swap(localIdx, localIdx + 1, cids);
    __syncthreads();
  }

  output[idx] = cids[localIdx];
  output[idx - 1] = cids[localIdx - 1];
}

___device__
void merge(short lo,
           short sz,
           short key, 
           unsigned char pred, 
           unsigned short * const input,
           unsigned short * const output) {

  short idx = lo;
  short beginning = idx - idx % (sz + sz);
  short start = beginning + (pred ? sz : 0);
  short end = start + sz-1;
  short mid = 0;

  if ( idx % (2*sz) == 0 ) {
    if ( key <= input[start] )
      output[beginning] = key;
    else
      output[beginning + 1] = key;
    return;
  }

  while ( start <= end ) {

    mid = start + (end - start)/2;
    if ( key > input[mid] )
        start = mid + 1;
    else if ( key < input[mid] )
        end = mid - 1;
    else
      break;
  }
 
  output[beginning + 1 + (mid % sz) + (idx % sz) ] = key; 
}

__global__
void mergeSort(const unsigned int* const input, //INPUT
               unsigned int* const histo,      //OUPUT
               int numVals) {

  short idx = blockDim.x * blockIdx.x + threadIdx.x;
  short lo = threadIdx.x;

  if ( idx >= numVals ) return; 

  __shared__ unsigned short cids[xlength];
  __shared__ unsigned short aux[xlength];

  cids[lo] = input[idx] / 10;
  __syncthreads();

  // merge sort in the kernal
  while ( sz < 64 ) {
         
      merge(lo, sz, aux[lo], lo % (sz + sz) == lo % sz, aux, cids);
      __syncthreads();
        
       aux[lo] = cids[lo];
       sz+=sz;
      __syncthreads();
    }

  histo[threadIdx.x] = cids[threadIdx.x];
}

void computeHistogram(const unsigned int* const d_vals, //INPUT
                      unsigned int* const d_histo,      //OUTPUT
                      const unsigned int numBins,
                      const unsigned int numElems) {

  unsigned int * d_temp, *d_localHist, *h_seqHist;//, *your_histo;
     
  const size_t M = 64;   
  
  checkCudaErrors(cudaMalloc( &d_temp, sizeof(unsigned int) * numBins));
  checkCudaErrors(cudaMemset( d_temp, 0, sizeof(unsigned int) * numBins));

  checkCudaErrors(cudaMalloc( &d_localHist, sizeof(unsigned int) * xlength));
  checkCudaErrors(cudaMemset( d_localHist, 0, sizeof(unsigned int) * xlength));
  
  bubbleSort<<< xlength/32, 32, sizeof(unsigned int) * 32 >>>(d_vals, d_temp, xlength);

  mergeSort<<< 1, xlength >>>(d_temp, d_localHist, xlength);
  h_seqHist = new unsigned int[xlength];

  checkCudaErrors(cudaMemcpy(h_seqHist, d_localHist , sizeof(unsigned int) * xlength, cudaMemcpyDeviceToHost ));
    
  for(short k = 0; k < M; k++ ) {
        std::cout<< "bin: " << k << " | " << h_seqHist[k] << "\n";
  }
  
  delete[] h_seqHist;
  checkCudaErrors(cudaFree(d_localHist));
}   