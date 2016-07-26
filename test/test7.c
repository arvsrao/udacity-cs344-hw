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

 
  __device__
  short binarySearchRightSubArray(const short key,
             short start,
             const short sz, 
             unsigned short * const input) {

    short mid = 0;
    short end = sz + sz - 1; 

    while ( start <= end ) {
        
      mid = start + (end - start)/2;

      if ( key > input[mid] )
          start = mid + 1;
      else if ( mid == end ) 
        break;
      else if ( key < input[mid] )
          end = mid;
      else if ( start < mid )
          end = mid;
      else 
        break;
    }

    return ( start >= sz + sz ) ? sz : mid - sz;
  }

  __device__
  short binarySearchLeftSubArray(const short key,
             short start,
             const short sz, 
             unsigned short * const input) {

    short mid = 0;
    short end = sz - 1;

    if ( input[sz-1] == key )
      return sz;

    while ( start <= end ) {
        
      mid = start + (end - start)/2;

      if ( key > input[mid] ) { 
          start = mid + 1;
          mid++;
      }
      else if ( mid == end ) 
        break;
      else if ( key < input[mid] )
        end = mid;
      else if ( mid <= end )
        start = mid + 1;
      }
  
    return ( start >= sz ) ? sz : mid;
  }

  __global__
  void mergeSort(const unsigned int* const input, //INPUT
                 unsigned int* const output,      //OUPUT
                 int N) {

    short idx = blockDim.x * blockIdx.x + threadIdx.x;

    if ( idx >= N ) return; 

    extern __shared__ unsigned short in[];
    unsigned short outIdx = threadIdx.x + blockDim.x;
    __shared__ unsigned short sz;

    sz = blockDim.x / 2;
    in[threadIdx.x] = input[idx];
    __syncthreads();
    
    if ( threadIdx.x < sz )
      in[outIdx] = binarySearchRightSubArray(in[threadIdx.x], sz, sz, in);
    else
      in[outIdx] = binarySearchLeftSubArray(in[threadIdx.x], 0, sz, in);
    __syncthreads();

    output[blockDim.x * blockIdx.x + in[outIdx] + (threadIdx.x % sz)] = in[threadIdx.x]; 
  }

  __global__
void yourHisto(const unsigned int* const input, //INPUT
               unsigned int* const histo,      //OUPUT
               int numVals) {

  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  short lo = threadIdx.x;

  if ( idx >= numVals ) return; 

  __shared__ unsigned short vals[xlength];
  __shared__ unsigned short localHist[xlength];
  __shared__ unsigned char ends[xlength];

  vals[lo] = input[idx]; 
  localHist[lo] = 0;
  __syncthreads();

  ends[lo] = ( lo > 0 ) ? vals[lo] != vals[lo-1] : 1;
  __syncthreads();

  if ( ends[lo] ) { 
    localHist[vals[lo]]++;
   // for ( short j = lo+1; ends[j] == 0 && j < blockDim.x; j++ ) localHist[vals[j]]++;
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

  unsigned int *d_sums, *d_out, *d_in; 
     
  checkCudaErrors(cudaMalloc( &d_sums, sizeof(unsigned int) * xlength * xSumsDim ));
  checkCudaErrors(cudaMemset( d_sums, 0, sizeof(unsigned int) * xlength * xSumsDim ));

  checkCudaErrors(cudaMalloc( &d_in , sizeof(unsigned int) * numElems));
  checkCudaErrors(cudaMemset( d_in, 0, sizeof(unsigned int) * numElems));
    
  checkCudaErrors(cudaMalloc( &d_out , sizeof(unsigned int) * numElems));
  checkCudaErrors(cudaMemset( d_out, 0, sizeof(unsigned int) * numElems));

  bubbleSort<<< numElems/256, 256, sizeof(unsigned int) * 256 >>>(d_vals, d_out, numElems);

  for ( short N = 512; N <= xlength; N*=2 ) {
    mergeSort<<< numElems/N, N, sizeof(unsigned short) * 2 * N >>>(d_out, d_in, numElems);
    std::swap(d_out, d_in);
  }

  dim3 grid1( xSumsDim, 1024 ); // xSumsDim = 40
  dim3 grid2(1, 1024); 

  yourHisto<<< xDim, xlength >>>(d_in, d_out, numElems);

      h_temp = new unsigned int[xlength];
    h_seqHist = new unsigned int[xlength];

    checkCudaErrors(cudaMemcpy(h_temp, d_temp, sizeof(unsigned int) * xlength, cudaMemcpyDeviceToHost ));
    checkCudaErrors(cudaMemcpy(h_seqHist, d_localHist, sizeof(unsigned int) * xlength, cudaMemcpyDeviceToHost ));
      
    for(short k = 0; k < xlength; k++ ) {
          std::cout<< "bin: " << k << " | " << h_temp[k] << " | " << h_seqHist[k] << "\n";
    }
    
    delete[] h_seqHist;
    checkCudaErrors(cudaFree(d_localHist));
  }   