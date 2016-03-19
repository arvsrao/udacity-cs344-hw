  #include "utils.h"
  #include "reference.cpp"

  const short xlength = 1024;
  const short xMax = 14*6*256;

  __device__
  void swap(unsigned int x, unsigned int y, unsigned int * input) {

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
                  int N) {

    short idx = blockDim.x * blockIdx.x + threadIdx.x;
    short idy = blockDim.y * blockIdx.y;

    extern __shared__ unsigned int cids[];

    if ( idx + idy * xMax >= N) return;

    cids[threadIdx.x] = data[idx + idy * xMax];
    __syncthreads();

  // block sort -- parallel version of bubble sort presort arrays of size 32
    short p = 0;
    for( short k = 0; k < blockDim.x; k++) {
        if ( threadIdx.x % 2 == p % 2 && threadIdx.x != blockDim.x - 1 ) 
            swap(threadIdx.x, threadIdx.x + 1, cids);
        p += 1;
        __syncthreads();
    }
  
    output[idx + idy * xMax] = cids[threadIdx.x];
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

    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

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

  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned short lo = threadIdx.x;
  unsigned short j, level, next_level;    
  if ( idx >= numVals ) return; 

  __shared__ unsigned short vals[xlength];
  __shared__ unsigned char ends[xlength];    
  __shared__ unsigned short localHist[xlength];

  vals[threadIdx.x] = input[idx];
  localHist[threadIdx.x] = 0; 
  __syncthreads();

  ends[threadIdx.x] = ( threadIdx.x > 0 ) ? vals[lo] != vals[lo-1] : 1;
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

    unsigned int *d_sums, * d_in, *d_out;//, *your_histo;
    
    checkCudaErrors(cudaMalloc( &d_sums, sizeof(unsigned int) * xlength * xSumsDim ));
    checkCudaErrors(cudaMemset( d_sums, 0, sizeof(unsigned int) * xlength * xSumsDim ));
     
    checkCudaErrors(cudaMalloc( &d_in , sizeof(unsigned int) * numElems));
    checkCudaErrors(cudaMemset( d_in, 0, sizeof(unsigned int) * numElems));

    checkCudaErrors(cudaMalloc( &d_out, sizeof(unsigned int) * numElems));
    checkCudaErrors(cudaMemset( d_out, 0, sizeof(unsigned int) * numElems));
    
    dim3 grid(14*12, 477); 
    bubbleSort<<< grid, 128, sizeof(unsigned int) * 128 >>>(d_vals, d_out, numElems);

    for ( short N = 256; N <= xlength; N*=2 ) {
      mergeSort<<< numElems/N, N, sizeof(unsigned short) * 2 * N >>>(d_out, d_in, numElems);
      std::swap(d_out, d_in);
    }

    yourHisto<<< xDim, xlength >>>(d_out, d_in, numElems);

    dim3 grid1( xSumsDim, 1024 ); // xSumsDim = 40
    dim3 grid2(1, 1024); 

    sum<<< grid1, 128, sizeof(unsigned int) * 128 >>>(d_in, d_sums, xDim);
    sum<<< grid2, 128, sizeof(unsigned int) * 128 >>>(d_sums, d_histo, xSumsDim );

    checkCudaErrors(cudaFree(d_sums));
    checkCudaErrors(cudaFree(d_in));
    checkCudaErrors(cudaFree(d_out));
  }   