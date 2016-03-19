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

  void computeHistogram(const unsigned int* const d_vals, //INPUT
                        unsigned int* const d_histo,      //OUTPUT
                        const unsigned int numBins,
                        const unsigned int numElems) {

    const unsigned int xDim = numElems / xlength;
    const unsigned int xSumsDim = ceil(xDim/ (float) 128);

    unsigned int *h_vals, *h_temp, * d_in, *d_out;//, *your_histo;
         
    checkCudaErrors(cudaMalloc( &d_in , sizeof(unsigned int) * numElems));
    checkCudaErrors(cudaMemset( d_in, 0, sizeof(unsigned int) * numElems));

    checkCudaErrors(cudaMalloc( &d_out, sizeof(unsigned int) * numElems));
    checkCudaErrors(cudaMemset( d_out, 0, sizeof(unsigned int) * numElems));
    
    dim3 grid(14*6, 477); 
    bubbleSort<<< grid, 256, sizeof(unsigned int) * 256 >>>(d_vals, d_out, numElems);

    for ( short N = 512; N < xlength; N*=2 ) {
      mergeSort<<< numElems/N, N, sizeof(unsigned short) * 2 * N >>>(d_out, d_in, numElems);
      std::swap(d_out, d_in);
    }

    h_temp = new unsigned int[numElems];
    h_vals = new unsigned int[numElems];
    checkCudaErrors(cudaMemcpy(h_vals, d_vals, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToHost ));
    checkCudaErrors(cudaMemcpy(h_temp, d_out, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToHost ));

    count = 0;
    for (int k = 1; k < numElems; k++) {
      if (h_temp[k] < h_temp[k-1]) {
           count++;
      }
    }
    std::cout<< "# of transistions is: " << count << "\n"; 
   
    delete[] h_temp;
    delete[] h_vals;
    checkCudaErrors(cudaFree(d_in));
    checkCudaErrors(cudaFree(d_out));
  }   