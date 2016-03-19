  #include "utils.h"
  #include "reference.cpp"

  const int xlength = 1024;
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

  void computeHistogram(const unsigned int* const d_vals, //INPUT
                        unsigned int* const d_histo,      //OUTPUT
                        const unsigned int numBins,
                        const unsigned int numElems) {

    const unsigned int xDim = numElems / xlength;

    unsigned int *h_vals, *h_temp, *d_out;//, *your_histo;
     
    checkCudaErrors(cudaMalloc( &d_out, sizeof(unsigned int) * numElems));
    checkCudaErrors(cudaMemset( d_out, 0, sizeof(unsigned int) * numElems));

    dim3 grid(14*6, 477); 
    bubbleSort<<< grid, 256, sizeof(unsigned int) * 256 >>>(d_vals, d_out, numElems);

    h_temp = new unsigned int[numElems];
    h_vals = new unsigned int[numElems];
    checkCudaErrors(cudaMemcpy(h_vals, d_vals, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToHost ));
    checkCudaErrors(cudaMemcpy(h_temp, d_out, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToHost ));

    for (int k = 0; k < xlength; k++) {
      std::cout<< h_temp[k+6000*xlength] << " " << h_vals[k+6000*xlength] << "\n";
    }
   
    delete[] h_temp;
    delete[] h_vals;
    checkCudaErrors(cudaFree(d_out));
  }