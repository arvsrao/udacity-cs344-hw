#include "utils.h"
#include <cfloat>

__global__
void kernalMin(const float* const d_input, float* d_output, const size_t imgLength)
{
  int Idx = blockIdx.x * blockDim.x + threadIdx.x;
  int endGrid = (int) ceil(imgLength / (double) blockDim.x) - 1;
  int endIdx = blockDim.x;

  // Implement max/min using parallel reduce paradigm.  
  if ( threadIdx.x != 0 || Idx > imgLength ) return; 
      
  if (blockIdx.x == endGrid && imgLength % blockDim.x != 0) 
    endIdx = imgLength % blockDim.x;      

  float min = d_input[Idx];
  for(int i = Idx; i < endIdx; i++) min = fminf(d_input[Idx], min);

  d_output[blockIdx.x] = min;
  __syncthreads();
}

__global__
void kernalMax(const float* const d_input, float* d_output, const size_t imgLength)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
  extern __shared__ float chunk2[];

  if (idx > imgLength ) return;
  
  int endGrid = (int) ceil(imgLength / (double) blockDim.x) - 1;
  int endIdx = blockDim.x;

  chunk2[threadIdx.x] = d_input[idx];
  __syncthreads();
  
  if ( threadIdx.x != 0 ) return; 
      
  if (blockIdx.x == endGrid && imgLength % blockDim.x !=0) 
    endIdx = imgLength % blockDim.x;      

  float max = -FLT_MAX;
  for(int i = 0; i < endIdx; i++) max = fmaxf(chunk2[i], max);

  d_output[blockIdx.x] = max;
  __syncthreads();
}

__global__
void kernalHist(const float* d_input, unsigned int* hist
        , const unsigned int numBins, const float min, const float range, const size_t N)
{

    const int M = 1024;

  __shared__ unsigned int histTemp[M];
  histTemp[threadIdx.x] = 0;
      __syncthreads();
    
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) return;
    
      int bin = (int) floor( (double) numBins * (d_input[idx] - min) / range );
      atomicAdd(&histTemp[bin], 1);
  __syncthreads();

  atomicAdd(&hist[threadIdx.x], histTemp[threadIdx.x]);
}

__global__
void scan(const unsigned int* input, unsigned int* output, const size_t N)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x; 
  if (idx >= N) return;

  // array shared by all threads
  extern __shared__ unsigned int temp[];

    temp[idx] = (idx > 0) ? input[idx -1] : 0;
  __syncthreads();
  
    temp[idx+N] = (idx > 0) ? input[idx -1] : 0;
    __syncthreads();
        
  int j = 0;
  for(int offset = 1; offset < N; offset *=2)
        {
      if( offset <= idx )
        temp[idx + ((j+1) % 2)*N] += temp[idx - offset + (j % 2) * N];
          else
        temp[idx + ((j+1) % 2)*N] = temp[idx + (j % 2) * N];
      temp[idx + (j % 2)*N] = temp[idx + ((j+1) % 2)*N];
      j++;
          __syncthreads();
      }
  output[idx] = temp[idx + (j % 2)*N];
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  const int BLOCKLENGTH = 256;
    const dim3 blockSize(BLOCKLENGTH, 1);
    const dim3 blockSize_cdf(1024,1);
    const dim3 gridSize_hist( ceil( numRows * numCols / float(blockSize_cdf.x)), 1);

    const dim3 gridSize( ceil( numRows * numCols / float(blockSize.x)), 1);
    const dim3 gridSize_cdf( ceil( numBins / float(blockSize_cdf.x)), 1);

    //lauch kernal that finds the max & min using Reduce
  float *d_min, *d_max;
  unsigned int *d_hist;
  float *h_min = new float[512];
  float *h_max = new float[512];

  unsigned int *h_hist = new unsigned int[numBins];
    unsigned int *h_cdf = new unsigned int[numBins];
  const size_t nBytes = sizeof(float) * 512;

  checkCudaErrors(cudaMalloc(&d_min, nBytes));
  checkCudaErrors(cudaMalloc(&d_max, nBytes));
  checkCudaErrors(cudaMalloc(&d_hist, sizeof(unsigned int) * numBins ));
    checkCudaErrors(cudaMemset(d_hist, 0, sizeof(unsigned int) * numBins ));

  kernalMin<<<gridSize, blockSize>>>(d_logLuminance, d_min, numRows * numCols);
  kernalMax<<<gridSize, blockSize, sizeof(float)*blockSize.x >>>(d_logLuminance, d_max, numRows * numCols);

  checkCudaErrors(cudaMemcpy(h_min, d_min, nBytes, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_max, d_max, nBytes, cudaMemcpyDeviceToHost));

  min_logLum = h_min[0];
  max_logLum = h_max[0];
    
  for(int i = 0; i < gridSize.x ; i++ )
  {
    min_logLum = std::min(h_min[i], min_logLum);
    max_logLum = std::max(h_max[i], max_logLum);
  } 

    // compute range & histogram
  float range = max_logLum - min_logLum;

  std::cout<< "minimum is: " << min_logLum <<"\n";
  std::cout<< "maximum is: " << max_logLum <<"\n";
  std::cout<< "range is: " << range << "\n";  
  std::cout<< "number of bins: " << numBins <<"\n";
  std::cout<< "histogram binning example: "<< floor( (double) numBins * (0.4205 - min_logLum) / range ) <<"\n"; 
  
    kernalHist<<<gridSize_hist, blockSize_cdf, sizeof(unsigned int)*numBins >>>(d_logLuminance, d_hist, numBins, min_logLum, range, numRows * numCols); 
    scan<<<gridSize_cdf, blockSize_cdf, 2*sizeof(unsigned int) * numBins>>>(d_hist, d_cdf, numBins);    
}