//Udacity HW 6
//Poisson Blending

/* Background
   ==========

   The goal for this assignment is to take one image (the source) and
   paste it into another image (the destination) attempting to match the
   two images so that the pasting is non-obvious. This is
   known as a "seamless clone".

   The basic ideas are as follows:

   1) Figure out the interior and border of the source image
   2) Use the values of the border pixels in the destination image 
      as boundary conditions for solving a Poisson equation that tells
      us how to blend the images.
   
      No pixels from the destination except pixels on the border
      are used to compute the match.

   Solving the Poisson Equation
   ============================

   There are multiple ways to solve this equation - we choose an iterative
   method - specifically the Jacobi method. Iterative methods start with
   a guess of the solution and then iterate to try and improve the guess
   until it stops changing.  If the problem was well-suited for the method
   then it will stop and where it stops will be the solution.

   The Jacobi method is the simplest iterative method and converges slowly - 
   that is we need a lot of iterations to get to the answer, but it is the
   easiest method to write.

   Jacobi Iterations
   =================

   Our initial guess is going to be the source image itself.  This is a pretty
   good guess for what the blended image will look like and it means that
   we won't have to do as many iterations compared to if we had started far
   from the final solution.

   ImageGuess_prev (Floating point)
   ImageGuess_next (Floating point)

   DestinationImg
   SourceImg

   Follow these steps to implement one iteration:

   1) For every pixel p in the interior, compute two sums over the four neighboring pixels:
      Sum1: If the neighbor is in the interior then += ImageGuess_prev[neighbor]
             else if the neighbor in on the border then += DestinationImg[neighbor]

      Sum2: += SourceImg[p] - SourceImg[neighbor]   (for all four neighbors)

   2) Calculate the new pixel value:
      float newVal= (Sum1 + Sum2) / 4.f  <------ Notice that the result is FLOATING POINT
      ImageGuess_next[p] = min(255, max(0, newVal)); //clamp to [0, 255]


    In this assignment we will do 800 iterations.
   */

#include "utils.h"
#include <thrust/host_vector.h>
#include "reference_calc.cpp"

const unsigned int NUM_COLS = 500;
const unsigned int NUM_ROWS = 333;
//const int nhbs[4] = {-1, 1, -NUM_COLS, NUM_COLS};

__device__
bool isNotWhite(const uchar4 in) {
  return (in.x + in.y + in.z < 3 * 255);
}

__global__ 
void blendImage( const unsigned char * const mask, 
                     float * const red, 
                     float * const blue,
                     float * const green, 
                     uchar4 * const dest ) {

  unsigned int row = blockIdx.x;
  unsigned int col = threadIdx.x;
  unsigned int idx = col + row * NUM_COLS; // global index 

  if ( row >= NUM_ROWS || col >= NUM_COLS ) return;

  if ( mask[idx] < 2 ) return;

  dest[idx].x = (unsigned char) red[idx];
  dest[idx].y = (unsigned char) blue[idx];
  dest[idx].z = (unsigned char) green[idx];
}

__global__
void preProcessSource(const uchar4 * const in,
                      const uchar4 * const dest,
                      const unsigned char * const mask,
                      float * const red,
                      float * const blue,
                      float * const green ) {

  unsigned int row = blockIdx.x;
  unsigned int col = threadIdx.x;
  unsigned int idx = col + row * NUM_COLS; // global index 

  if ( row >= NUM_ROWS || col >= NUM_COLS ) return;

  switch ( mask[idx] ) {
    case 0:
      red[idx] = 0; blue[idx] = 0; green[idx] = 0;
      break;
    case 1: 
      red[idx] = dest[idx].x;
      blue[idx] = dest[idx].y;
      green[idx] = dest[idx].z;
      break;
    case 2:
      red[idx] = in[idx].x;
      blue[idx] = in[idx].y;
      green[idx] = in[idx].z;
  }
}

__global__
void separateIntoChannels(const uchar4 * const in,
                          float * const red,
                          float * const blue,
                          float * const green ) {

  unsigned int row = blockIdx.x;
  unsigned int col = threadIdx.x;
  unsigned int idx = col + row * NUM_COLS; // global index 

  if ( row >= NUM_ROWS || col >= NUM_COLS ) return;

  red[idx] = in[idx].x;
  blue[idx] = in[idx].y;
  green[idx] = in[idx].z;
}

__global__
void makeMask( const uchar4* const in,
               unsigned char * const out) {
  // the source image is addresses in column-major order

  __shared__ unsigned char img[3 * NUM_COLS];
  __shared__ short neighbors[4]; 

  unsigned int row = blockIdx.x;
  unsigned int col = threadIdx.x;
  unsigned int idx = col + row * NUM_COLS; // global index 
  unsigned int sIdx = col + NUM_COLS;  // index in shared array

  if ( row >= NUM_ROWS || col >= NUM_COLS ) return;

  neighbors[0] = -1;
  neighbors[1] = 1;
  neighbors[2] = -NUM_COLS;
  neighbors[3] = NUM_COLS;
    
  img[sIdx - NUM_COLS] = ( row > 0 ) ? 2 * isNotWhite( in[idx - NUM_COLS] ) : 0;
  img[sIdx] = isNotWhite( in[idx] ) ? 2 : 0;
  img[sIdx + NUM_COLS] = ( row + 1 < NUM_ROWS ) ? 2 * isNotWhite( in[idx + NUM_COLS] ) : 0;
  __syncthreads();

  if ( img[sIdx] < 1 ) return;

  unsigned char acc = 0;
  for ( short i = 0; i < 4 ; i++ ) acc += img[sIdx + neighbors[i]];

  out[idx] = ( acc < 8 ) ? 1 : 2;
}

__global__
void computeMetric( const float * const in,
                    const unsigned char * const mask,
                    float * const out ) {

 __shared__ unsigned char img[3 * NUM_COLS];
 __shared__ short neighbors[4]; 

  unsigned int row = blockIdx.x;
  unsigned int col = threadIdx.x;
  unsigned int idx = col + row * NUM_COLS; // global index 
  unsigned int sIdx = col + NUM_COLS;  // index in shared array
 
  if ( row >= NUM_ROWS || col >= NUM_COLS ) return;

  neighbors[0] = -1;
  neighbors[1] = 1;
  neighbors[2] = -NUM_COLS;
  neighbors[3] = NUM_COLS;
 
  img[sIdx - NUM_COLS] = ( row > 0 ) ? in[idx - NUM_COLS] : 0;
  img[sIdx] = in[idx];
  img[sIdx + NUM_COLS] = ( row + 1 < NUM_ROWS ) ? in[idx + NUM_COLS] : 0;
  __syncthreads();

  if ( mask[idx] < 2 ) return;

  float acc = 4.f * img[sIdx];  // Sum2: += SourceImg[p] - SourceImg[neighbor]   (for all four neighbors)
  for ( short i = 0; i < 4; i++ ) acc -= img[sIdx + neighbors[i]];

  out[idx] = acc;
}

__global__
void computeIterate( const unsigned char * const mask, 
                     float * const in, 
                     float * const out,
                     float * const metric) {

 __shared__ float img[3 * NUM_COLS];
 __shared__ short neighbors[4]; 

  unsigned int row = blockIdx.x;
  unsigned int col = threadIdx.x;
  unsigned int idx = col + row * NUM_COLS; // global index 
  unsigned int sIdx = col + NUM_COLS;  // index in shared array
    
  neighbors[0] = -1;
  neighbors[1] = 1;
  neighbors[2] = -NUM_COLS;
  neighbors[3] = NUM_COLS;

  if ( row >= NUM_ROWS || col >= NUM_COLS ) return;

  img[sIdx - NUM_COLS] = ( row > 0 ) ? in[idx - NUM_COLS] : 0;
  img[sIdx] = in[idx];
  img[sIdx + NUM_COLS] = ( row + 1 < NUM_ROWS ) ? in[idx + NUM_COLS] : 0;
  __syncthreads();

  if ( mask[idx] == 1) out[idx] = img[sIdx];
  if ( mask[idx] < 2 ) return;

  float acc = 0.f;
  for ( short i = 0; i < 4 ; i++ ) acc += img[sIdx + neighbors[i]];

  out[idx] = fminf(255.f, fmaxf(0.f, (acc + metric[idx]) / 4.f));  
}


void your_blend(const uchar4* const h_sourceImg,  //IN
                const size_t numRowsSource, const size_t numColsSource,
                const uchar4* const h_destImg, //IN
                uchar4* const h_blendedImg) //OUT 
{
  /* To Recap here are the steps you need to implement
  
     1) Compute a mask of the pixels from the source image to be copied
        The pixels that shouldn't be copied are completely white, they
        have R=255, G=255, B=255.  Any other pixels SHOULD be copied.

     2) Compute the interior and border regions of the mask.  An interior
        pixel has all 4 neighbors also inside the mask.  A border pixel is
        in the mask itself, but has at least one neighbor that isn't.

     3) Separate out the incoming image into three separate channels

     4) Create two float(!) buffers for each color channel that will
        act as our guesses.  Initialize them to the respective color
        channel of the source image since that will act as our intial guess.

     5) For each color channel perform the Jacobi iteration described 
        above 800 times.

     6) Create the output image by replacing all the interior pixels
        in the destination image with the result of the Jacobi iterations.
        Just cast the floating point values to unsigned chars since we have
        already made sure to clamp them to the correct range.

      Since this is final assignment we provide little boilerplate code to
      help you.  Notice that all the input/output pointers are HOST pointers.

      You will have to allocate all of your own GPU memory and perform your own
      memcopies to get data in and out of the GPU memory.

      Remember to wrap all of your calls with checkCudaErrors() to catch any
      thing that might go wrong.  After each kernel call do:

      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

      to catch any errors that happened while executing the kernel.
      
      the # of rows: 333
      the n# cols: 500
  */

    const unsigned int srcSize = numRowsSource * numColsSource;
    uchar4 * d_sourceImg, *d_destImg;
    unsigned char *d_mask;
    float *d_blended_red_in, *d_blended_red_out, *d_blended_green_in, 
          *d_blended_green_out, *d_blended_blue_in, *d_blended_blue_out, 
          *d_dest_red, *d_dest_green, *d_dest_blue, *d_metric_red,
          *d_metric_blue, *d_metric_green;
    
    checkCudaErrors(cudaMalloc( &d_sourceImg, sizeof(uchar4)  * srcSize ));
    checkCudaErrors(cudaMemset( d_sourceImg, 0, sizeof(uchar4)  * srcSize ));
    
    checkCudaErrors(cudaMalloc( &d_destImg, sizeof(uchar4)  * srcSize ));
    checkCudaErrors(cudaMemset( d_destImg, 0, sizeof(uchar4)  * srcSize ));
    
    checkCudaErrors(cudaMalloc( &d_mask, sizeof(unsigned char) * srcSize ));
    checkCudaErrors(cudaMemset( d_mask, 0, sizeof(unsigned char) * srcSize ));

    checkCudaErrors(cudaMalloc( &d_blended_red_in, sizeof(float) * srcSize ));
    checkCudaErrors(cudaMemset( d_blended_red_in, 0, sizeof(float) * srcSize ));

    checkCudaErrors(cudaMalloc( &d_blended_green_in, sizeof(float) * srcSize ));
    checkCudaErrors(cudaMemset( d_blended_green_in, 0, sizeof(float) * srcSize ));

    checkCudaErrors(cudaMalloc( &d_blended_blue_in, sizeof(float) * srcSize ));
    checkCudaErrors(cudaMemset( d_blended_blue_in, 0, sizeof(float) * srcSize ));

    checkCudaErrors(cudaMalloc( &d_blended_red_out, sizeof(float) * srcSize ));
    checkCudaErrors(cudaMemset( d_blended_red_out, 0, sizeof(float) * srcSize ));

    checkCudaErrors(cudaMalloc( &d_blended_green_out, sizeof(float) * srcSize ));
    checkCudaErrors(cudaMemset( d_blended_green_out, 0, sizeof(float) * srcSize ));

    checkCudaErrors(cudaMalloc( &d_blended_blue_out, sizeof(float) * srcSize ));
    checkCudaErrors(cudaMemset( d_blended_blue_out, 0, sizeof(float) * srcSize ));

    checkCudaErrors(cudaMalloc( &d_dest_red, sizeof(float) * srcSize ));
    checkCudaErrors(cudaMemset( d_dest_red, 0, sizeof(float) * srcSize ));

    checkCudaErrors(cudaMalloc( &d_dest_green, sizeof(float) * srcSize ));
    checkCudaErrors(cudaMemset( d_dest_green, 0, sizeof(float) * srcSize ));

    checkCudaErrors(cudaMalloc( &d_dest_blue, sizeof(float) * srcSize ));
    checkCudaErrors(cudaMemset( d_dest_blue, 0, sizeof(float) * srcSize ));

    checkCudaErrors(cudaMalloc( &d_metric_red, sizeof(float) * srcSize ));
    checkCudaErrors(cudaMemset( d_metric_red, 0, sizeof(float) * srcSize ));

    checkCudaErrors(cudaMalloc( &d_metric_blue, sizeof(float) * srcSize ));
    checkCudaErrors(cudaMemset( d_metric_blue, 0, sizeof(float) * srcSize ));

    checkCudaErrors(cudaMalloc( &d_metric_green, sizeof(float) * srcSize ));
    checkCudaErrors(cudaMemset( d_metric_green, 0, sizeof(float) * srcSize ));

    checkCudaErrors(cudaMemcpy(d_sourceImg, h_sourceImg, sizeof(uchar4)  * srcSize, cudaMemcpyHostToDevice ));
    checkCudaErrors(cudaMemcpy(d_destImg, h_destImg, sizeof(uchar4)  * srcSize, cudaMemcpyHostToDevice ));

    // makeMask & separateIntoChannels can be done simultanouesly.
    makeMask<<< 333, 512 >>>(d_sourceImg, d_mask);
    separateIntoChannels<<< 333, 512 >>>(d_sourceImg, d_blended_red_in, d_blended_blue_in, d_blended_green_in);

    computeMetric<<< 333, 512 >>>(d_blended_red_in, d_mask, d_metric_red);
    computeMetric<<< 333, 512 >>>(d_blended_blue_in, d_mask, d_metric_blue);
    computeMetric<<< 333, 512 >>>(d_blended_green_in, d_mask, d_metric_green);

    preProcessSource<<< 333, 512 >>>(d_sourceImg, d_destImg, d_mask, d_blended_red_in, d_blended_blue_in, d_blended_green_in);
    preProcessSource<<< 333, 512 >>>(d_sourceImg, d_destImg, d_mask, d_blended_red_out, d_blended_blue_out, d_blended_green_out);

    for( short k = 0; k < 800; k++) {
      computeIterate<<< 333, 512 >>>(d_mask, d_blended_red_in, d_blended_red_out, d_metric_red);
      std::swap(d_blended_red_in, d_blended_red_out);
    }

    for( short k = 0; k < 800; k++) {
      computeIterate<<< 333, 512 >>>(d_mask, d_blended_blue_in, d_blended_blue_out, d_metric_blue);
      std::swap(d_blended_blue_in, d_blended_blue_out);
    }

    for( short k = 0; k < 800; k++) {
      computeIterate<<< 333, 512 >>>(d_mask, d_blended_green_in, d_blended_green_out, d_metric_green);
      std::swap(d_blended_green_in, d_blended_green_out);
    }

    blendImage<<< 333, 512 >>>(d_mask, d_blended_red_in, d_blended_blue_in, d_blended_green_in, d_destImg);
    checkCudaErrors(cudaMemcpy(h_blendedImg, d_destImg, sizeof(uchar4)  * srcSize, cudaMemcpyDeviceToHost ));

    checkCudaErrors(cudaFree(d_mask));
    checkCudaErrors(cudaFree(d_blended_red_in));
    checkCudaErrors(cudaFree(d_blended_blue_in));
    checkCudaErrors(cudaFree(d_blended_green_in));
    checkCudaErrors(cudaFree(d_blended_red_out));
    checkCudaErrors(cudaFree(d_blended_blue_out));
    checkCudaErrors(cudaFree(d_blended_green_out));
    checkCudaErrors(cudaFree(d_metric_red));
    checkCudaErrors(cudaFree(d_metric_blue));
    checkCudaErrors(cudaFree(d_metric_green));
    checkCudaErrors(cudaFree(d_dest_red));
    checkCudaErrors(cudaFree(d_dest_blue));
    checkCudaErrors(cudaFree(d_dest_green));
    checkCudaErrors(cudaFree(d_sourceImg));
    checkCudaErrors(cudaFree(d_destImg));
}