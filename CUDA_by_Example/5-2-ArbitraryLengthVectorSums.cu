#include "common/book.h"

#define N 10

__global__ void add( int *a, int *b, int *c ) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	while (tid < N ) {
		c[tid] = a[tid] + b[tid];
		tid += blockDim.x * gridDim.x;
	}
}

int main( void )  {
	int a[N], b[N], c[N];
	int *dev_a, *dev_b, *dev_c;

	// Allocate the memory on the GPU

	HANDLE_ERROR( cudaMalloc( (void**)&dev_a, N * sizeof(int) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&dev_b, N * sizeof(int) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&dev_c, N * sizeof(int) ) );

	// fill the arrays 'a' and 'b' on the CPU
	for (int ii = 0; ii < N; ii++) {
		a[ii] = ii;
		b[ii] = ii * ii;
	}

	HANDLE_ERROR( cudaMemcpy( dev_a,
				a,
				N * sizeof(int),
				cudaMemcpyHostToDevice ) );


	HANDLE_ERROR( cudaMemcpy( dev_b,
				b,
				N * sizeof(int),
				cudaMemcpyHostToDevice ) );
	add<<<1,N>>>( dev_a, dev_b, dev_c );

	HANDLE_ERROR( cudaMemcpy( c,
				dev_c,
				N * sizeof(int),
				cudaMemcpyDeviceToHost ) );

	// display the results
	for (int ii = 0; ii < N; ii++)
	{
		printf( "%d + %d = %d\n", a[ii], b[ii], c[ii] );
	}

	cudaFree( dev_a );
	cudaFree( dev_b );
	cudaFree( dev_c );

	return 0;
}
