#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <omp.h>
#include <stdlib.h>

const unsigned COUNT_THREADS[] = {1U, 6U, 12U, 24U, 48U};
//const unsigned COUNT_ITERATIONS[] = {24000000U, 48000000U, 960000000U, 1920000000U, 3840000000U};
const unsigned COUNT_ITERATIONS[] = {134217728U, 268435456U, 536870912U, 1073741824U, 2147483648U};
#define AVERAGE_REPEAT 5
#define PRINTF_PI_PRECISION 11
#define PRINTF_PI_PRECISION_FULL 23
#define PRINTF_TIME_PRECISION 10

//#define ldouble_t long double // Just to use more capable 'long double' on Ida's gcc
#define getElemName(var) #var // Just to get the element (like variable) name
#define ASSIGNMENT_PI 3.14159265358979323846264

// Forward declarations
double runOnHostOpenMpCalcPiTask1(unsigned inThreadCount, unsigned inIterationCount, double *outResultPI); // Calculates PI using OpenMP (returns wall-time)
double runOnCudaDeviceKernelCalcPiTask2(const unsigned inIterationCount, double *outResultPI, bool isWithMallocTime, bool isDisplayResults); // Calculates PI using CUDA (returns wall-time)

int main(void) {
	const unsigned tmpThreadCalcs = sizeof(COUNT_THREADS) / sizeof(unsigned int);
	const unsigned tmpIterCalcs = sizeof(COUNT_ITERATIONS) / sizeof(unsigned int);
	double tmpPI = 0.0;

	//printf("wtick[%.23lf]\n", omp_get_wtick());
	//printf("DBL_EPSILON[%.55lf]\n", DBL_EPSILON);

	for (unsigned th = 0; th < tmpThreadCalcs; th++) {
		for (unsigned itr = 0; itr < tmpIterCalcs; itr++) {
			printf("OpenMP PI calculations using '%u' threads and '%u' iterations repeated '%u' time:\n", COUNT_THREADS[th], COUNT_ITERATIONS[itr], AVERAGE_REPEAT);
			for (unsigned rep = 0; rep < AVERAGE_REPEAT; rep++) {
				runOnHostOpenMpCalcPiTask1(COUNT_THREADS[th], COUNT_ITERATIONS[itr], &tmpPI);
			}
		}
	}

	for (unsigned itr = 0; itr < tmpIterCalcs; itr++) {
		printf("CUDA PI calculations using '%u' iterations (including 'cudaMalloc' time) repeated '%u' time:\n", COUNT_ITERATIONS[itr], AVERAGE_REPEAT);
		for (unsigned rep = 0; rep < AVERAGE_REPEAT; rep++) {
			runOnCudaDeviceKernelCalcPiTask2(COUNT_ITERATIONS[itr], &tmpPI, true, true);
		}
	}

	return 0;
}

int doubleComparator(const void *leftVal, const void *rightVal) {
	if (*(double *)leftVal < *(double *)rightVal) {
		return -1;
	} else if (*(double *)leftVal > * (double *)rightVal) {
		return 1;
	}
	return 0;
}

double trimmedMean(double *inValuesArray, unsigned arraySize) {
	qsort(inValuesArray, arraySize, sizeof(double), doubleComparator); // Sort first

}

// vvvvvvvvvvvvvvvv Start Task2 Part vvvvvvvvvvvvvvvv //

enum CudaSmCores { // Each streaming-multiprocessor's cores based on compute capability version
	SmCores_UNKNOWN = 16,
	SmCores_V2_0 = 32,
	SmCores_V2_1 = 48,
	SmCores_V2_1_POW_OF_TWO = 32, // Used when pow of two is required
	SmCores_V3_X = 192,
	SmCores_V3_X_POW_OF_TWO = 128, // Used when pow of two is required
	SmCores_V5_X = 128,
	SmCores_V6_0 = 64,
	SmCores_V6_X = 128,
	SmCores_V7_X = 64
};

// Similar to recommendations at: https://docs.nvidia.com/cuda/cuda-c-programming-guide and https://stackoverflow.com/a/32531982
unsigned getCudaPrefBlockSize(bool isOnlyPoweOfTwo) {
	cudaDeviceProp tmpPrp = {0};
	if (cudaGetDeviceProperties(&tmpPrp, 0) != cudaSuccess) return 0;
	switch (tmpPrp.major) {
		case 2: // Fermi
			switch (tmpPrp.minor) {
				case 0:
					return SmCores_V2_0; // Number of each Streaming processor's cores for v2.0
				case 1:
					return isOnlyPoweOfTwo ? SmCores_V2_1_POW_OF_TWO : SmCores_V2_1; // Number of each Streaming processor's cores for v2.1(or its pow of 2 equivalent)
			}
		case 3: // Kepler
			return isOnlyPoweOfTwo ? SmCores_V3_X_POW_OF_TWO : SmCores_V3_X; // Number of each Streaming processor's core for all (or its pow of 2 equivalent)
		case 5: // Maxwell
			return SmCores_V5_X; // Number of each Streaming processor's core for all
		case 6: // Pascal
			switch (tmpPrp.minor) {
				case 0:
					return SmCores_V6_0; // Number of each Streaming processor's cores for v6.0
				case 1:
					return SmCores_V6_X; // Number of each Streaming processor's cores for v2.1
			}
		case 7: // Volta
			return SmCores_V7_X;
	}
	//return 0; // Unknown
	return SmCores_UNKNOWN; // Unknown is minimal (May change this)
}

__global__ void cudaKernelCalcPI(unsigned iterCount, double inItersReciprocal, double *outArrayPI) {
	extern __shared__ double tmpBlockResults[]; // Dynamically allocated array to store interim block results (allocated by kernel call)
	unsigned tmpIter = blockDim.x * blockIdx.x + threadIdx.x;

	// At this point we should check if index/iteration is in the range of the total iteration count, but in our case the input is always in rage
	double tmpNthIter = ((double)tmpIter + 0.5) * inItersReciprocal;//
	tmpBlockResults[threadIdx.x] = 4.0 / (1.0 + tmpNthIter * tmpNthIter);							// Each thread's share of the PI calculations
	__syncthreads(); // Barrier

	// Next, reduce block's shared array (based on Nvidia help documentation). Here, we need the block size to be of
	// power of two (to avoid extra calculations on first reduction when '2' does not divide 'i')
	for (unsigned i = blockDim.x >> 1; i > 0; i >>= 1) { // Reversed loop iterates on first half of block results array
		if (threadIdx.x < i && tmpIter + i < iterCount) {
			tmpBlockResults[threadIdx.x] += tmpBlockResults[threadIdx.x + i];
		}
		__syncthreads(); // Barrier
	}
	if (threadIdx.x == 0) {
		outArrayPI[blockIdx.x] = tmpBlockResults[0]; // Copy the block's calculation result to output result array
	}
}

__global__ void cudaKernelReductionPhase(unsigned rangeToReduce, double *inArray, double *outArray) {
	extern __shared__ double tmpBlockReduction[]; // Dynamically allocated array to store interim block reduction (allocated by kernel call)
	unsigned tmpIndex = blockDim.x * blockIdx.x + threadIdx.x;
	tmpBlockReduction[threadIdx.x] = inArray[tmpIndex];
	__syncthreads(); // Barrier

	// Next, reduce block's shared array (based on Nvidia help documentation)
	for (unsigned i = (blockDim.x + 1) >> 1; i > 0; i >>= 1) { // Reversed loop iterates on first half (ceiling half) of block results array
		if (threadIdx.x < i && tmpIndex + i < rangeToReduce) { // Being careful not to go out of array range (and out of the original array range)
			tmpBlockReduction[threadIdx.x] += tmpBlockReduction[threadIdx.x + i];
		}
		__syncthreads(); // Barrier
	}
	if (threadIdx.x == 0) { // Only one thread from the block
		outArray[blockIdx.x] = tmpBlockReduction[0]; // Copy the block's calculation result to output result array
	}
}

bool cudaCheckError(cudaError_t theErr, const char *commandName, const char *causeName) {
	if (theErr != cudaSuccess) {
		if (commandName && causeName) {
			printf("Cuda error using '%s' on '%s' with error-code: [%d], and error-message: [%s]\n", commandName, causeName, theErr, cudaGetErrorString(theErr));
		} else if (commandName) {
			printf("Cuda error using '%s' with error-code: [%d], and error-message: [%s]\n", commandName, theErr, cudaGetErrorString(theErr));
		} else {
			printf("Cuda error with error-code: [%d], and error-message: [%s]\n", theErr, cudaGetErrorString(theErr));
		}
		return true;
	}
	return false;
}

double runOnCudaDeviceKernelCalcPiTask2(unsigned inIterationCount, double *outResultPI, bool isWithMallocTime, bool isDisplayResults) {
	unsigned tmpPrefBlockSize = getCudaPrefBlockSize(true); // Preferred cuda block size (thread count per block)
	unsigned tmpBlockCount = (inIterationCount + tmpPrefBlockSize - 1) / tmpPrefBlockSize; // Cuda block count (ceiling=(numerator+denominator-1)/denominator)
	unsigned tmpDblSize = sizeof(double), tmpSize; // Sizes
	double tmpItersReciprocal = 1.0 / (double)inIterationCount; // The reciprocal of iteration-count (needed for PI calculation)
	double *tmpDevPI, *tmpDevPiReduce; // Device version of the outputted PI result (with the number of blocks as size)
	double tmpTime; // Timer
	bool tmpIsUseOrig = true; // A flag to check if the final PI result is in 'tmpDevPI' or 'tmpDevPiReduce' ('true' for 'tmpDevPI')

	if (isWithMallocTime) tmpTime = omp_get_wtime(); // Timer start (if 'isWithMallocTime' is true)

	// This block of code allocates result arrays on cuda global memory (while checking for errors)
	if (cudaCheckError(cudaMalloc(&tmpDevPI, tmpBlockCount * tmpDblSize), getElemName(cudaMalloc), getElemName(tmpDevPI))) return -1.0;
	if (cudaCheckError(cudaMalloc(&tmpDevPiReduce, tmpBlockCount * tmpDblSize), getElemName(cudaMalloc), getElemName(tmpDevPiReduce))) return -1.0;

	if (!isWithMallocTime) {
		if (cudaCheckError(cudaDeviceSynchronize(), getElemName(cudaKernelReducePhase), NULL)) return -1.0; // Block host thread until cudaMallocs finish
		tmpTime = omp_get_wtime(); // Timer start (if 'isWithMallocTime' is false)
	}

	cudaKernelCalcPI << <tmpBlockCount, tmpPrefBlockSize, tmpPrefBlockSize *tmpDblSize >> > (inIterationCount, tmpItersReciprocal, tmpDevPI); // calculate PI to an array of blocks
	while (tmpBlockCount > 1) { // We flipping flag and two arrays so that we don't create a new array or copy data each time we reduce
		tmpSize = tmpBlockCount;
		tmpBlockCount = (tmpBlockCount + tmpPrefBlockSize - 1) / tmpPrefBlockSize;
		cudaKernelReductionPhase << <tmpBlockCount, tmpPrefBlockSize, tmpPrefBlockSize *tmpDblSize >> > (
			tmpSize, tmpIsUseOrig ? tmpDevPI : tmpDevPiReduce, tmpIsUseOrig ? tmpDevPiReduce : tmpDevPI); // Flip used array based on flag
		tmpIsUseOrig = !tmpIsUseOrig; // Flip the flag
	}
	if (cudaCheckError(cudaDeviceSynchronize(), getElemName(cudaKernelReducePhase), NULL)) return -1.0; // Block host thread until kernels finish (kernel calls are asynchronous)

	if (!isWithMallocTime) tmpTime = omp_get_wtime() - tmpTime; // Timer end (if 'isWithMallocTime' is false)

	// Next line will copy the reduced result from cuda global memory (while checking for errors)
	if (cudaCheckError(cudaMemcpy(outResultPI, (tmpIsUseOrig ? &tmpDevPI[0] : &tmpDevPiReduce[0]), tmpDblSize, cudaMemcpyDeviceToHost), getElemName(cudaMemcpy), getElemName(outResultPI))) return -1.0;
	*outResultPI *= tmpItersReciprocal; // One last instruction for PI calculation (done by a single thread is OK).
	if (isWithMallocTime) tmpTime = omp_get_wtime() - tmpTime; // Timer end (if 'isWithMallocTime' is true)

	cudaCheckError(cudaFree(tmpDevPI), getElemName(cudaFree), getElemName(tmpDevPI));				// Free the cuda allocated global memory. Even if an error happened
	cudaCheckError(cudaFree(tmpDevPiReduce), getElemName(cudaFree), getElemName(tmpDevPiReduce));	// here, we already got our result (so we just print the error)
	if (isDisplayResults) printf("Calculate PI using CUDA with block-size '%u': [%.*lf] [%.*lf]. with time: [%.*lf]\n", tmpPrefBlockSize, // Display results if needed
								 PRINTF_PI_PRECISION_FULL, *outResultPI, PRINTF_PI_PRECISION, *outResultPI, PRINTF_TIME_PRECISION, tmpTime);
	return tmpTime;
}
// ^^^^^^^^^^^^^^^^  End Task2 Part  ^^^^^^^^^^^^^^^^ //


// vvvvvvvvvvvvvvvv Start Task1 Part vvvvvvvvvvvvvvvv //

double runOnHostOpenMpCalcPiTask1(unsigned inThreadCount, unsigned inIterationCount, double *outResultPI) {
	double outPI = 0.0; // The PI result
	double tmpNthIter; // The N-th iteration
	double tmpItersReciprocal = 1.0 / (double)inIterationCount; // The reciprocal of iteration-count (needed for PI calculation)
	double tmpTime; // Timer

	tmpTime = omp_get_wtime();

	// In next parallel for:
	// - Reducing 'outPI'.
	// - Sharing 'tmpItersReciprocal' and 'inIterationCount' (same value used by all and reading with no writing)
	// - Private 'tmpNthIter' for every thread to avoid race hazards and helps avoiding using any atomic variable/operation
	// - Number of thread as the function parameter specifies
	// - Schedule to static can be used to avoid auto/default decision of schedule
	// - Ordered can be used (although slower on slow machines) to avoid floating-point associative calculation
	//   differences: https://en.wikipedia.org/wiki/Floating-point_arithmetic#Accuracy_problems
#pragma omp parallel for default(none) reduction(+:outPI) shared(tmpItersReciprocal, inIterationCount) private(tmpNthIter) num_threads(inThreadCount) schedule(static) ordered
	for (unsigned i = 0; i < inIterationCount; i++) {
		tmpNthIter = ((double)i + 0.5) * tmpItersReciprocal;
		outPI += 4.0 / (1.0 + tmpNthIter * tmpNthIter);
	}
	outPI *= tmpItersReciprocal; // One instruction done by a single thread is OK.

	tmpTime = omp_get_wtime() - tmpTime;

	printf("Calculate PI using OpenMp with '%2d' threads: [%.*lf] [%.*lf]. with time: [%.*lf]\n",
		   inThreadCount, PRINTF_PI_PRECISION_FULL, outPI, PRINTF_PI_PRECISION, outPI, PRINTF_TIME_PRECISION, tmpTime);
	*outResultPI = outPI;
	return tmpTime;
}
// ^^^^^^^^^^^^^^^^  End Task1 Part  ^^^^^^^^^^^^^^^^ //
