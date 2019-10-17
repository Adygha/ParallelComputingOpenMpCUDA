/**
 * A project for the 'Parallel Computing' course at Linnaeus University using OpenMP and CUDA APIs and measure their calculation wall-times.
 * @author Janty Azmat
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <omp.h>
#include <float.h>
#ifdef _WIN32
#include <intrin.h>
#else
#include <cpuid.h>
#endif // _WIN32

const unsigned COUNT_THREADS[] = {1U, 6U, 12U, 24U, 48U}; // The thread count used in every phase of OpenMP calculation.
const unsigned COUNT_ITERATIONS[] = {240000000U, 480000000U, 960000000U, 1920000000U, 3840000000U}; // The number of PI calculation iterations used by OpenMp and CUDA
#define DISPLAY_NON_MEAN_RESULTS true // A flag to specify whether to display every iteration's results (before calculating mean/average)
#define INCLUDE_CUDA_MALLOC_TIME true // A flag to specify whether to 'cudaMalloc' time in CUDA PI calculation time
#define MEAN_REPEAT 120 // Number of repeates to calculate the mean/average
#define MEAN_TRIM 10 // The number of trimmed repeates (highest and lowestwith total 2*MEAN_TRIM trim) from the repeates to calculate the mean/average
#define PRINTF_PI_PRECISION 11 // Low precision for displaying PI results
#define PRINTF_PI_PRECISION_FULL 23 // High precision for displaying PI results
#define PRINTF_TIME_PRECISION 10 // Precision for displaying wall-time
#define FILE_RESULTS "results.txt" // A file to put the results in

#define getElemName(var) #var // Just to get the element (like variable) name
#define ASSIGNMENT_PI 3.14159265358979323846264

// Forward declarations
double runOnHostOpenMpCalcPiTask1(unsigned inThreadCount, unsigned inIterationCount, double *outResultPI, bool isDisplayResult); // Calculates PI using OpenMP (returns wall-time)
double runOnCudaDeviceKernelCalcPiTask2(const unsigned inIterationCount, double *outResultPI, bool isWithMallocTime, bool isDisplayResults); // Calculates PI using CUDA (returns wall-time)
double trimmedMean(double *inValuesArray, unsigned arraySize, unsigned trimCount); // Calculates trimmed/truncated mean of a double-array
double standardDeviation(double *inSortedValuesArray, unsigned arraySize, unsigned trimCount, double theMean);
void printCpuGpuInfo(); // Displays CPU and GPU detailed names for now

/**
 * Main entry.
 * @author Janty Azmat
 */
int main() {
	const unsigned tmpThreadCalcs = sizeof(COUNT_THREADS) / sizeof(unsigned int);
	const unsigned tmpIterCalcs = sizeof(COUNT_ITERATIONS) / sizeof(unsigned int);
	double tmpOpenMpResults[tmpThreadCalcs][tmpIterCalcs][MEAN_REPEAT];
	double tmpCudaResults[tmpIterCalcs][MEAN_REPEAT];
	double tmpOpenMpTimes[tmpThreadCalcs][tmpIterCalcs][MEAN_REPEAT];
	double tmpCudaTimes[tmpIterCalcs][MEAN_REPEAT];
	double tmpPI = 0.0; // PI interim mean result
	double tmpTime = 0.0; // Timer interim mean result
	double tmpStdDevi = 0.0; // Timer interim mean standard deviation
	FILE *tmpFile = fopen(FILE_RESULTS, "w");

	printf("\nThis machine's double-epsilon is: [%.*lf]. It should be considered by observers if it matters.\n\n", PRINTF_PI_PRECISION_FULL, DBL_EPSILON);
	printCpuGpuInfo();

	for (unsigned th = 0U; th < tmpThreadCalcs; th++) { // OpenMP phase
		for (unsigned itr = 0U; itr < tmpIterCalcs; itr++) {
			printf("OpenMP PI calculate using '%u' threads and '%u' iterations repeated '%u' times:\n", COUNT_THREADS[th], COUNT_ITERATIONS[itr], MEAN_REPEAT);
			for (unsigned rep = 0; rep < MEAN_REPEAT; rep++) {
				tmpOpenMpTimes[th][itr][rep] = runOnHostOpenMpCalcPiTask1(COUNT_THREADS[th], COUNT_ITERATIONS[itr], &tmpOpenMpResults[th][itr][rep], DISPLAY_NON_MEAN_RESULTS);
			}
			printf("\nLast round trimmed mean/average PI result:%40s[%.*lf] [%.*lf]\n",
					" ", PRINTF_PI_PRECISION_FULL, (tmpPI = trimmedMean(&tmpOpenMpResults[th][itr][0], MEAN_REPEAT, MEAN_TRIM)), PRINTF_PI_PRECISION, tmpPI);
			printf("Last round trimmed mean/average PI result standard deviation:%21s[%.*lf]\n",
					" ", PRINTF_PI_PRECISION_FULL, standardDeviation(&tmpOpenMpResults[th][itr][0], MEAN_REPEAT, MEAN_TRIM, tmpPI));
			printf("Last round trimmed mean/average PI result difference from assignment PI constant: [%.*lf]\n", PRINTF_PI_PRECISION_FULL, tmpPI - ASSIGNMENT_PI);
			printf("Last round trimmed mean/average time:%45s[%.*lf]\n", " ", PRINTF_TIME_PRECISION, (tmpTime = trimmedMean(&tmpOpenMpTimes[th][itr][0], MEAN_REPEAT, MEAN_TRIM)));
			printf("Last round trimmed mean/average time standard deviation:%26s[%.*lf]\n\n\n",
					" ", PRINTF_TIME_PRECISION, (tmpStdDevi = standardDeviation(&tmpOpenMpTimes[th][itr][0], MEAN_REPEAT, MEAN_TRIM, tmpTime)));
			fprintf(tmpFile, "Result of OpenMP PI calculate using '%u' threads and '%u' iterations:\n"
					"\tPI result:%48s[%.*lf] [%.*lf]\n"
					"\tPI result difference from assignment PI constant:%9s[%.*lf]\n"
					"\tCalculation trimmed mean/average time:%20s[%.*lf]\n"
					"\tCalculation trimmed mean/average time standard deviation: [%.*lf]\n\n",
					COUNT_THREADS[th], COUNT_ITERATIONS[itr],
					" ", PRINTF_PI_PRECISION_FULL, tmpPI, PRINTF_PI_PRECISION, tmpPI,
					" ", PRINTF_PI_PRECISION_FULL, tmpPI - ASSIGNMENT_PI,
					" ", PRINTF_TIME_PRECISION, tmpTime,
					PRINTF_TIME_PRECISION, tmpStdDevi);
		}
	}

	for (unsigned itr = 0U; itr < tmpIterCalcs; itr++) { // CUDA phase
		printf("CUDA PI calculate using '%u' iterations (including 'cudaMalloc' time) repeated '%u' times:\n", COUNT_ITERATIONS[itr], MEAN_REPEAT);
		for (unsigned rep = 0; rep < MEAN_REPEAT; rep++) {
			tmpCudaTimes[itr][rep] = runOnCudaDeviceKernelCalcPiTask2(COUNT_ITERATIONS[itr], &tmpCudaResults[itr][rep], INCLUDE_CUDA_MALLOC_TIME, DISPLAY_NON_MEAN_RESULTS);
		}
		printf("\nLast round trimmed mean/average PI result:%40s[%.*lf] [%.*lf]\n",
				" ", PRINTF_PI_PRECISION_FULL, (tmpPI = trimmedMean(&tmpCudaResults[itr][0], MEAN_REPEAT, MEAN_TRIM)), PRINTF_PI_PRECISION, tmpPI);
		printf("Last round trimmed mean/average PI result standard deviation:%21s[%.*lf]\n",
				" ", PRINTF_PI_PRECISION_FULL, standardDeviation(&tmpCudaResults[itr][0], MEAN_REPEAT, MEAN_TRIM, tmpPI));
		printf("Last round trimmed mean/average PI result difference from assignment PI constant: [%.*lf]\n", PRINTF_PI_PRECISION_FULL, tmpPI - ASSIGNMENT_PI);
		printf("Last round trimmed mean/average time:%45s[%.*lf]\n", " ", PRINTF_TIME_PRECISION, (tmpTime = trimmedMean(&tmpCudaTimes[itr][0], MEAN_REPEAT, MEAN_TRIM)));
		printf("Last round trimmed mean/average time standard deviation:%26s[%.*lf]\n\n\n",
				" ", PRINTF_TIME_PRECISION, standardDeviation(&tmpCudaTimes[itr][0], MEAN_REPEAT, MEAN_TRIM, tmpTime));
		fprintf(tmpFile, "Result of CUDA PI calculate using '%u' iterations:\n"
				"\tPI result:%48s[%.*lf] [%.*lf]\n"
				"\tPI result difference from assignment PI constant:%9s[%.*lf]\n"
				"\tCalculation trimmed mean/average time:%20s[%.*lf]\n"
				"\tCalculation trimmed mean/average time standard deviation: [%.*lf]\n\n",
				COUNT_ITERATIONS[itr],
				" ", PRINTF_PI_PRECISION_FULL, tmpPI, PRINTF_PI_PRECISION, tmpPI,
				" ", PRINTF_PI_PRECISION_FULL, tmpPI - ASSIGNMENT_PI,
				" ", PRINTF_TIME_PRECISION, tmpTime,
				PRINTF_TIME_PRECISION, tmpStdDevi);
	}

	fclose(tmpFile);
	return 0;
}

/**
 * Used to display CPU and GPU detailed names.
 * @author Janty Azmat
 */
void printCpuGpuInfo() { // Based on: https://stackoverflow.com/questions/850774
	char tmpCpuBrand[64] = {0};
	int tmpCpuInfo[4] = {-1};
	unsigned tmpExIdCount;
	cudaDeviceProp tmpPrp = {0};
#ifdef _WIN32
	__cpuid(tmpCpuInfo, 0x80000000);
#else
	__cpuid(0x80000000, tmpCpuInfo[0], tmpCpuInfo[1], tmpCpuInfo[2], tmpCpuInfo[3]);
#endif // _WIN32
	tmpExIdCount = tmpCpuInfo[0];
	for (unsigned i = 0x80000002; i <= tmpExIdCount || i <= 0x80000004; ++i) {
#ifdef _WIN32
		__cpuid(tmpCpuInfo, i);
#else
		__cpuid(i, tmpCpuInfo[0], tmpCpuInfo[1], tmpCpuInfo[2], tmpCpuInfo[3]);
#endif // _WIN32
		if (i == 0x80000002)
			memcpy(tmpCpuBrand, tmpCpuInfo, sizeof(tmpCpuInfo));
		else if (i == 0x80000003)
			memcpy(tmpCpuBrand + 16, tmpCpuInfo, sizeof(tmpCpuInfo));
		else if (i == 0x80000004)
			memcpy(tmpCpuBrand + 32, tmpCpuInfo, sizeof(tmpCpuInfo));
	}
	printf("CPU Name: [%s]\n", tmpCpuBrand);
	if (cudaGetDeviceProperties(&tmpPrp, 0) != cudaSuccess) return;
	printf("GPU Name: [%s]\n\n", tmpPrp.name);
}

/**
 * Used as a double comparator callbeck with 'qsort'.
 * @author Janty Azmat
 * @param leftVal	left value to compare.
 * @param rightVal	right value to compare.
 * @return			integer comparison result ('0' on equality, '-1' if left < right, or '1' if left > right)
 */
int doubleComparator(const void *leftVal, const void *rightVal) {
	if (*(double *)leftVal < *(double *)rightVal) {
		return -1;
	} else if (*(double *)leftVal > * (double *)rightVal) {
		return 1;
	}
	return 0;
}

/**
 * Used to get the trimmed/truncated mean/average for the values in the specified array after sorting it (input array might change).
 * @author Janty Azmat
 * @param inValuesArray		the array pointer that contains the values (it's values will be sorted).
 * @param arraySize			the size/length of the array.
 * @param trimCount			the number of values to trim from each side of the sorted array.
 * @return					the trimmed/truncated mean/average for the values in the specified array.
 */
double trimmedMean(double *inValuesArray, unsigned arraySize, unsigned trimCount) {
	double outMean = 0.0;
	if (arraySize > 2U * trimCount) {
		qsort(inValuesArray, arraySize, sizeof(double), doubleComparator); // Sort first
		for (unsigned i = trimCount; i < arraySize - trimCount; i++) { // Loop and sum excluding the trimmed part
			outMean += inValuesArray[i];
		}
		outMean /= (double)(arraySize - 2U * trimCount);
	}
	return outMean;
}

/**
 * Used to get the standard-deviation of the mean for the values in the specified sorted array.
 * @author Janty Azmat
 * @param inSortedValuesArray	the array pointer that contains the sorted values.
 * @param arraySize				the size/length of the array.
 * @param trimCount				the number of values to trim from each side of the sorted array.
 * @param theMedian				the already calculated mean/average for the array's values.
 * @return						the standard-deviation of the mean for the values in the specified sorted array.
 */
double standardDeviation(double *inSortedValuesArray, unsigned arraySize, unsigned trimCount, double theMean) {
	double outDevi = 0.0;
	if (arraySize > 2U * trimCount && theMean > 0.0) {
		for (unsigned i = trimCount; i < arraySize - trimCount; i++) { // Loop to calculate first art while excluding the trimmed part
			outDevi += pow(inSortedValuesArray[i] - theMean, 2.0);
		}
		outDevi = sqrt(outDevi / (double)(arraySize - 1U));
	}
	return outDevi;
}

// vvvvvvvvvvvvvvvv Start Task2 Part vvvvvvvvvvvvvvvv //

/**
 * An enumeration to get the number of each streaming-multiprocessor's cores based on compute capability version.
 * @author Janty Azmat
 */
enum CudaSmCores {
	SmCores_UNKNOWN = 16,			// Used when the compute capability version is unknown
	SmCores_V2_0 = 32,				// Used for V2.0
	SmCores_V2_1 = 48,				// Used for V2.1
	SmCores_V2_1_POW_OF_TWO = 32,	// Used when pow of two cores required for V2.1
	SmCores_V3_X = 192,				// Used for V3.x
	SmCores_V3_X_POW_OF_TWO = 128,	// Used when pow of two cores required for V3.x
	SmCores_V5_X = 128,				// Used for V5.x
	SmCores_V6_0 = 64,				// Used for V6.0
	SmCores_V6_X = 128,				// Used for V6.x (other than V6.1)
	SmCores_V7_X = 64				// Used for V7.x
};

/**
 * Used to get the preferred block size (number of threads) for the currently set CUDA device.
 * @author Janty Azmat
 * @param isOnlyPoweOfTwo	'true' to specify that only power of two result is required.
 * @return					the preferred block size (number of threads).
 */
unsigned getCudaPrefBlockSize(bool isOnlyPoweOfTwo) {
	// Solution based on recommendations at: https://docs.nvidia.com/cuda/cuda-c-programming-guide and https://stackoverflow.com/a/32531982
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

/**
 * A CUDA kernel that is used to calculate PI using the specified iteration count and writes every block's result to 'outArrayPI' array.
 * @author Janty Azmat
 * @param iterCount				the number of iterations the PI calculation will be conducted.
 * @param inItersReciprocal		the reciprocal of 'iterCount' as double precision float.
 * @param outArrayPI			the output array that will contain every block's result (it's size should be >= number of blocks passed to the kernel call).
 */
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

/**
 * A CUDA kernel that is used to sum-reduce the values of 'inArray' and write results to 'outArray'.
 * @author Janty Azmat
 * @param rangeToReduce		the number of values from 'inArray' to sum-reduce.
 * @param inArray			the input array that containes the values to be sum-reduced (its size should be >= 'rangeToReduce').
 * @param outArray			the output array that will contain every block's sum-reduction result (it's size should be >= number of blocks passed to the kernel call)
 */
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

/**
 * A helper function that is used to check the specified CUDA error and display its information.
 * @author Janty Azmat
 * @param theErr		the CUDA error.
 * @param commandName	the command/function that is being checked as possible cause of the error (can be null).
 * @param causeName		the parameter used with 'commandName' and possibly caused the error.
 * @return				'true' if there was an actual error, or-else 'false'.
 */
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

/**
 * Runs the project's TASK-2 requirements: Calculates PI using CUDA API with the specified conditions and calculate its run wall-time.
 * @author Janty Azmat
 * @param inIterationCount	the number of iterations the PI calculation will be conducted.
 * @param outResultPI		the calculated PI result.
 * @param isWithMallocTime	'true' to specify that 'cudaMalloc' calls' times should be included in the wall-time calculations, or-else 'false'.
 * @param isDisplayResult	'true' to display this call/phase result by this function.
 * @return					the wall-time taken conducting this phase's calculations done by CUDA.
 */
double runOnCudaDeviceKernelCalcPiTask2(unsigned inIterationCount, double *outResultPI, bool isWithMallocTime, bool isDisplayResult) {
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

	cudaKernelCalcPI<<<tmpBlockCount, tmpPrefBlockSize, tmpPrefBlockSize * tmpDblSize>>>(inIterationCount, tmpItersReciprocal, tmpDevPI); // calculate PI to an array of blocks
	while (tmpBlockCount > 1) { // We flipping flag and two arrays so that we don't create a new array or copy data each time we reduce
		tmpSize = tmpBlockCount;
		tmpBlockCount = (tmpBlockCount + tmpPrefBlockSize - 1) / tmpPrefBlockSize;
		cudaKernelReductionPhase<<<tmpBlockCount, tmpPrefBlockSize, tmpPrefBlockSize * tmpDblSize>>>(
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
	if (isDisplayResult) printf("Calculate PI using CUDA with block-size '%u': [%.*lf] [%.*lf]. with time: [%.*lf]\n", tmpPrefBlockSize, // Display results if needed
								PRINTF_PI_PRECISION_FULL, *outResultPI, PRINTF_PI_PRECISION, *outResultPI, PRINTF_TIME_PRECISION, tmpTime);
	return tmpTime;
}
// ^^^^^^^^^^^^^^^^  End Task2 Part  ^^^^^^^^^^^^^^^^ //


// vvvvvvvvvvvvvvvv Start Task1 Part vvvvvvvvvvvvvvvv //

/**
 * Runs the project's TASK-1 requirements: Calculates PI using OpenMP API with the specified conditions and calculate its run wall-time.
 * @author Janty Azmat
 * @param inThreadCount		the number of threads OpenMP should assign for this phase.
 * @param inIterationCount	the number of iterations the PI calculation will be conducted.
 * @param outResultPI		the calculated PI result.
 * @param isDisplayResult	'true' to display this call/phase result by this function.
 * @return					the wall-time taken conducting this phase's calculations done by OpenMP.
 */
double runOnHostOpenMpCalcPiTask1(unsigned inThreadCount, unsigned inIterationCount, double *outResultPI, bool isDisplayResult) {
	double outPI = 0.0; // The PI result
	double tmpNthIter; // The N-th iteration
	double tmpItersReciprocal = 1.0 / (double)inIterationCount; // The reciprocal of iteration-count (needed for PI calculation)
	double tmpTime; // Timer

	tmpTime = omp_get_wtime();

	// In next parallel for:
	// - Reducing 'outPI'.
	// - Sharing 'tmpItersReciprocal' and 'inIterationCount' (same value used by all and reading with no writing)
	// - Private 'tmpNthIter' for every thread to avoid race hazards and helps avoiding using any atomic variable/operation
	// - Number of threads as the function parameter specifies
	// - Schedule to static can be used to avoid auto/default decision of schedule
	// - Ordered can be used (although slower on slow machines) to avoid floating-point associative calculation
	//   differences: https://en.wikipedia.org/wiki/Floating-point_arithmetic#Accuracy_problems
#pragma omp parallel for default(none) reduction(+:outPI) shared(tmpItersReciprocal, inIterationCount) private(tmpNthIter) \
																		num_threads(inThreadCount) //schedule(static) ordered
	for (unsigned i = 0U; i < inIterationCount; i++) {
		tmpNthIter = ((double)i + 0.5) * tmpItersReciprocal;
		outPI += 4.0 / (1.0 + tmpNthIter * tmpNthIter);
	}
	outPI *= tmpItersReciprocal; // One instruction done by a single thread is OK.

	tmpTime = omp_get_wtime() - tmpTime;

	if (isDisplayResult) {
		printf("Calculate PI using OpenMp with '%2d' threads: [%.*lf] [%.*lf]. with time: [%.*lf]\n",
			   inThreadCount, PRINTF_PI_PRECISION_FULL, outPI, PRINTF_PI_PRECISION, outPI, PRINTF_TIME_PRECISION, tmpTime);
	}
	*outResultPI = outPI;
	return tmpTime;
}
// ^^^^^^^^^^^^^^^^  End Task1 Part  ^^^^^^^^^^^^^^^^ //
