#define _GNU_SOURCE
#include <stdlib.h>
#include "OMPFK.h"
#include "Common.h"
#include <stdio.h>



unsigned long buildChecksumCPU(int* outputColumn, size_t outputCount);
unsigned long buildChecksumOMP(int* outputColumn, size_t outputCount){
#pragma offload_transfer target(mic:0) out(outputColumn: length(outputCount) alloc_if(0) free_if(0))
	return buildChecksumCPU(outputColumn, outputCount);
};

int* performOMPFKJoin(int* restrict outputColumn, const int* const restrict inputColumn, const size_t inputCount, const int* const restrict lookupColumn, const size_t lookupColumnCount){
	const unsigned int threadCount = numberOfThreads;
#pragma offload target(mic:0) in(inputColumn:length(0) align(64) alloc_if(0) free_if(0)) out(outputColumn:length(0) alloc_if(0) free_if(0)) in(lookupColumn:length(0) alloc_if(0) free_if(0)) in(inputCount)
	{
#pragma omp parallel for num_threads(threadCount)
#pragma novector
		for (size_t i = 0; i < inputCount; i++)
			outputColumn[i] = lookupColumn[inputColumn[i]];
	}
	return outputColumn;
};


int* prepareOMPFKJoin(int* restrict outputColumn, const int* const restrict inputColumn, const size_t inputCount, const int* const restrict lookupColumn, const size_t lookupColumnCount){
#pragma offload_transfer target(mic:0) in(inputColumn:length(inputCount) alloc_if(1) align(64) free_if(0)) in(lookupColumn:length(lookupColumnCount) free_if(0) alloc_if(1)) nocopy(outputColumn:length(inputCount) free_if(0) alloc_if(1))
	performOMPFKJoin(outputColumn, inputColumn, inputCount, lookupColumn, lookupColumnCount);
	return outputColumn;
};


int* doOMPReorder(int* restrict outputColumn, const int* const restrict inputColumn, const size_t inputCount, const int* const restrict lookupColumn, const size_t lookupColumnCount, unsigned int forReal){
	const unsigned int threadCount = numberOfThreads;
#pragma offload target(mic:0) in(inputColumn:length(0) align(64) alloc_if(0) free_if(0)) out(outputColumn:length(0) alloc_if(0) free_if(0)) in(lookupColumn:length(0) alloc_if(0) free_if(0)) in(inputCount) in(forReal)
	{
		if(forReal) {
#pragma omp parallel for num_threads(threadCount)
#pragma novector
			for (size_t i = 0; i < inputCount; i++)
				outputColumn[inputColumn[i]] = inputColumn[i];
		}
	}
	return outputColumn;
}

int* performOMPReorder(int* restrict outputColumn, const int* const restrict inputColumn, const size_t inputCount, const int* const restrict lookupColumn, const size_t lookupColumnCount){
	return doOMPReorder(outputColumn, inputColumn, inputCount, lookupColumn, lookupColumnCount, 1);
};


int* prepareOMPReorder(int* restrict outputColumn, const int* const restrict inputColumn, const size_t inputCount, const int* const restrict lookupColumn, const size_t lookupColumnCount){
#pragma offload_transfer target(mic:0) in(inputColumn:length(inputCount) alloc_if(1) align(64) free_if(0)) in(lookupColumn:length(lookupColumnCount) free_if(0) alloc_if(1)) nocopy(outputColumn:length(inputCount) free_if(0) alloc_if(1))
	doOMPReorder(outputColumn, inputColumn, inputCount, lookupColumn, lookupColumnCount, 0);
	return outputColumn;
};



//////////////////// Project ////////////////////
int* prepareOMPProject(int* restrict outputColumn, const int* const restrict inputColumn, const size_t inputCount, const int* const restrict lookupColumn, const size_t lookupColumnCount){
	const unsigned int strideConst = stride;
#pragma offload_transfer target(mic:0) in(inputColumn:length(inputCount) alloc_if(1) align(64) free_if(0)) in(lookupColumn:length(lookupColumnCount) free_if(0) alloc_if(1)) nocopy(outputColumn:length(inputCount) free_if(0) alloc_if(1))
	if(1) performOMPProject(outputColumn, inputColumn, inputCount, lookupColumn, lookupColumnCount);
	return outputColumn;

};


int* performOMPProject(int* restrict outputColumn, const int* const restrict inputColumn, const size_t inputCount, const int* const restrict lookupColumn, const size_t lookupColumnCount){
const unsigned int threadCount = numberOfThreads;
	const unsigned int strideConst = stride;
	if(__builtin_popcount(inputCount) != 1){
		printf("inputCount not a power of two: %lu\n", inputCount);
	}
#pragma offload target(mic:0) in(inputColumn:length(0) align(64) alloc_if(0) free_if(0)) out(outputColumn:length(0) alloc_if(0) free_if(0)) in(lookupColumn:length(0) alloc_if(0) free_if(0)) /* in(clamp) */ in(strideConst)
	{
		const int inputMask = inputCount-1;
#pragma omp parallel for num_threads(threadCount)
		/* #pragma novector */
		for (int i = 0, j = 0; i < inputCount; i++){
			outputColumn[i] = inputColumn[(j+=strideConst)&inputMask];
		}

	}
	return outputColumn;
};

//////////////////// Murmur ////////////////////
int* prepareOMPMurmur(int* restrict outputColumn, const int* const restrict inputColumn, const size_t inputCount, const int* const restrict lookupColumn, const size_t lookupColumnCount){
	const unsigned int strideConst = stride;
#pragma offload_transfer target(mic:0) in(inputColumn:length(inputCount) alloc_if(1) align(64) free_if(0)) nocopy(outputColumn:length(inputCount) free_if(0) alloc_if(1))
	if(1) performOMPMurmur(outputColumn, inputColumn, inputCount, lookupColumn, lookupColumnCount);
	return outputColumn;

};


static const unsigned int
#if defined(__ICC) || defined(__INTEL_COMPILER)
__attribute__((target(mic)))
#endif
c1 = 0xcc9e2d51;
static const unsigned int
#if defined(__ICC) || defined(__INTEL_COMPILER)
__attribute__((target(mic)))
#endif
c2 = 0x1b873593;
static const unsigned int
#if defined(__ICC) || defined(__INTEL_COMPILER)
__attribute__((target(mic)))
#endif
r1 = 15;
static const unsigned int
#if defined(__ICC) || defined(__INTEL_COMPILER)
__attribute__((target(mic)))
#endif
r2 = 13;
static const unsigned int
#if defined(__ICC) || defined(__INTEL_COMPILER)
__attribute__((target(mic)))
#endif
m = 5;
static const unsigned int
#if defined(__ICC) || defined(__INTEL_COMPILER)
__attribute__((target(mic)))
#endif
n = 0xe6546b64;

inline unsigned int
#if defined(__ICC) || defined(__INTEL_COMPILER)
__attribute__((target(mic)))
#endif
murmur3_32(const unsigned int key) {
  unsigned int hash = 312779;

  const int nblocks = 1;
	{
    unsigned int k = key;
    k *= c1;
    k = (k << r1) | (k >> (32 - r1));
    k *= c2;

    hash ^= k;
    hash = ((hash << r2) | (hash >> (32 - r2))) * m + n;
  }

  hash ^= 4;
  hash ^= (hash >> 16);
  hash *= 0x85ebca6b;
  hash ^= (hash >> 13);
  hash *= 0xc2b2ae35;
  hash ^= (hash >> 16);

  return hash;
}



int* performOMPMurmur(int* restrict outputColumn, const int* const restrict inputColumn, const size_t inputCount, const int* const restrict lookupColumn, const size_t lookupColumnCount){
	const unsigned int threadCount = numberOfThreads;
	const unsigned int iterations = forIterationPerKernelInvocation;
#pragma offload target(mic:0) in(inputColumn:length(0) align(64) alloc_if(0) free_if(0)) out(outputColumn:length(0) alloc_if(0) free_if(0) align(64)) /* in(lookupColumn:length(0) alloc_if(0) free_if(0)) */
	{
#pragma omp parallel for num_threads(threadCount)
		for (int i = 0, j = 0; i < inputCount; i++){
			/* outputColumn[i] = murmur3_32(murmur3_32(murmur3_32(murmur3_32(inputColumn[i])))); */
			unsigned int tmp = inputColumn[i];
			for (size_t j = 0; j < iterations; j++) {
				tmp = murmur3_32(tmp);
			}
			outputColumn[i] = tmp;
		}

	}
	return outputColumn;
};


//////////////////// Coherency Pingpong ////////////////////

int* prepareOMPCoherencyPingpong(int* restrict outputColumn, const int* const restrict inputColumn, const size_t inputCount, const int* const restrict lookupColumn, const size_t lookupColumnCount){
#pragma offload_transfer target(mic:0) in(inputColumn:length(inputCount) alloc_if(1) align(64) free_if(0)) in(lookupColumn:length(1) free_if(0) alloc_if(1)) nocopy(outputColumn:length(inputCount) free_if(0) alloc_if(1))
	if(1)
		performOMPCoherencyPingpong(outputColumn, inputColumn, inputCount, lookupColumn, lookupColumnCount);
	return outputColumn;

};

int* performOMPCoherencyPingpong(int* restrict outputColumn, const int* const restrict inputColumn, const size_t inputCount, const int* const restrict lookupColumn, const size_t lookupColumnCount){
	const unsigned int threadCount = numberOfThreads;
#pragma offload target(mic:0) in(inputColumn:length(0) align(64) alloc_if(0) free_if(0)) out(outputColumn:length(0) alloc_if(0) free_if(0)) in(lookupColumn:length(0) alloc_if(0) free_if(0)) in(inputCount)
	{
#pragma omp parallel for num_threads(threadCount) shared(inputCount)
		for (int thread = 0; thread < threadCount; thread++) {
			int position = (thread%(threadCount/2))*16;
			int stateBit = thread>=(threadCount/2);
			outputColumn[position] = 0;
			register int __attribute__((target(mic))) currentValue = 0;
#pragma novector
			while(currentValue<inputCount){
				currentValue = __sync_add_and_fetch(outputColumn + position, (stateBit!=(currentValue&1)));
			}
		}
	}
	return outputColumn;
};



/* Local Variables: */
/* compile-command: "icc -opt-streaming-stores always -openmp -O3 -std=c99 -march=native -mtune=native -o main *.c $CL_FLAGS" */
/* End: */
