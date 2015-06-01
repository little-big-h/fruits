#define _GNU_SOURCE
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <string.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif
#include <execinfo.h>
#include <getopt.h>
#include "Common.h"


////////////////////////////// CPUFK //////////////////////////////
int* performCPUFKJoin(int* outputColumn, int* inputColumn, size_t inputCount, int* lookupColumn, size_t lookupColumnCount){
#pragma omp parallel for
	for (size_t i = 0; i < inputCount; i++)
		outputColumn[i] = lookupColumn[inputColumn[i]];
	return outputColumn;
}

int* prepareCPUFKJoin(int* outputColumn, int* inputColumn, size_t inputCount, int* lookupColumn, size_t lookupColumnCount) {
	return outputColumn;
}

unsigned long buildChecksumCPU(int* outputColumn, size_t outputCount){
	unsigned long checksum = 0;
	if(strcmp(experimentName, "OMPReorder") == 0 || strcmp(experimentName, "GPUReorder") == 0){
		checksum = stride;
		for (size_t i = stride; i < outputCount; i+=stride)
			checksum += (outputColumn[i]-outputColumn[i-stride]);
	} else {
		for (size_t i = 0; i < outputCount; i++)
			checksum += (outputColumn[i]&255)*(i%7);
	}
	return checksum;
}
