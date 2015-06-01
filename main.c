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

int verbose = 0;
unsigned int stride = 1;
unsigned int prefetchWidth = 0;
unsigned int uniqueItems = 0;
unsigned int forIterationPerKernelInvocation = 1;
extern unsigned int numberOfThreads = 236;
char* experimentName = "GPUFK";
size_t multiplier = 1;
size_t domain = 0;
unsigned long buildChecksumCPU(int* outputColumn, size_t outputCount);
int* prepareCPUFKJoin(int* outputColumn, int* inputColumn, size_t inputCount, int* lookupColumn, size_t lookupColumnCount);
int* performCPUFKJoin(int* outputColumn, int* inputColumn, size_t inputCount, int* lookupColumn, size_t lookupColumnCount);

////////////////////////////// GPUFK //////////////////////////////

#include "GPU.h"
#include "OMPFK.h"

////////////////////////////// Framework //////////////////////////////




typedef int* (*implementation)(int* output, int* input, size_t inputCount, int* lookup, size_t lookupSize);
typedef unsigned long (*buildChecksum)(int* output, size_t outputCount);
typedef struct {implementation prepare;implementation perform; buildChecksum buildChecksum; char* name;} experiment;

const experiment* findExperiment(const experiment implementationList[16], char* name){
	for (int i = 0; i < 16; i++) {
		if(implementationList[i].name && (strcmp(implementationList[i].name, name) == 0))
			return implementationList+i;
	}
	return NULL;
}

int main(int argc, char *argv[]) {
	size_t inputCount = 1024*1024*512/sizeof(int);//500MB
	size_t lookupCount = 1024*32/sizeof(int);//32K
	int sequentialLookups = 0;
	int c;
	while((c = getopt(argc, argv, "h:d:f:e:l:t:p:m:i:vsu")) != -1){
		if(c == 'l'){
			lookupCount = atoi(optarg)/sizeof(int);
		} else if(c == 'f'){
			forIterationPerKernelInvocation = atoi(optarg);
		} else if(c == 'h'){
			numberOfThreads = atoi(optarg);
		} else if(c == 'd'){
			domain = atoi(optarg);
		} else if(c == 's'){
			sequentialLookups = 1;
		} else if(c == 'p'){
			prefetchWidth = atoi(optarg);
		} else if(c == 't'){
			stride = atoi(optarg);
		} else if(c == 'i'){
			inputCount = atoi(optarg)/sizeof(int);
		} else if(c == 'u'){
			uniqueItems = 1;
		} else if(c == 'm'){
			multiplier = atoi(optarg);
		} else if(c == 'e'){
			experimentName = optarg;
		} else if(c == 'v'){
			verbose = 1;
		} else {
			abort();
		}
	}
	if(domain == 0) domain = inputCount;
	if(verbose) printf("lookupCount: %zu, inputCount: %zu\n", lookupCount, inputCount);

	int* inputColumn = malloc(sizeof(int)*inputCount);
	int* lookupColumn = (strstr(experimentName, "FK") != 0)?malloc(sizeof(int)*lookupCount):calloc(1, sizeof(int));


	const experiment definedExperiments[16] = {
		{.name = "CPUFK", .perform = &performCPUFKJoin, .prepare = &prepareCPUFKJoin, .buildChecksum=&buildChecksumCPU},
		{.name = "GPUFK", .perform = &performGPUFKJoin, .prepare = &prepareGPUFKJoin, .buildChecksum=&buildChecksumGPU},
		{.name = "OMPFK", .perform = &performOMPFKJoin, .prepare = &prepareOMPFKJoin, .buildChecksum=&buildChecksumOMP},
		{.name = "GPUSum", .perform = &performGPUSum, .prepare = &prepareGPUSum, .buildChecksum=&buildChecksumGPU},
		{.name = "GPUProject", .perform = &performGPUProject, .prepare = &prepareGPUProject, .buildChecksum=&buildChecksumGPU},
		{.name = "OMPProject", .perform = &performOMPProject, .prepare = &prepareOMPProject, .buildChecksum=&buildChecksumOMP},
		{.name = "OMPCoherencyPingpong", .perform = &performOMPCoherencyPingpong, .prepare = &prepareOMPCoherencyPingpong, .buildChecksum=&buildChecksumOMP},
		{.name = "OMPReorder", .perform = &performOMPReorder, .prepare = &prepareOMPReorder, .buildChecksum=&buildChecksumOMP},
		{.name = "GPUReorder", .perform = &performGPUReorder, .prepare = &prepareGPUReorder, .buildChecksum=&buildChecksumGPU},
		{.name = "GPULinearProbe", .perform = &performGPULinearProbe, .prepare = &prepareGPULinearProbe, .buildChecksum=&buildChecksumGPU},
		{.name = "GPUMurmur", .perform = &performGPUMurmur, .prepare = &prepareGPUMurmur, .buildChecksum=&buildChecksumGPU},
		{.name = "OMPMurmur", .perform = &performOMPMurmur, .prepare = &prepareOMPMurmur, .buildChecksum=&buildChecksumOMP}
	};


	if(uniqueItems){
		if(inputCount > RAND_MAX){
			printf("max rand value (%d) is less than inputCount (%d)\n", RAND_MAX, inputCount);
			exit(1);
		}
		for (size_t i = 0; i < inputCount;)
			for (size_t j = 0; (j < domain && i < inputCount); j+=multiplier)
				inputColumn[i++] = j;

		for (size_t i = 0; i < inputCount; i+=stride) {
			unsigned int tmp = inputColumn[i];
			size_t newPosition = (rand()%(inputCount/stride))*stride;
			inputColumn[i] = inputColumn[newPosition];
			inputColumn[newPosition] = tmp;
		}

	} else {
		for (size_t i = 0; i < inputCount; i++)
			inputColumn[i] = ((sequentialLookups?i:rand())%lookupCount)*multiplier;
	}
	if(strstr(experimentName, "FK") != 0)
		for (size_t i = 0; i < lookupCount; i++)
			lookupColumn[i] = rand();

	int* outputColumn = malloc(sizeof(int)* inputCount);

	{
		const experiment* selectedImplementation = findExperiment(definedExperiments, experimentName);
		if(!selectedImplementation){
			printf("no such experiment: %s\n", experimentName);
			exit(1);
		}

		struct timeval before, after, beforePrepare;
		gettimeofday(&beforePrepare, NULL);
		selectedImplementation->prepare(outputColumn, inputColumn, inputCount, lookupColumn, lookupCount);
		gettimeofday(&before, NULL);
		selectedImplementation->perform(outputColumn, inputColumn, inputCount, lookupColumn, lookupCount);
		gettimeofday(&after, NULL);
		printf ("{\"timeInMicroseconds\": %ld, \"preparationTimeInMicroseconds\": %ld, \"checksum\": %lu, \"deviceType\": \"%s\", \"inputSizeInBytes\": %zu,"
						" \"lookupSizeInBytes\": %zu, \"forIterationPerKernelInvocation\":%u, \"stride\":%u, \"prefetchWidth\":%u, \"multiplier\":%lu, \"threads\":%lu}\n",
						(after.tv_sec*1000000+after.tv_usec)-(before.tv_sec*1000000+before.tv_usec),
						(before.tv_sec*1000000+before.tv_usec)-(beforePrepare.tv_sec*1000000+beforePrepare.tv_usec),
						selectedImplementation->buildChecksum(outputColumn, inputCount),
						getCLStuff().usedDeviceType, inputCount*sizeof(int), lookupCount*sizeof(int),
						forIterationPerKernelInvocation, stride, prefetchWidth, multiplier, numberOfThreads);
	}


	unsigned long sum = 0;
	for (size_t i = 0; i < inputCount; i++)
		sum+=outputColumn[i];



	free(outputColumn);
	free(inputColumn);
	free(lookupColumn);

	return 0;
}
