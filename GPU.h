#ifndef GPUFK_H
#define GPUFK_H
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif



typedef struct {
	int initialized;
	cl_context context;
	cl_command_queue queue;
	cl_device_id device;
	cl_platform_id platform;
	char* usedDeviceType;
	cl_program program;
} CLStuff;

unsigned long buildChecksumGPU(int* outputColumn, size_t outputCount);
int* prepareGPUFKJoin(int* outputColumn, int* inputColumn, size_t inputCount, int* lookupColumn, size_t lookupColumnCount);
int* performGPUFKJoin(int* outputColumn, int* inputColumn, size_t inputCount, int* lookupColumn, size_t lookupColumnCount);
CLStuff getCLStuff();


//////////////////// Sum ////////////////////

int* prepareGPUSum(int* outputColumn, int* inputColumn, size_t inputCount, int* lookupColumn, size_t lookupColumnCount);
int* performGPUSum(int* outputColumn, int* inputColumn, size_t inputCount, int* lookupColumn, size_t lookupColumnCount);



//////////////////// Project ////////////////////
int* prepareGPUProject(int* outputColumn, int* inputColumn, size_t inputCount, int* lookupColumn, size_t lookupColumnCount);
int* performGPUProject(int* outputColumn, int* inputColumn, size_t inputCount, int* lookupColumn, size_t lookupColumnCount);

//////////////////// LinearProbe ////////////////////
int* prepareGPULinearProbe(int* outputColumn, int* inputColumn, size_t inputCount, int* lookupColumn, size_t lookupColumnCount);
int* performGPULinearProbe(int* outputColumn, int* inputColumn, size_t inputCount, int* lookupColumn, size_t lookupColumnCount);

//////////////////// Murmur ////////////////////
int* performGPUMurmur(int* outputColumn, int* inputColumn, size_t inputCount, int* lookupColumn, size_t lookupColumnCount);
int* prepareGPUMurmur(int* outputColumn, int* inputColumn, size_t inputCount, int* lookupColumn, size_t lookupColumnCount);


//////////////////// Reorder ////////////////////
int* performGPUReorder(int* outputColumn, int* inputColumn, size_t inputCount, int* lookupColumn, size_t lookupColumnCount);
int* prepareGPUReorder(int* outputColumn, int* inputColumn, size_t inputCount, int* lookupColumn, size_t lookupColumnCount);

#endif /* GPUFK_H */
