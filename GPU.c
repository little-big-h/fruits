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
#include "GPU.h"
#include <math.h>
#include "Common.h"

void checkError(int err, char* operation){

	static const char* codemap[]
		= {[-CL_SUCCESS] = "CL_SUCCESS",
			 [-CL_DEVICE_NOT_FOUND] = "CL_DEVICE_NOT_FOUND",
			 [-CL_DEVICE_NOT_AVAILABLE] = "CL_DEVICE_NOT_AVAILABLE",
			 [-CL_COMPILER_NOT_AVAILABLE] = "CL_COMPILER_NOT_AVAILABLE",
			 [-CL_MEM_OBJECT_ALLOCATION_FAILURE] = "CL_MEM_OBJECT_ALLOCATION_FAILURE",
			 [-CL_OUT_OF_RESOURCES] = "CL_OUT_OF_RESOURCES",
			 [-CL_OUT_OF_HOST_MEMORY] = "CL_OUT_OF_HOST_MEMORY",
			 [-CL_PROFILING_INFO_NOT_AVAILABLE] = "CL_PROFILING_INFO_NOT_AVAILABLE",
			 [-CL_MEM_COPY_OVERLAP] = "CL_MEM_COPY_OVERLAP",
			 [-CL_IMAGE_FORMAT_MISMATCH] = "CL_IMAGE_FORMAT_MISMATCH",
			 [-CL_IMAGE_FORMAT_NOT_SUPPORTED] = "CL_IMAGE_FORMAT_NOT_SUPPORTED",
			 [-CL_BUILD_PROGRAM_FAILURE] = "CL_BUILD_PROGRAM_FAILURE",
			 [-CL_MAP_FAILURE] = "CL_MAP_FAILURE",
			 [-CL_INVALID_VALUE] = "CL_INVALID_VALUE",
			 [-CL_INVALID_DEVICE_TYPE] = "CL_INVALID_DEVICE_TYPE",
			 [-CL_INVALID_PLATFORM] = "CL_INVALID_PLATFORM",
			 [-CL_INVALID_DEVICE] = "CL_INVALID_DEVICE",
			 [-CL_INVALID_CONTEXT] = "CL_INVALID_CONTEXT",
			 [-CL_INVALID_QUEUE_PROPERTIES] = "CL_INVALID_QUEUE_PROPERTIES",
			 [-CL_INVALID_COMMAND_QUEUE] = "CL_INVALID_COMMAND_QUEUE",
			 [-CL_INVALID_HOST_PTR] = "CL_INVALID_HOST_PTR",
			 [-CL_INVALID_MEM_OBJECT] = "CL_INVALID_MEM_OBJECT",
			 [-CL_INVALID_IMAGE_FORMAT_DESCRIPTOR] = "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR",
			 [-CL_INVALID_IMAGE_SIZE] = "CL_INVALID_IMAGE_SIZE",
			 [-CL_INVALID_SAMPLER] = "CL_INVALID_SAMPLER",
			 [-CL_INVALID_BINARY] = "CL_INVALID_BINARY",
			 [-CL_INVALID_BUILD_OPTIONS] = "CL_INVALID_BUILD_OPTIONS",
			 [-CL_INVALID_PROGRAM] = "CL_INVALID_PROGRAM",
			 [-CL_INVALID_PROGRAM_EXECUTABLE] = "CL_INVALID_PROGRAM_EXECUTABLE",
			 [-CL_INVALID_KERNEL_NAME] = "CL_INVALID_KERNEL_NAME",
			 [-CL_INVALID_KERNEL_DEFINITION] = "CL_INVALID_KERNEL_DEFINITION",
			 [-CL_INVALID_KERNEL] = "CL_INVALID_KERNEL",
			 [-CL_INVALID_ARG_INDEX] = "CL_INVALID_ARG_INDEX",
			 [-CL_INVALID_ARG_VALUE] = "CL_INVALID_ARG_VALUE",
			 [-CL_INVALID_ARG_SIZE] = "CL_INVALID_ARG_SIZE",
			 [-CL_INVALID_KERNEL_ARGS] = "CL_INVALID_KERNEL_ARGS",
			 [-CL_INVALID_WORK_DIMENSION] = "CL_INVALID_WORK_DIMENSION",
			 [-CL_INVALID_WORK_GROUP_SIZE] = "CL_INVALID_WORK_GROUP_SIZE",
			 [-CL_INVALID_WORK_ITEM_SIZE] = "CL_INVALID_WORK_ITEM_SIZE",
			 [-CL_INVALID_GLOBAL_OFFSET] = "CL_INVALID_GLOBAL_OFFSET",
			 [-CL_INVALID_EVENT_WAIT_LIST] = "CL_INVALID_EVENT_WAIT_LIST",
			 [-CL_INVALID_EVENT] = "CL_INVALID_EVENT",
			 [-CL_INVALID_OPERATION] = "CL_INVALID_OPERATION",
			 [-CL_INVALID_GL_OBJECT] = "CL_INVALID_GL_OBJECT",
			 [-CL_INVALID_BUFFER_SIZE] = "CL_INVALID_BUFFER_SIZE",
			 [-CL_INVALID_MIP_LEVEL] = "CL_INVALID_MIP_LEVEL",
			 [-CL_INVALID_GLOBAL_WORK_SIZE] = "CL_INVALID_GLOBAL_WORK_SIZE" };
	if(err) {
		void* buffer[128];
		int backtraceSize = backtrace(buffer, 128);
		char** symbols = backtrace_symbols(buffer, backtraceSize);
		for (int i = 0; i < backtraceSize; i++)
			printf("%d  %s\n", i, symbols[i]);
		printf("ERROR %s (%d) during %s, AAARRRRGGGHHHHHH!!!!!\n", codemap[-err], err, operation);
		exit(err);
	}
}


#define MAX_MEM_OBJECTS 64
struct {
	void* CPUObject;
	cl_mem GPUObject;
} CPUToGPUObjectMapping[MAX_MEM_OBJECTS] = {};

cl_mem lookupGPUObject (void* CPUObject){
	for (int i = 0; i < MAX_MEM_OBJECTS; i++) {
    if(CPUToGPUObjectMapping[i].CPUObject == CPUObject)
			return CPUToGPUObjectMapping[i].GPUObject;
	}
	printf("lookup failed\n" );
	return NULL;
}



CLStuff initCLStuff(CLStuff stuff){
	if(!stuff.initialized){
		cl_int err;
		/* cl_device_id device; */
		/* cl_platform_id platform; */
		{
			cl_uint num_platforms;
			cl_platform_id platforms[4];
			clGetPlatformIDs(4, platforms, &num_platforms);
			stuff.platform = platforms[0];
			if(num_platforms > 1){printf("found more than one platform: %d\n", num_platforms );exit(1);}
			if(num_platforms == 0){printf("no platforms found\n" );exit(1);}
			if(verbose) printf("found %d platforms\n", num_platforms);
			stuff.usedDeviceType = "ACCELERATOR";
			cl_uint num_devices;
			err = clGetDeviceIDs(stuff.platform, CL_DEVICE_TYPE_ACCELERATOR, 1, &(stuff.device), &num_devices);
			if(err && err != CL_DEVICE_NOT_FOUND)
				checkError(err, "clGetDeviceIDs ACCELERATOR");
			if(num_devices == 0){
				stuff.usedDeviceType = "GPU";
				if(verbose) printf("switching to device class GPU\n" );
				err = clGetDeviceIDs(stuff.platform, CL_DEVICE_TYPE_GPU, 1, &(stuff.device), &num_devices);
				checkError(err, "clGetDeviceIDs ACC");
			} else {
				if(verbose) printf("sticking to device class ACCELERATOR\n" );
			}
			if(num_devices == 0){printf("no devices found\n" );exit(1);}
		}

		stuff.context = clCreateContext((cl_context_properties[]){CL_CONTEXT_PLATFORM, (cl_context_properties)stuff.platform, 0}, 1, &(stuff.device), NULL, NULL, &err);
		checkError(err, "clCreateContext");
		stuff.queue = clCreateCommandQueue(stuff.context, stuff.device, 0, &err);
		checkError(err, "clCreateCommandQueue");
		stuff.initialized = 1;
	}
	return stuff;
}

static CLStuff clStuff = {.usedDeviceType = "CPU"};

CLStuff getCLStuff(){
	return clStuff;
}


int* runProgram(cl_program program, int* outputColumn, int* inputColumn, int* lookupColumn, size_t inputCount){
	cl_int err;
	cl_kernel kernel = clCreateKernel(program, "doIt", &err);
	checkError(err, "createKernel");
	for (int i = 0; i < 3; i++) {
		err = clSetKernelArg(kernel, i, sizeof(cl_mem), (cl_mem[]){lookupGPUObject(((int*[]){outputColumn, inputColumn, lookupColumn})[i])});
		checkError(err, "clSetKernelArg");
	}
	const unsigned int spaceReduction = (strstr(experimentName, "urmur") != 0)?1:forIterationPerKernelInvocation;
	err = clEnqueueNDRangeKernel(clStuff.queue, kernel, 1, (size_t[]){0}, (size_t[]){inputCount/spaceReduction}, NULL, 0, NULL, NULL);
	checkError(err, "clEnqueueNDRangeKernel");
	err = clFinish(clStuff.queue);
	checkError(err, "finish");
	return outputColumn;
}

cl_program createProgramFromSource(char** code, unsigned int codeLineCount){
	cl_int err;
	cl_program program = clCreateProgramWithSource(clStuff.context, codeLineCount, (const char**)code, NULL, &err);
	checkError(err, "clCreateProgramWithSource");
	err = clBuildProgram(program, 1, &(clStuff.device), "", NULL, NULL);
	if(err == CL_BUILD_PROGRAM_FAILURE){
		char log[4096*4];
		clGetProgramBuildInfo(program, clStuff.device, CL_PROGRAM_BUILD_LOG, 4096*4, log, NULL);
		printf("log:\n%s\n", log);
	}
	checkError(err, "clBuildProgram");
	return program;
}

int* executeGPUFKJoin(int* outputColumn, int* inputColumn, size_t inputCount, int* lookupColumn, size_t lookupColumnCount, int prepareOnly, int cheatALittleByPreInitializingBuffer){
	cl_int err;
	if(!clStuff.program){
		char* code[] = {
			"__kernel void doIt(__global int* const restrict outputColumn, __global const int* const restrict inputColumn, __global const int* const restrict lookupColumn){\n",
			"  outputColumn[get_global_id(0)] =  lookupColumn[inputColumn[get_global_id(0)]];\n",
			"}\n"
		};
		if(forIterationPerKernelInvocation > 1)
			asprintf(code+1, "for (int i = 0; i < %1$d; i++)\n outputColumn[get_global_id(0)*%1$d+i] = lookupColumn[inputColumn[get_global_id(0)*%1$d+i]];\n", forIterationPerKernelInvocation);
		if(stride > 1) {
			asprintf(code+1,
							 "  const size_t position = ((get_global_id(0)*%1$d)&%2$lu)+((get_global_id(0)*%1$d)>>%3$d);"
							 "  outputColumn[position] =  lookupColumn[inputColumn[position]];\n",
							 stride, inputCount-1, (int)round(log2(inputCount)));
		}
		clStuff.program = createProgramFromSource(code, sizeof(code)/sizeof(code[0]));
	}
	cl_program program = clStuff.program;
	if(prepareOnly){
		char* code[] = {
			"__kernel void doIt(__global int* restrict outputColumn, __global const int* const restrict inputColumn, __global const int* const restrict lookupColumn){\n",
			cheatALittleByPreInitializingBuffer?
			"  outputColumn[get_global_id(0)] =  lookupColumn[inputColumn[get_global_id(0)]];\n":
			"  outputColumn[get_global_id(0)] = 0;\n",
			"}\n"
		};
		program = createProgramFromSource(code, sizeof(code)/sizeof(code[0]));
	}

	return runProgram(program, outputColumn, inputColumn, lookupColumn, inputCount);
}

int* performGPUFKJoin(int* outputColumn, int* inputColumn, size_t inputCount, int* lookupColumn, size_t lookupColumnCount){
	return executeGPUFKJoin(outputColumn, inputColumn, inputCount, lookupColumn, lookupColumnCount, 0, 0);
}

void copyInputData(int* outputColumn, int* inputColumn, size_t inputCount, int* lookupColumn, size_t lookupColumnCount){
	cl_int err;
	int nextFreeSlot = -1;
	for (int i = 0; (i < MAX_MEM_OBJECTS && nextFreeSlot == -1); i++) {
		if(CPUToGPUObjectMapping[i].CPUObject == NULL)
			nextFreeSlot = i;
	}
	if(nextFreeSlot == -1 || nextFreeSlot > MAX_MEM_OBJECTS-2){
		printf("not enough memobject mapping slots available\n");
		exit(1);
	}

	CPUToGPUObjectMapping[nextFreeSlot].CPUObject = inputColumn;
	CPUToGPUObjectMapping[nextFreeSlot].GPUObject = clCreateBuffer(clStuff.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, inputCount*sizeof(int), inputColumn, &err);
	checkError(err, "clCreateBuffer input");

	CPUToGPUObjectMapping[nextFreeSlot+1].CPUObject = lookupColumn;
	CPUToGPUObjectMapping[nextFreeSlot+1].GPUObject = clCreateBuffer(clStuff.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, ((strstr(experimentName, "FK") != 0)?lookupColumnCount:1)*sizeof(int), lookupColumn, &err);
	checkError(err, "clCreateBuffer lookup");

	CPUToGPUObjectMapping[nextFreeSlot+2].CPUObject = outputColumn;
	CPUToGPUObjectMapping[nextFreeSlot+2].GPUObject = clCreateBuffer(clStuff.context, CL_MEM_WRITE_ONLY, (inputCount+1)*sizeof(int), NULL, &err);
	checkError(err, "clCreateBuffer output");

}

int* prepareGPUFKJoin(int* outputColumn, int* inputColumn, size_t inputCount, int* lookupColumn, size_t lookupColumnCount){
	clStuff = initCLStuff(clStuff);
	copyInputData(outputColumn, inputColumn, inputCount, lookupColumn, lookupColumnCount);
	return executeGPUFKJoin(outputColumn, inputColumn, inputCount, lookupColumn, lookupColumnCount, 1, 1);
}

extern unsigned long buildChecksumCPU(int* outputColumn, size_t outputCount);
unsigned long buildChecksumGPU(int* outputColumn, size_t outputCount){
	cl_int err = clEnqueueReadBuffer(clStuff.queue, lookupGPUObject(outputColumn), 1, 0, (outputCount+1)*sizeof(int), outputColumn, 0, NULL, NULL);
	checkError(err, "clEnqueueReadBuffer");
	if(verbose) printf("conflicts: %d\n", outputColumn[outputCount]);
	return buildChecksumCPU(outputColumn, outputCount);
}


//////////////////// Sum ////////////////////

static cl_program sumprogram = NULL;
static char* sumcode[]= {
	"__attribute__((reqd_work_group_size(64, 1, 1)))"
	"__kernel void doIt(__global int* const restrict outputColumn, __global const int* const restrict inputColumn, __global const int* const restrict lookupColumn){\n",
	"  if(inputColumn[get_global_id(0)] > 1073741824)"
	"    outputColumn[(get_group_id(0)%(64*60*256))+get_local_id(0)] +=  inputColumn[get_global_id(0)];\n",
	"}\n"
};
int* performGPUSum(int* outputColumn, int* inputColumn, size_t inputCount, int* lookupColumn, size_t lookupColumnCount){
	return runProgram(sumprogram, outputColumn, inputColumn, lookupColumn, inputCount);
}


int* prepareGPUSum(int* outputColumn, int* inputColumn, size_t inputCount, int* lookupColumn, size_t lookupColumnCount){
	if(stride > 0) {
		asprintf(sumcode+1,
						 "  const size_t position = ((get_global_id(0)*%1$d)&%2$lu)+((get_global_id(0)*%1$d)>>%3$d);"
						 "  prefetch(inputColumn+position+$1$d*%4$d, 1);\n"
						 "  outputColumn[(get_group_id(0)%%(64*60*64))+get_local_id(0)] +=  (inputColumn[position] > 1073741824);\n",
						 stride, inputCount-1, (int)round(log2(inputCount)), prefetchWidth);
	}

	clStuff = initCLStuff(clStuff);
	copyInputData(outputColumn, inputColumn, inputCount, lookupColumn, lookupColumnCount);
	sumprogram = createProgramFromSource(sumcode, sizeof(sumcode)/sizeof(sumcode[0]));
	runProgram(sumprogram, outputColumn, inputColumn, lookupColumn, inputCount);
	return executeGPUFKJoin(outputColumn, inputColumn, inputCount, lookupColumn, lookupColumnCount, 1, 0);
}


//////////////////// Reorder ////////////////////

static cl_program reorderprogram = NULL;

int* performGPUReorder(int* outputColumn, int* inputColumn, size_t inputCount, int* lookupColumn, size_t lookupColumnCount){

	return runProgram(reorderprogram, outputColumn, inputColumn, lookupColumn, inputCount/stride);
}


int* prepareGPUReorder(int* outputColumn, int* inputColumn, size_t inputCount, int* lookupColumn, size_t lookupColumnCount){
	static char* reordercode[]= {
		"__kernel void doIt(__global int* const restrict outputColumn, __global const int* const restrict inputColumn, __global const int* const restrict lookupColumn){\n",
		"  outputColumn[inputColumn[get_global_id(0)]] = inputColumn[get_global_id(0)];",
		"}\n"
	};

	if(stride>1)
		asprintf(reordercode+1,
						 "  outputColumn[inputColumn[get_global_id(0)*%1$d]] = inputColumn[get_global_id(0)*%1$d];",
						 stride);


	clStuff = initCLStuff(clStuff);
	copyInputData(outputColumn, inputColumn, inputCount, lookupColumn, lookupColumnCount);
	reorderprogram = createProgramFromSource(reordercode, sizeof(reordercode)/sizeof(reordercode[0]));
	runProgram(reorderprogram, outputColumn, inputColumn, lookupColumn, inputCount/stride);
	return executeGPUFKJoin(outputColumn, inputColumn, inputCount, lookupColumn, lookupColumnCount, 1, 0);
}


//////////////////// Project ////////////////////

static cl_program projectprogram = NULL;

int* performGPUProject(int* outputColumn, int* inputColumn, size_t inputCount, int* lookupColumn, size_t lookupColumnCount){

	return runProgram(projectprogram, outputColumn, inputColumn, lookupColumn, inputCount/stride);
}


int* prepareGPUProject(int* outputColumn, int* inputColumn, size_t inputCount, int* lookupColumn, size_t lookupColumnCount){
	static char* projectcode[]= {
		"__kernel void doIt(__global int* const restrict outputColumn, __global const int* const restrict inputColumn, __global const int* const restrict lookupColumn){\n",
		"",
		"}\n"
	};

	asprintf(projectcode+1,
					 "    outputColumn[get_global_id(0)] = inputColumn[get_global_id(0)*%1$d];\n",
					 stride?stride:1);

	clStuff = initCLStuff(clStuff);
	copyInputData(outputColumn, inputColumn, inputCount, lookupColumn, lookupColumnCount);
	projectprogram = createProgramFromSource(projectcode, sizeof(projectcode)/sizeof(projectcode[0]));
	runProgram(projectprogram, outputColumn, inputColumn, lookupColumn, inputCount/stride);
	return executeGPUFKJoin(outputColumn, inputColumn, inputCount, lookupColumn, lookupColumnCount, 1, 0);
}


//////////////////// LinearProbe ////////////////////

static cl_program linearProbeprogram = NULL;

int* performGPULinearProbe(int* outputColumn, int* inputColumn, size_t inputCount, int* lookupColumn, size_t lookupColumnCount){

	return runProgram(linearProbeprogram, outputColumn, inputColumn, lookupColumn, inputCount/stride);
}


int* prepareGPULinearProbe(int* outputColumn, int* inputColumn, size_t inputCount, int* lookupColumn, size_t lookupColumnCount){
	static char* linearProbecode[]= {
		"__kernel void doIt(__global int* const restrict outputColumn, __global const int* const restrict inputColumn, __global const int* const restrict lookupColumn){\n",
		"  const int value = inputColumn[get_global_id(0)];\n"
		"  int iteration = 0;\n"
		"  while(iteration<33 && atomic_cmpxchg(outputColumn+(value+iteration)%get_global_size(0), 0, value)){\n"
		"    iteration++;\n"
		"  }\n"
		/* "  atomic_add(outputColumn + get_global_size(0), iteration);\n" */
		"",
		"}\n"
	};

	clStuff = initCLStuff(clStuff);
	copyInputData(outputColumn, inputColumn, inputCount, lookupColumn, lookupColumnCount);
	linearProbeprogram = createProgramFromSource(linearProbecode, sizeof(linearProbecode)/sizeof(linearProbecode[0]));
	runProgram(linearProbeprogram, outputColumn, inputColumn, lookupColumn, inputCount);
	runProgram(createProgramFromSource((char*[]){
				"__kernel void doIt(__global int* restrict outputColumn, __global const int* const restrict inputColumn, __global const int* const restrict lookupColumn){\n",
					"  outputColumn[get_global_id(0)] = 0;\n",
					"}\n"
					}, 3), outputColumn, inputColumn, lookupColumn, inputCount+1);
	return outputColumn;
}



//////////////////// Murmur ////////////////////
static char* code[]= {
	"  static const __constant unsigned int c1 = 0xcc9e2d51;\n"
	"  static const __constant unsigned int c2 = 0x1b873593;\n"
	"  static const __constant unsigned int r1 = 15;\n"
	"  static const __constant unsigned int r2 = 13;\n"
	"  static const __constant unsigned int m = 5;\n"
	"  static const __constant unsigned int n = 0xe6546b64;\n"
	" \n"
	"inline unsigned int murmur3_32(const unsigned int key) {\n"
	"  unsigned int hash = 312779;\n"
	" \n"
	"  const int nblocks = 1;\n"
	"{\n"
	"    unsigned int k = key;\n"
	"    k *= c1;\n"
	"    k = (k << r1) | (k >> (32 - r1));\n"
	"    k *= c2;\n"
	" \n"
	"    hash ^= k;\n"
	"    hash = ((hash << r2) | (hash >> (32 - r2))) * m + n;\n"
	"  }\n"
	" \n"
	"  hash ^= 4;\n"
	"  hash ^= (hash >> 16);\n"
	"  hash *= 0x85ebca6b;\n"
	"  hash ^= (hash >> 13);\n"
	"  hash *= 0xc2b2ae35;\n"
	"  hash ^= (hash >> 16);\n"
	" \n"
	"  return hash;\n"
	"}\n"
	"\n"
	"__kernel void doIt(__global int* const restrict outputColumn, __global const int* const restrict inputColumn, __global const int* const restrict lookupColumn){\n",
	"  outputColumn[get_global_id(0)] =  murmur3_32(inputColumn[get_global_id(0)]);\n",
	"}\n"
};

char** getMurmurCode(){
	if(0){
		char* hashChainString = strdup("inputColumn[get_global_id(0)]");
		for (int i = 0; i < forIterationPerKernelInvocation; i++) {
			char* newString;
			asprintf(&newString, "murmur3_32(%s)", hashChainString);
			free(hashChainString);
			hashChainString = newString;
		}
		asprintf(code+1, "  outputColumn[get_global_id(0)] =  %s;\n", hashChainString);
		free(hashChainString);
	} else {
			asprintf(code+1,
							 "  unsigned int tmp = inputColumn[get_global_id(0)];\n"
							 "  for(int i = 0; i < lookupColumn[0]; i++){\n"
							 "    tmp = murmur3_32(tmp);\n"
							 "  }\n"
							 "  outputColumn[get_global_id(0)] = tmp;"
							 /* , forIterationPerKernelInvocation */);
	}
	return code;
}

int* performGPUMurmur(int* outputColumn, int* inputColumn, size_t inputCount, int* lookupColumn, size_t lookupColumnCount){
	return runProgram(clStuff.program, outputColumn, inputColumn, lookupColumn, inputCount);
}


int* prepareGPUMurmur(int* outputColumn, int* inputColumn, size_t inputCount, int* lookupColumn, size_t lookupColumnCount){
	clStuff = initCLStuff(clStuff);
	lookupColumn[0] = forIterationPerKernelInvocation;
	copyInputData(outputColumn, inputColumn, inputCount, lookupColumn, lookupColumnCount);
	int* result = executeGPUFKJoin(outputColumn, inputColumn, inputCount, lookupColumn, lookupColumnCount, 1, 0);
	clStuff.program = createProgramFromSource(getMurmurCode(), 3);
	return result;
}
