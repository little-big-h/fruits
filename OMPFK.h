
unsigned long buildChecksumOMP(int* outputColumn, size_t outputCount);
int* performOMPFKJoin(int* restrict outputColumn, const int* const restrict inputColumn, const size_t inputCount, const int* const restrict lookupColumn, const size_t lookupColumnCount);
int* prepareOMPFKJoin(int* restrict outputColumn, const int* const restrict inputColumn, const size_t inputCount, const int* const restrict lookupColumn, const size_t lookupColumnCount);


int* prepareOMPProject(int* restrict outputColumn, const int* const restrict inputColumn, const size_t inputCount, const int* const restrict lookupColumn, const size_t lookupColumnCount);
int* performOMPProject(int* restrict outputColumn, const int* const restrict inputColumn, const size_t inputCount, const int* const restrict lookupColumn, const size_t lookupColumnCount);

int* performOMPReorder(int* restrict outputColumn, const int* const restrict inputColumn, const size_t inputCount, const int* const restrict lookupColumn, const size_t lookupColumnCount);
int* prepareOMPReorder(int* restrict outputColumn, const int* const restrict inputColumn, const size_t inputCount, const int* const restrict lookupColumn, const size_t lookupColumnCount);

int* prepareOMPCoherencyPingpong(int* restrict outputColumn, const int* const restrict inputColumn, const size_t inputCount, const int* const restrict lookupColumn, const size_t lookupColumnCount);
int* performOMPCoherencyPingpong(int* restrict outputColumn, const int* const restrict inputColumn, const size_t inputCount, const int* const restrict lookupColumn, const size_t lookupColumnCount);


int* prepareOMPMurmur(int* restrict outputColumn, const int* const restrict inputColumn, const size_t inputCount, const int* const restrict lookupColumn, const size_t lookupColumnCount);
int* performOMPMurmur(int* restrict outputColumn, const int* const restrict inputColumn, const size_t inputCount, const int* const restrict lookupColumn, const size_t lookupColumnCount);
