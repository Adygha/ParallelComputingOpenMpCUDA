# ParallelComputingOpenMpCUDA
A project for the 'Parallel Computing' course at Linnaeus University using OpenMP and CUDA APIs and measure their calculation wall-times.

# Build
On university's server, 'nvcc' ('nvcc' uses 'gcc' for host code on university's server) was used with these arguments:
```
nvcc -gencode=arch=compute_52,code=\"sm_52,compute_52\" -O0 -Xcompiler "-fopenmp -O0" Loader.cu -o Loader.cud
```
for a 'Loader.cud' executable file (using '-O0' on both 'nvcc' and 'gcc' to try and disable optimization).
