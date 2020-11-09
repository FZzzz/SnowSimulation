#ifndef _TOOL_CUH_
#define _TOOL_CUH_

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_math.h>
#include <helper_functions.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/scan.h>

void cuda_tool_fill_uint(uint* d_arr, size_t start, size_t end, uint value);
void cuda_tool_fill_float(float* d_arr, size_t start, size_t end, float value);

#endif