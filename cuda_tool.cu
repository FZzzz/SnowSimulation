#include "cuda_tool.cuh"

void cuda_tool_fill_uint(uint* d_arr, size_t start, size_t end, uint value)
{
	thrust::device_ptr<uint> d_ptr = thrust::device_pointer_cast(d_arr);
	thrust::fill(d_ptr + start, d_ptr + end, value);
}
