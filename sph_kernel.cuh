#include <cuda_runtime.h>
#include <math_constants.h>
#include <math.h>
#include <vector_functions.h>

inline __device__ float Poly6_W_CUDA(float distance, float effective_radius)
{
	if (distance >= 0 && distance <= effective_radius)
	{
		const float h = (float)(effective_radius);
		const float d = (float)(distance);

		float h2 = h * h;
		float h9 = pow(h, 9);
		float d2 = d * d;
		float q = h2 - d2;
		float q3 = q * q * q;

		float result = (315.0f / (64.0f * CUDART_PI * h9)) * q3;

		return (float)(result);
	}
	else
	{
		return 0.0f;
	}
}

inline __device__ float3 Poly6_W_Gradient_CUDA(float3 diff, float distance, float effective_radius)
{
	if (distance >= 0 && distance <= effective_radius)
	{
		const float h = (float)(effective_radius);
		const float d = (float)(distance);

		float h2 = h * h;
		float h9 = pow(h, 9);
		float d2 = d * d;
		float  q = h2 - d2;
		float q2 = q * q;

		float scalar = (-945.0f / (32.0f * CUDART_PI * h9));
		scalar = scalar * q2;
		float3 result = make_float3(scalar * diff.x, scalar * diff.y, scalar * diff.z);

		return result;
	}
	else
	{
		return make_float3(0,0,0);
	}
}


inline __device__ float Spiky_W_CUDA(float distance, float effective_radius)
{
	if (distance >= 0 && distance <= effective_radius)
	{
		const float h = (float)(effective_radius);
		const float d = (float)(distance);

		float h6 = pow(h, 6);
		float q = h - d;
		float q3 = q * q * q;

		float result = (float)((15.0f / (CUDART_PI * h6)) * q3);

		return result;
	}
	else
	{
		return 0.0f;
	}
}


inline __device__ float3 Spiky_W_Gradient_CUDA(float3 diff, float distance, float effective_radius)
{
	if (distance >= 0 && distance <= effective_radius)
	{
		const float h = (float)(effective_radius);
		const float d = (float)(distance);
		float h6 = pow(h, 6);
		float q = h - d;
		float q2 = q * q;

		float scalar = (-45.0f / (CUDART_PI*h6)) * (q2 / distance);
		float3 result = make_float3(scalar*diff.x, scalar*diff.y, scalar*diff.z);

		return result;
	}
	else
	{
		return make_float3(0,0,0);
	}
}