#ifndef _SPH_KERNEL_H_
#define _SPH_KERNEL_H_

#include "common.h"

class SPHKernel
{
public:
	static float Poly6_W(float distance, float effective_radius)
	{
		if (distance >= 0 && distance <= effective_radius)
		{
			const double h = static_cast<double>(effective_radius);
			const double d = static_cast<double>(distance);

			double h2 = h * h;
			double h9 = glm::pow(h, 9);
			double d2 = d * d;
			double q = h2 - d2;
			double q3 = q * q * q;
			
			double result = (315.0 / (64.0 * M_PI * h9)) * q3;

			return static_cast<float>(result);
		}
		else
		{
			return 0.0f;
		}
	}

	static glm::vec3 Poly6_W_Gradient(glm::vec3 diff, float distance, float effective_radius)
	{
		if (distance >= 0 && distance <= effective_radius)
		{
			const double h = static_cast<double>(effective_radius);
			const double d = static_cast<double>(distance);

			double h2 = h * h;
			double h9 = glm::pow(h, 9);
			double d2 = d * d;
			double  q = h2 - d2;
			double q2 = q * q;

			double scalar = (-945.0 / (32.0 * M_PI * h9));
			glm::vec3 result = static_cast<float>(scalar) * static_cast<float>(q2) * diff;

			return result;
		}
		else
		{
			return glm::vec3(0);
		}
	}

	static float Spiky_W(float distance, float effective_radius)
	{
		if (distance >= 0 && distance <= effective_radius)
		{
			const double h = static_cast<double>(effective_radius);
			const double d = static_cast<double>(distance);

			double h6 = glm::pow(h, 6);
			double q = h - d;
			double q3 = q * q * q;

			float result = static_cast<float>((15.0 / (M_PI * h6)) * q3);

			return result;
		}
		else
		{
			return 0.0f;
		}
	}


	static glm::vec3 Spiky_W_Gradient(glm::vec3 diff, float distance, float effective_radius)
	{
		if (distance >= 0 && distance <= effective_radius)
		{
			const double h = static_cast<double>(effective_radius);
			const double d = static_cast<double>(distance);
			double h6 = glm::pow(h, 6);
			double q = h - d;
			double q2 = q * q;

			double scalar = (-45.0 / (M_PI * h6)) * (q2 / distance);
			glm::vec3 result = static_cast<float>(scalar) * diff;

			return result;
		}
		else
		{
			return glm::vec3(0);
		}
	}


};

#endif
