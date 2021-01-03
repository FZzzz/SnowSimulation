#ifndef _UTIL_FUNC_H_
#define _UTIL_FUNC_H_

#include "common.h"
#include <fstream>
#include <iostream>



namespace UtilFunctions
{
	glm::quat RotateBetweenVectors(glm::vec3 u, glm::vec3 v)
	{
		u = glm::normalize(u);
		v = glm::normalize(v);

		float cosine = glm::dot(u, v);
		glm::vec3 rotation_axis;

		if (cosine < -1 + 0.001f)
		{
			rotation_axis = glm::cross(glm::vec3(0, 0, 1), u);
			if (glm::length2(rotation_axis) < 0.01f)
			{
				rotation_axis = glm::cross(glm::vec3(1, 0, 0), u);
			}
			rotation_axis = normalize(rotation_axis);
			return glm::angleAxis(glm::radians(180.0f), rotation_axis);
		}

		rotation_axis = glm::cross(u, v);
		float s = sqrt((1 + cosine) * 2);
		float invs = 1 / s;

		return glm::quat(
			s * 0.5f,
			rotation_axis.x * invs,
			rotation_axis.y * invs,
			rotation_axis.z * invs
			);

	}
}


#endif
