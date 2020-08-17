#ifndef _LIGHT_H_
#define _LIGHT_H_

#include "common.h"
/*
 * Different types of light
 */

struct DirectionalLight
{
    glm::vec3 direction;
    glm::vec3 ambient;
    glm::vec3 diffuse;
    glm::vec3 specular;
};
#endif