#ifndef _LIGHT_SYSTEM_H_
#define _LIGHT_SYSTEM_H_

#include "Light.h"
#include <vector>
#include <memory>
/*
 * Manage light sources
*/

class LightSystem
{
public:

    LightSystem();
    ~LightSystem();

    //getters
    inline std::shared_ptr<DirectionalLight> getDirectionalLight() { return m_directional_light; };

private:

    std::shared_ptr<DirectionalLight> m_directional_light;

};

#endif