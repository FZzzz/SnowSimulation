#ifndef _CAMERA_H_
#define _CAMERA_H_

#include <glm\glm.hpp>
#include <memory>
#include "GameObject.h"

/*
	Simple look at camera
*/


struct CameraDesc
{
	float fov;
	float screen_width;
	float screen_height;
	float near_plane;
	float far_plane;
	glm::vec3 target_position;
	glm::vec3 position;
	GLuint ubo;
	
};

class Camera
{
public:
	Camera(CameraDesc desc);
	Camera(glm::mat4 projection, glm::mat4 lookAt, glm::vec3 position, GLuint ubo);
	~Camera();

	void Update();

	void Zoom(float fov_change);
	void Rotate(float phi_change, float theta_change);

	glm::mat4 m_cameraMat;
	glm::mat4 m_projection;
	glm::mat4 m_lookAt;
	glm::vec3 m_position;
	
	bool m_lock_on_mode;
	GLuint m_camera_ubo;
	//std::weak_ptr<GameObject> lookAtObj;
	
private:
	float m_near_plane;
	float m_far_plane;
	float m_rotate_radius;
	float m_theta = 0.0f;
	float m_phi = 0.0f;
	float m_fov;
	float m_screen_width;
	float m_screen_height;
	glm::vec3 m_target_position;
	glm::vec3 m_right;
	glm::vec3 m_up;
	glm::vec3 m_front;
};

#endif
