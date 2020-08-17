#include "Camera.h"
#include "common.h"
#include <iostream>
#include "util.h"
#include "imgui/imgui.h"

Camera::Camera(CameraDesc desc) : 
	m_position(desc.position),
	m_fov(desc.fov),
	m_screen_width(desc.screen_width),
	m_screen_height(desc.screen_height),
	m_near_plane(desc.near_plane),
	m_far_plane(desc.far_plane),
	m_camera_ubo(desc.ubo),
	m_target_position(desc.target_position)
{
	m_rotate_radius = glm::distance(m_position, m_target_position);

	m_front = glm::normalize( m_target_position - m_position);
	glm::mat4 vec_rot = glm::toMat4(RotateBetweenVectors(m_front, glm::vec3(0, 0, -1)));

	m_right = glm::normalize(glm::vec3(vec_rot * glm::vec4(1, 0, 0, 1)));
	m_up = glm::normalize(glm::cross(m_right, m_front));

	m_projection = glm::perspective(m_fov, m_screen_width / m_screen_height, m_near_plane, m_far_plane);
	m_lookAt = glm::lookAt(m_position, m_target_position, m_up);

}

Camera::Camera(glm::mat4 projection, glm::mat4 lookAt, glm::vec3 position, GLuint ubo)
	: m_lock_on_mode(false),
	  m_camera_ubo(ubo),
	  m_theta(0.0f)
{
	this->m_projection = projection;
	this->m_lookAt = lookAt;
	this->m_position = position;
	m_cameraMat = projection * lookAt;
}

Camera::~Camera()
{
}

void Camera::Update()
{
	/* 
	//Recompute front-right-up (Bug :<)
	m_front = glm::normalize(m_target_position - m_position);

	 glm::mat4 vec_rot = glm::toMat4(RotateBetweenVectors(m_front, glm::vec3(0, 0, -1)));

	m_right = glm::normalize(glm::vec3(vec_rot * glm::vec4(1, 0, 0, 1)));
	m_up = glm::normalize(glm::cross(m_right, m_front));
	*/

	m_lookAt = glm::lookAt(m_position, m_target_position, glm::vec3(0,1,0));
	m_projection = glm::perspective(m_fov, m_screen_width / m_screen_height, m_near_plane, m_far_plane);

	m_cameraMat = m_projection * m_lookAt;
	
	if (m_camera_ubo != -1)
	{
		glBindBuffer(GL_UNIFORM_BUFFER, m_camera_ubo);
		glBufferSubData(
			GL_UNIFORM_BUFFER,
			0,
			sizeof(glm::mat4),
			glm::value_ptr(m_projection));
		glBufferSubData(
			GL_UNIFORM_BUFFER,
			sizeof(glm::mat4),
			sizeof(glm::mat4),
			glm::value_ptr(m_lookAt));
		glBindBuffer(GL_UNIFORM_BUFFER, 0);
	}
	/*
	{
		ImGui::Begin("Camera");
		ImGui::Text("Front:  %f, %f, %f", m_front.x, m_front.y, m_front.z);
		ImGui::Text("Right:  %f, %f, %f", m_right.x, m_right.y, m_right.z);
		ImGui::Text("Up:     %f, %f, %f", m_up.x, m_up.y, m_up.z);
		ImGui::End();
	}
	*/

	/*
	if (!lock_on_mode)
		return;
	position = lookAtObj->m_mesh->m_tranlation + glm::vec3(0, 10, -5);

	const glm::vec3 up = glm::vec3(0, 1, 0);
	const glm::vec3 front = lookAtObj->m_mesh->m_tranlation - position;
	const glm::vec3 right = glm::normalize(glm::cross(up , front));
	const glm::vec3 camUp = glm::cross(front, right);

	lookAt = glm::lookAt(position , position + front , camUp);
	*/
}

void Camera::Zoom(float fov_change)
{
	m_fov += fov_change;

	if (m_fov > 120.f) m_fov = 120.f;
	if (m_fov < 30.f) m_fov = 30;


}

void Camera::Rotate(float phi_change, float theta_change)
{
	m_phi += phi_change;
	m_theta += theta_change;

	if (m_phi > 0.49f * M_PI)
		m_phi = 0.49f * M_PI;
	if (m_phi < -0.49f * M_PI)
		m_phi = -0.49f * M_PI;

	if (m_theta > 0.49f * M_PI)
		m_theta = 0.49f * M_PI;
	if (m_theta < 0)
		m_theta = 0;


	m_position.x = m_target_position.x + m_rotate_radius * glm::cos(m_theta) * glm::sin(m_phi);
	m_position.y = m_target_position.y + m_rotate_radius * glm::sin(m_theta);
	m_position.z = m_target_position.z + m_rotate_radius * glm::cos(m_theta) * glm::cos(m_phi) ;
}
