#include "Transform.h"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/rotate_vector.hpp>

Transform::Transform() : m_translation(glm::vec3(0,0,0)) , m_eulerAngles(glm::vec3(0,0,0)) , m_modelMat(glm::mat4(1)) , 
						m_front(glm::vec3(0, 0, 1)) , m_up(glm::vec3(0, 1, 0)) , m_right(glm::vec3(1, 0, 0)) , m_scale(1.0f),
						m_parent(nullptr)
{

}

Transform::Transform(const glm::vec3 &translation, const glm::vec3 &eulerAngles, const glm::vec3 &scale) 
	: m_scale(scale), m_eulerAngles(eulerAngles), m_translation(translation)
{
	m_quaternion = glm::quat(eulerAngles);
	
	m_translationMat = glm::translate(glm::mat4(1), m_translation);
	m_rotationMat = glm::toMat4(m_quaternion);
	m_scaleMat = glm::scale(glm::mat4(1), m_scale);

	m_modelMat = m_scaleMat * m_rotationMat * m_translationMat;
}

Transform::~Transform()
{
}

void Transform::Update()
{
	/*
		Use quternion for rotation
		Model = Translation * Rotation * Scale
	*/
	m_translationMat = glm::translate(glm::mat4(1), m_translation);
	m_rotationMat = glm::toMat4(m_quaternion);
	m_scaleMat = glm::scale(glm::mat4(1), m_scale);

	// Same as Unity3D transformation order
	m_modelMat = m_translationMat * m_rotationMat * m_scaleMat;
	
	Compute_ModelMat_World();
}

void Transform::Compute_ModelMat_World()
{
	auto node = m_parent;
	m_modelMat_world = m_modelMat;
	while (node)
	{
		m_modelMat_world = node->m_modelMat * m_modelMat_world;
		node = node->m_parent;
	}

}

void Transform::CopyTo(Transform & other)
{
	// TODO: insert return statement here
	if (this != &other)
	{
		m_front = other.m_front;
		m_up = other.m_up;
		m_right = other.m_right;

		m_scale = other.m_scale;
		m_eulerAngles = other.m_eulerAngles;
		m_translation = other.m_translation;
		m_modelMat = other.m_modelMat;
		m_quaternion = other.m_quaternion;
		m_scaleMat = other.m_scaleMat;
		m_rotationMat = other.m_rotationMat;
		m_translationMat = other.m_translationMat;
	}
}

void Transform::setRotation(glm::quat input)
{
	m_quaternion = input;
	m_eulerAngles = glm::eulerAngles(m_quaternion);
}

void Transform::setRotation(glm::vec3 eulerAngles)
{
	m_eulerAngles = eulerAngles;
	m_quaternion = glm::quat(eulerAngles);
}

void Transform::setRotationDegree(glm::vec3 degree)
{
	m_eulerAngles = glm::radians(degree);
	m_quaternion = glm::quat(m_eulerAngles);
}

void Transform::setParent(Transform * transform)
{
	m_parent = transform;
}
