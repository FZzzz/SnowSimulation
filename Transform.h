#ifndef _TRANSFORM_H_
#define _TRANSFORM_H_

#include "common.h"

class Transform;

class Transform
{
public:
	Transform();
	Transform(const glm::vec3 &translation, const glm::vec3 &eulerAngles, const glm::vec3 &scale);
	
	~Transform();

	void Update();
	void Compute_ModelMat_World();
	void CopyTo(Transform& other);
	void setRotation(glm::quat input);
	void setRotation(glm::vec3 input);
	void setRotationDegree(glm::vec3 input);
	void setParent(Transform* transform);

	inline const glm::vec3& getEulerAngles() const { return m_eulerAngles; }
	inline const glm::quat& getQuaternion() const { return m_quaternion; }
	
	// Expose state matrix
	inline const glm::mat4& getModelMat() const { return m_modelMat; }
	inline const glm::mat4& getModelMatWorld() const { return m_modelMat_world; }
	inline const glm::mat4& getScaleMat() const { return m_scaleMat; }
	inline const glm::mat4& getRotationMat() const { return m_rotationMat; }
	inline const glm::mat4& getTranslationMat() const { return m_translationMat; }
	inline Transform* getParent() { return m_parent; }

	// Object-oriented axis
	glm::vec3 m_front;
	glm::vec3 m_up;
	glm::vec3 m_right;

	glm::vec3 m_scale;
	glm::vec3 m_translation;

protected:
	
	glm::mat4 m_modelMat;
	glm::mat4 m_modelMat_world;

	glm::quat m_quaternion;

	glm::mat4 m_scaleMat;
	glm::mat4 m_rotationMat;
	glm::mat4 m_translationMat;

	Transform* m_parent;

private:
	// DO NOT ALLOW DIRECT ACESS EULER ANGLES
	glm::vec3 m_eulerAngles;

};

#endif
