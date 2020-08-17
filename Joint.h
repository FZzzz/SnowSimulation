#ifndef _JOINT_H_
#define _JOINT_H_

#include "GameObject.h"
#include <vector>
#include <memory>

/*
* Joint has channels decribing its degree of freedom and an offset from its parent
* PS. Root is also a joint
*/

class Joint;

/*
enum JOINT_DOF_ENUM
{
	DOF_NONE,
	DOF_RX,
	DOF_RY,
	DOF_RZ,
	DOF_RX_RY,
	DOF_RY_RZ,
	DOF_RX_RZ,
	DOF_RX_RY_RZ,
	DOF_6DOF
};
*/

struct JointLimit
{
	float min = 0, max = 0;
};

class Joint : public GameObject
{
public:
	Joint(std::string name);
	~Joint();

	void Initialize();
	void Initialize(Transform& trans);

	void AddChild(std::shared_ptr<Joint> child);

	void setParent(std::shared_ptr<Joint> parent);
	inline std::shared_ptr<Joint> getParent() { return m_parent; };
	inline std::vector<std::shared_ptr<Joint>>& getChilds() { return m_childs; };

	//Joint config
	glm::vec3 m_offset_from_parent;
	JointLimit m_joint_limit[6];
	std::vector<std::string> m_channel_order;
	uint16_t m_num_channels;
	
	//uint16_t id;

private:
	//JOINT_DOF_ENUM dof_type;
	std::shared_ptr<Joint> m_parent;
	std::vector<std::shared_ptr<Joint>> m_childs;

};


#endif
