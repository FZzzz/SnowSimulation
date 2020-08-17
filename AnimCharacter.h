#ifndef _ANIM_CHARACTER_H_
#define _ANIM_CHARACTER_H_

#include "GameObject.h"
#include "Mesh.h"
//#include "MotionParser/Character.h"
#include "Transform.h"
#include "Animation.h"
#include "Joint.h"
#include <vector>
#include <memory>

class Bone;
class AnimCharacter;


/*
* Bone is the implied connection between joints
*/
class Bone : public GameObject
{
public:

	Bone(std::shared_ptr<Joint> start_joint, std::shared_ptr<Joint> end_joint);
	~Bone();

	void Initialize();
	void Initialize(Transform& trans);
	void Update();
	
	// bone config
	float length;
	//glm::vec3 direction; 

private:
	std::shared_ptr<Joint> m_start_joint;
	std::shared_ptr<Joint> m_end_joint;
	   	
};

class AnimCharacter
{
public:
	AnimCharacter();
	~AnimCharacter();

	void Initialize();
	//void Initialize(MotionParser::Character& mocap_cahracter);
	void Update();
	
	void setName(std::string name);
	void setRoot(std::shared_ptr<Joint> root);
	void setJoints(const std::vector<std::shared_ptr<Joint>>& joints);
	void setBones(const std::vector<std::shared_ptr<Bone>>& bones);
	void setAnimation(std::shared_ptr<Animation> animation);
	
	inline std::string getName() { return m_name; };
	inline std::vector<std::shared_ptr<Joint>>& getJoints() { return m_joints; };
	inline std::shared_ptr<Animation> getAnimation() { return m_animation; };
	inline std::shared_ptr<Joint> getRoot() { return m_root; };


private:
	//void LoadFromASFCharacter(MotionParser::Character& mocap_character);

	// Inner setup functions
	void SetUpJointMeshes();
	void SetUpBoneMeshes();

	// Sub-function of LoadFromASFCharacter
	//void CreateBones(const std::map<std::string, MotionParser::BoneNode*>& bone_name_map);

	// Sub-function to update joint information
	void UpdateJointTransform();

	// Basic class member
	std::string m_name;

	// root transform
	Transform m_transform;
	std::vector<std::shared_ptr<Bone>> m_bones;
	std::shared_ptr<Joint> m_root;
	std::vector<std::shared_ptr<Joint>> m_joints;
	std::shared_ptr<Animation> m_animation;
};

#endif