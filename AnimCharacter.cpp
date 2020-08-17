#include "AnimCharacter.h"
#include "GLFWApp.h"
#include "common.h"

#define M_PI        3.1415926f

static inline glm::quat RotateBetweenVectors(glm::vec3 u, glm::vec3 v)
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
				s *0.5f,
				rotation_axis.x * invs,
				rotation_axis.y * invs,
				rotation_axis.z * invs
			);

}

// Bone
Bone::Bone(std::shared_ptr<Joint> start_joint, std::shared_ptr<Joint> end_joint) : 
	m_start_joint(start_joint), m_end_joint(end_joint)
{
	setName("Bone");
}

Bone::~Bone()
{
}

void Bone::Initialize()
{
	GameObject::Initialize();
}

void Bone::Initialize(Transform & trans)
{
	m_transform = trans;
	GameObject::Initialize();
}

void Bone::Update()
{
	
	glm::vec3 offset = m_end_joint->m_transform.m_translation;
	glm::vec3 scale = glm::vec3(glm::length(offset), 1, 1);

	// Compute bone rotation
	glm::vec3 norm_dir = glm::normalize(offset);
	glm::vec3 u = glm::vec3(1, 0, 0);
	auto rot_quat = RotateBetweenVectors(u, norm_dir);

	m_transform.m_scale = scale;
	m_transform.setRotation(rot_quat);
	//m_transform.setRotation(m_start_joint->m_transform.getQuaternion());
	//m_transform.m_translation = m_start_joint->m_transform.m_translation;
	

	GameObject::Update();
}

// AnimCharacter
AnimCharacter::AnimCharacter() : m_name("AnimCharacter"), m_root(nullptr), m_animation(nullptr)
{
}


AnimCharacter::~AnimCharacter()
{
}

void AnimCharacter::Initialize()
{
	SetUpJointMeshes();
	SetUpBoneMeshes();
	UpdateJointTransform();
}

void AnimCharacter::Update()
{
	// synchronize joint and keyframe
	UpdateJointTransform();
	m_animation->Step();
}

void AnimCharacter::setName(std::string name)
{
	m_name = name;
}

void AnimCharacter::setRoot(std::shared_ptr<Joint> root)
{
	m_root = root;
}

void AnimCharacter::setJoints(const std::vector<std::shared_ptr<Joint>>& joints)
{
	m_joints = joints;
}

void AnimCharacter::setBones(const std::vector<std::shared_ptr<Bone>>& bones)
{
	m_bones = bones;
}

void AnimCharacter::setAnimation(std::shared_ptr<Animation> animation)
{
	m_animation = animation;
}

void AnimCharacter::SetUpJointMeshes()
{
	auto resource_manager = GLFWApp::getInstance()->getResourceManager();
	const auto mesh_list = resource_manager->getMeshes();

	// TODO: remember to replace this with findMesh(name)
	auto mesh = mesh_list[2];

	for (auto it = m_joints.begin(); it != m_joints.end(); ++it)
	{
		(*it)->setMesh(mesh);
		resource_manager->AddGameObject(std::static_pointer_cast<GameObject>(*it));
	}

}

void AnimCharacter::SetUpBoneMeshes()
{
	auto resource_manager = GLFWApp::getInstance()->getResourceManager();
	const auto mesh_list = resource_manager->getMeshes();

	// TODO: remember to replace this with findMesh(name)
	auto mesh = mesh_list[3];
	if (mesh == nullptr)
		return;
	for (size_t i = 0; i < m_bones.size(); ++i)
	{
		m_bones[i]->setMesh(mesh);
		resource_manager->AddGameObject(std::static_pointer_cast<GameObject>(m_bones[i]));
	}
}

void AnimCharacter::UpdateJointTransform()
{
	const auto& current_frame = m_animation->getCurrentFrame();

	for (auto it = m_joints.begin(); it != m_joints.end(); ++it)
	{
		auto joint = (*it);
		const auto channel_value = current_frame.joint_channel_map.at(*it);
		const auto frame_data = current_frame.joint_framedata_map.at(*it);


		if (frame_data.movable)
			joint->m_transform.m_translation = frame_data.translation;

		//glm::quat rot_quat = glm::quat_cast(rot_mat);
		joint->m_transform.setRotation(frame_data.quaternion);

	}
}
