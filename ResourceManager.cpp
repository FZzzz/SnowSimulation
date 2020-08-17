#include "ResourceManager.h"
#include "GLFWApp.h"
#include "AnimCharacter.h"

ResourceManager::ResourceManager()
{
	m_obj_vec.reserve(1024);
	m_shader_vec.reserve(128);
	m_mesh_vec.reserve(256);
}


ResourceManager::~ResourceManager()
{
}

void ResourceManager::ShutDown()
{
	m_obj_vec.clear();
	m_static_obj_vec.clear();
	m_mesh_vec.clear();
	m_static_mesh_vec.clear();
	m_shader_vec.clear();

	m_obj_vec.shrink_to_fit();
	m_mesh_vec.shrink_to_fit();
	m_shader_vec.shrink_to_fit();

}

void ResourceManager::RemoveObject(std::weak_ptr<GameObject> object)
{
	auto obj_sp = object.lock();
	obj_sp->Release();
	obj_sp.reset();
}

void ResourceManager::RemoveObjectFromLast()
{
	if (m_obj_vec.size() > 0)
	{
		auto renderer = GLFWApp::getInstance()->getRenderer();
		auto object = *(m_obj_vec.cend() - 1);
		auto key_shader = object->getMesh()->getShader();

		std::cout << "Remove: " << (dynamic_pointer_cast<Joint>(object))->getName() << std::endl;
		std::cout << "Pos: " << object->m_transform.m_translation.x << ", "
							 << object->m_transform.m_translation.y << ", "
							 << object->m_transform.m_translation.z
							 << std::endl;
		/*
		glm::vec3 world_pos;
		world_pos = glm::vec3(
			object->m_transform.getParent()->getModelMatWorld() * glm::vec4(object->m_transform.m_translation, 1.0f));
		*/

		std::cout << "Pos: " << object->m_transform.getModelMatWorld()[3].x << ", "
							 << object->m_transform.getModelMatWorld()[3].y << ", "
							 << object->m_transform.getModelMatWorld()[3].z
							 << std::endl;

		RemoveObjectMapCache(key_shader, object);
		m_obj_vec.erase(m_obj_vec.end() - 1);
	}
#ifdef _DEBUG
	std::cout << "\nObject: Cap: " << m_obj_vec.capacity() << "\n";
	std::cout << "Mesh:   Cap: " << m_mesh_vec.capacity() << "\n";
	std::cout << "Shader: Cap: " << m_shader_vec.capacity() << "\n";
#endif
}

std::shared_ptr<GameObject> ResourceManager::FindObjByName(const string& name)
{
	for (size_t i = 0; i < m_obj_vec.size(); i++)
	{
		/*find by name*/
		if (m_obj_vec[i]->getName() == name)
		{
			return m_obj_vec[i];
		}
	}

	//if find nothing
	return nullptr;
}

void ResourceManager::SetMainCamera(std::shared_ptr<Camera> camera)
{
	m_camera = camera;
}

void ResourceManager::AddStaticObject(std::shared_ptr<GameObject> object)
{
#ifdef _DEBUG
	assert(object->getObjectType() == OBJECT_FLAG_ENUM::OBJECT_STATIC);
#endif
	m_static_obj_vec.push_back(std::move(object));
}

void ResourceManager::AddGameObject(std::shared_ptr<GameObject> object)
{
	if (object->hasMesh())
	{
		auto mesh_shader = object->getMesh()->getShader();
		AddtoShaderObjectMap(mesh_shader, object);
	}
	
	m_obj_vec.push_back(std::move(object));
	
}

void ResourceManager::AddMesh(std::shared_ptr<Mesh> mesh)
{
	m_mesh_vec.push_back(std::move(mesh));
}

void ResourceManager::AddShader(std::shared_ptr<Shader> shader)
{
	m_shader_vec.push_back(std::move(shader));
}

void ResourceManager::AddAnimCharacter(std::shared_ptr<AnimCharacter> anim_character)
{
	m_anim_character_vec.push_back(anim_character);
}

void ResourceManager::AddJelly(std::shared_ptr<Jelly> jelly)
{
	m_jellies.push_back(jelly);
}

void ResourceManager::GenerateParticle(glm::vec3 pos, float mass)
{
	auto particle = std::make_shared<Particle>(pos, mass);

	m_particles.push_back(particle);
}

std::shared_ptr<Mesh> ResourceManager::FindMeshByName(std::string name)
{
	for (auto it = m_mesh_vec.cbegin(); it != m_mesh_vec.cend(); ++it)
	{
		if ((*it)->getName() == name)
		{
			return (*it);
		}
	}

	return nullptr;
}

std::shared_ptr<Shader> ResourceManager::FindShaderByName(std::string name)
{
	for (auto shader : m_shader_vec)
	{
		if (shader->getName() == name)
			return shader;
	}
	return nullptr;
}

// TODO: Static objects may have different shader
inline int ResourceManager::CheckStaticMesh(const size_t& new_length)
{
	if (m_static_mesh_vec.size() == 0)
		return 1;
	auto last_mesh_it = m_static_mesh_vec.cend() - 1;
	if ((*last_mesh_it)->getNumberOfIndices() + new_length >= MAX_NUM_OF_INDICES)
	{
		// create new static mesh
		return 1;
	}
	return 0;
}

void ResourceManager::MergeStaticMesh(int opt, std::shared_ptr<GameObject> object, std::shared_ptr<Mesh> mesh)
{
	switch(opt)
	{
		case 0:
		{
			// append existing one
			auto static_mesh_it = m_mesh_vec.end() - 1;
			const auto& mesh_indices = mesh->getIndices();
			auto& static_mesh_indices = (*static_mesh_it)->getIndices();

			const auto& mesh_vert_vec = mesh->getPositions();
			// speed up by direct accessing data (no type-checking)
			auto const mesh_vert_arr = mesh->getPositions().data();
			auto& static_mesh_vert_vec = (*static_mesh_it)->getPositions();

			unsigned int offset = static_cast<unsigned int>((*static_mesh_it)->getPositions().size()) - 1;

			// append vetices
			for(size_t i = 0; i < mesh_vert_vec.size(); ++i)
			{
				const glm::vec3& vert_pos = mesh_vert_arr[i];
				glm::vec4 pos_4 = object->m_transform.getModelMat() * glm::vec4(vert_pos, 1.0);
				glm::vec3 pos = glm::vec3(pos_4);
				static_mesh_vert_vec.push_back(pos);

			}

			// append indices
			for (size_t i = 0; i < mesh->getNumberOfIndices(); ++i)
			{
				unsigned int idx = mesh_indices[i] + offset;
				static_mesh_indices.push_back(idx);
			}
			break;
		}		
		case 1:
		{
			auto static_obj = std::make_shared<GameObject>();
			auto new_mesh = std::make_shared<Mesh>();
			std::vector<glm::vec3> positions = mesh->getPositions();
			auto position_arr = positions.data();
			std::vector<unsigned int> indices = mesh->getIndices();

			// transform every pos to world space
			for (size_t i = 0; i < positions.size(); ++i)
			{
				glm::vec4 pos_4 = object->m_transform.getModelMat() * glm::vec4(position_arr[i], 1.0);
				position_arr[i] = glm::vec3(pos_4);
			}

			new_mesh->setPositions(positions);
			new_mesh->setIndices(indices);
			new_mesh->Initialize(object->getMesh()->getShader());
			m_static_mesh_vec.push_back(new_mesh);

			static_obj->setMesh(new_mesh);
			static_obj->Initialize(glm::vec3(0, 0, 0));
			m_static_obj_tmp_vec.push_back(static_obj);

			break;
		}
	
	}

}

void ResourceManager::AddtoShaderObjectMap(std::shared_ptr<Shader> shader, std::shared_ptr<GameObject> gameobject)
{
	if (m_shader_object_map.find(shader) != m_shader_object_map.end())
	{
		// Append
		m_shader_object_map[shader].push_back(gameobject);
	}
	else
	{
		// Create new vecotr
		std::vector<std::shared_ptr<GameObject>> vec;
		vec.push_back(gameobject);
		m_shader_object_map.emplace(shader, vec);
	}
}

void ResourceManager::RemoveObjectMapCache(std::shared_ptr<Shader> shader, std::shared_ptr<GameObject> object)
{
	auto map_it = m_shader_object_map.find(shader);
	auto& obj_vec = map_it->second;
	auto obj_it = std::find(obj_vec.begin(), obj_vec.end(), object);
	if (obj_it != obj_vec.end())
	{
		obj_vec.erase(obj_it);
	}
}

void ResourceManager::ArrangeStaticObjects()
{
	for (size_t i = 0; i < m_static_obj_vec.size(); ++i)
	{
		int opt = CheckStaticMesh(m_static_obj_vec[i]->getMesh()->getNumberOfIndices());
		MergeStaticMesh(opt, m_static_obj_vec[i], m_static_obj_vec[i]->getMesh());
	}

	// Update all static mehes info to GPU
	for (size_t i = 0; i < m_static_mesh_vec.size(); ++i)
	{
		m_static_mesh_vec[i]->SetupGLBuffers();
	}

	m_static_obj_vec.clear();
	m_static_obj_vec = m_static_obj_tmp_vec;

	for (auto it = m_static_obj_vec.cbegin(); it != m_static_obj_vec.cend(); ++it)
	{
		const auto object = *it;
		auto mesh_shader = object->getMesh()->getShader();
		AddtoShaderObjectMap(mesh_shader, object);
	}
	

}
