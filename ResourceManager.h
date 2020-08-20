#ifndef _RESOURCEMANAGER_H_
#define _RESOURCEMANAGER_H_

#include <memory>
#include "GameObject.h"
#include "Mesh.h"
#include "Camera.h"
#include "AnimCharacter.h"
#include "Particle.h"
/*GameObject_sp
	Resource Manager Caches all resources/ object in this program.
	Every new allocated object will be regist in ResourceManager.

	In game loop, when you need to call for another object in your custom class, use findObj().
*/

//using GameObject_sp = std::shared_ptr<GameObject>;
//using Mesh_sp = std::shared_ptr<Mesh>;
//using Shader_sp = std::shared_ptr<Shader>;

using Shader_Object_Map = std::unordered_map< std::shared_ptr<Shader>, std::vector<std::shared_ptr<GameObject>>>;

class ResourceManager
{
public:
	ResourceManager();
	~ResourceManager();

	void ShutDown();
	void RemoveObject(std::weak_ptr<GameObject> object);
	void RemoveObjectFromLast();
	void ArrangeStaticObjects();
	
	std::shared_ptr<GameObject> FindObjByName(const string& name);

	void SetMainCamera(std::shared_ptr<Camera> camera);
	void AddStaticObject(std::shared_ptr<GameObject> object);
	void AddGameObject(std::shared_ptr<GameObject> object);
	void AddMesh(std::shared_ptr<Mesh> mesh);
	void AddShader(std::shared_ptr<Shader> shader);
	void AddAnimCharacter(std::shared_ptr<AnimCharacter> anim_character);
	void GenerateParticle(glm::vec3 pos, float mass);

	std::shared_ptr<Mesh> FindMeshByName(std::string name);
	std::shared_ptr<Shader> FindShaderByName(std::string name);


	// getter
	inline std::shared_ptr<Camera> getMainCamera() { return m_camera; };
	inline const std::vector<std::shared_ptr<GameObject>>& getObjects() { return m_obj_vec; };
	inline const std::vector<std::shared_ptr<GameObject>>& getStaticObjects() { return m_static_obj_vec; };
	inline const std::vector<std::shared_ptr<Mesh>>& getMeshes() { return m_mesh_vec; };
	inline const std::vector<std::shared_ptr<Shader>>& getShaders() { return m_shader_vec; };
	inline const std::vector<std::shared_ptr<AnimCharacter>>& getAnimCharacters() { return m_anim_character_vec; };
	inline const std::vector<Particle_Ptr>& getParticles() { return m_particles; };
	inline Shader_Object_Map& getShaderObjectMap() { return m_shader_object_map; };
	//inline const std::shared_ptr<SimCharacter> getSimCharacter() { return m_sim_character; };


private:

	//void CreateNewStaticMesh();
	//void MergeToStaticMesh();
	// Ensure vetex number won't overflow
	inline int CheckStaticMesh(const size_t& new_length);
	void MergeStaticMesh(int opt, std::shared_ptr<GameObject> object, std::shared_ptr<Mesh> mesh);
	void AddtoShaderObjectMap(std::shared_ptr<Shader> shader, std::shared_ptr<GameObject> gameobject);
	void RemoveObjectMapCache(std::shared_ptr<Shader> shader, std::shared_ptr<GameObject> object);

	std::vector<std::shared_ptr<GameObject>> m_obj_vec;					// Game object vector
	std::vector<std::shared_ptr<GameObject>> m_static_obj_vec;			// combine all vertex and indices (ready for batching)
	std::vector<std::shared_ptr<GameObject>> m_static_obj_tmp_vec;		// Dirty work of creating static Gameobject (only use once)

	Shader_Object_Map m_shader_object_map;

	std::vector<std::shared_ptr<AnimCharacter>> m_anim_character_vec;	// update(anim_character), render(anim_character)

	std::vector<std::shared_ptr<Mesh>> m_mesh_vec;			// Mesh vector
	std::vector<std::shared_ptr<Mesh>> m_static_mesh_vec;	// combined static meshes

	//std::shared_ptr<SimCharacter> m_sim_character;						// Simulated Character
	std::vector<std::shared_ptr<Shader>> m_shader_vec;					// Shader vector

	std::vector<Particle_Ptr> m_particles;


	std::shared_ptr<Camera> m_camera;

};

#endif
