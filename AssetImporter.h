#ifndef _ASSET_IMPORTER_H_
#define _ASSET_IMPORTER_H_

#include "Mesh.h"
#include "GameObject.h"
#include "AnimCharacter.h"
#include <memory>
#include <assimp/Importer.hpp>

enum IMPORT_STATUS
{
	IMPORT_FAILED,
	IMPORT_SUCCESS
};

class AssetImporter
{
public:
	AssetImporter();
	~AssetImporter();
	
	// Cleanup Importer member data
	void CleanUpImporter();

	std::shared_ptr<Mesh> LoadMesh(std::string path, std::shared_ptr<Shader> shader, IMPORT_STATUS& status);
	std::shared_ptr<AnimCharacter> LoadBVH(std::string path, IMPORT_STATUS& status);
	std::shared_ptr<AnimCharacter> LoadBVHAssimp(std::string path, IMPORT_STATUS& status);

private:
	//void LoadBones(unsigned int index, const aiMesh* mesh, );
	bool OpenFile(std::string path);
	std::shared_ptr<Joint> CreateJoint(std::string name);
	void CreateChannel(Channel& channel, std::string type);
	std::shared_ptr<Bone> CreateBones(std::shared_ptr<Joint> start, std::shared_ptr<Joint> end);
	//bool ParseHierarchy_BVH();
	//bool ParseAnimation_BVH();
	

	std::stringstream ss_;
	IMPORT_STATUS m_status;
	std::string m_file_content;

};

#endif
