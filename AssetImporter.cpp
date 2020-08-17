#include "AssetImporter.h"
#include <glm/common.hpp>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <assimp/anim.h>
#include <fstream>
#include <sstream>

#define BUF_LENGTH 4096

AssetImporter::AssetImporter() : m_status(IMPORT_FAILED)
{
}

AssetImporter::~AssetImporter()
{
}

void AssetImporter::CleanUpImporter()
{
}

std::shared_ptr<Mesh> AssetImporter::LoadMesh(
	std::string path, 
	std::shared_ptr<Shader> shader,
	IMPORT_STATUS & status)
{
	/*
		Load Mesh
	*/
	status = IMPORT_STATUS::IMPORT_FAILED;
	if (path.find(".obj") == string::npos)
	{
		cout << "Only Support (.obj) Files" << endl;
		return nullptr;
	}

	Assimp::Importer importer;
	const auto scene = importer.ReadFile(
		path,
		aiProcess_CalcTangentSpace |
		aiProcess_Triangulate |
		aiProcess_GenSmoothNormals |
		aiProcess_FlipUVs
	);

	if (!scene)
	{
		std::cout << importer.GetErrorString() << std::endl;
		return nullptr;
	}

	if (!scene->HasMeshes())
		return nullptr;

	//std::shared_ptr<RenderableObject> mesh = nullptr;
	auto mesh = std::make_shared<Mesh>();
	mesh->Initialize(shader);

	// Obj file will only contain one mesh....
	const aiVector3D Zero3D(0.0f, 0.0f, 0.0f);

	std::vector<glm::vec3> positions;
	std::vector<glm::vec3> normals;
	std::vector<glm::vec2> texcoords;
	std::vector<unsigned int> indices;

	const auto assimp_mesh = scene->mMeshes[0];
	for (unsigned int i = 0; i < assimp_mesh->mNumVertices; ++i)
	{
		const auto* pos = &(assimp_mesh->mVertices[i]);
		const auto* normal = &(assimp_mesh->mNormals[i]);
		const auto* texcoord = assimp_mesh->HasTextureCoords(0) ? &(assimp_mesh->mTextureCoords[0][i]) : &Zero3D;

		positions.push_back(glm::vec3(pos->x, pos->y, pos->z));
		normals.push_back(glm::vec3(normal->x, normal->y, normal->z));
		texcoords.push_back(glm::vec2(texcoord->x, texcoord->y));
	}
	mesh->setPositions(positions);
	mesh->setNormals(normals);
	mesh->setTexCoord(texcoords);


	for (unsigned int i = 0; i < assimp_mesh->mNumFaces; ++i)
	{
		const aiFace& face = assimp_mesh->mFaces[i];
#ifdef _DEBUG
		assert(face.mNumIndices == 3);
#endif
		indices.push_back(face.mIndices[0]);
		indices.push_back(face.mIndices[1]);
		indices.push_back(face.mIndices[2]);
	}
	mesh->setIndices(indices);

	mesh->SetupGLBuffers();

	std::cout << "Load Sucess (Assimp)\t" << path << std::endl;
	status = IMPORT_STATUS::IMPORT_SUCCESS;

	return mesh;
}

std::shared_ptr<AnimCharacter> AssetImporter::LoadBVH(std::string path, IMPORT_STATUS & status)
{
	status = IMPORT_FAILED;
	
	if (!OpenFile(path))
	{
		return nullptr;;
	}
	
	std::stringstream line_ss;
	char buf[BUF_LENGTH];

	ss_ << m_file_content;

	std::shared_ptr<AnimCharacter> anim_character = nullptr;
	std::shared_ptr<Joint> root = nullptr;
	std::shared_ptr<Joint> prev = nullptr;
	std::shared_ptr<Joint> joint = nullptr;

	std::vector<std::shared_ptr<Joint>> joint_vec;
	std::vector<std::shared_ptr<Bone>> bone_vec;

	while (ss_.getline(buf, BUF_LENGTH))
	{
		std::string token;

		line_ss.clear();
		line_ss.str(buf);
		line_ss >> token;

		if (token == "HIERARCHY")
			continue;
		else if (token == "ROOT")
		{
			line_ss >> token;
			joint = CreateJoint(token);
			joint_vec.push_back(joint);

			root = joint;
			prev = joint;

		}
		else if (token == "JOINT")
		{
			line_ss >> token;
			joint = CreateJoint(token);
			joint_vec.push_back(joint);

			joint->setParent(prev);
			prev->AddChild(joint);

			prev = joint;
		}
		else if (token == "End")
		{
			// What about this?
			std::string name = prev->getName() + "_End_site";
			joint = CreateJoint(name);
			joint_vec.push_back(joint);

			joint->setParent(prev);
			prev->AddChild(joint);
			// skip one '}'
			//ss_.getline(buf, BUF_LENGTH);
		}
		else if (token == "}")
		{
			prev = (prev) ? prev->getParent() : nullptr;
			if (prev == nullptr)
				break;
		}
		else if (token == "MOTION")
		{
			std::cout << "Parsing Motion\n";
			break;
		}
			
	}

	// Create bones
	for (size_t i = 0; i < joint_vec.size(); ++i)
	{
		auto parent = joint_vec[i]->getParent();
		auto bone = CreateBones(joint_vec[i]->getParent(), joint_vec[i]);
		if(bone != nullptr)
			bone_vec.push_back(bone);
	}

	std::cout << "Joint Vec Size: " << joint_vec.size() << std::endl;

	// print hierarchy
	for (size_t i = 0; i < joint_vec.size(); ++i)
	{
		auto parent = joint_vec[i]->getParent();
		if (parent != nullptr)
			std::cout << joint_vec[i]->getName() << "->" << parent->getName() << "\n";
		else
			std::cout << joint_vec[i]->getName() << std::endl;
	}

	// Parsing MOTION
	std::shared_ptr<Animation> animation = std::make_shared<Animation>();
	std::vector<Frame> frames;
	while (ss_.getline(buf, BUF_LENGTH))
	{
		std::string token;

		line_ss.clear();
		line_ss.str(buf);
		line_ss >> token;

		if (token == "Frames:")
		{
			uint32_t frame_count;
			line_ss >> frame_count;
			animation->m_frame_count = frame_count;
		}
		else if (token == "Frame")
		{
			line_ss >> token;
			float frame_time;
			line_ss >> frame_time;
			animation->m_frame_time = frame_time;
			break;
		}
	}

	uint32_t frame_id = 0;
	// parse motion channels
	while (ss_.getline(buf, BUF_LENGTH))
	{
		float value;
		line_ss.clear();
		line_ss.str(buf);

		Frame frame_data;
		frame_data.frame_id = frame_id;
	
		
		for (size_t i = 0; i < joint_vec.size(); ++i)
		{
			std::vector<Channel> channel_vec;
			for (size_t j = 0; j < joint_vec[i]->m_channel_order.size(); ++j)
			{
				Channel channel; 
				CreateChannel(channel, joint_vec[i]->m_channel_order[j]);

				line_ss >> value;
				channel.value = value;
				channel_vec.push_back(channel);
			}
			frame_data.joint_channel_map.emplace(joint_vec[i], channel_vec);
		}
		frames.push_back(frame_data);
		frame_id++;
	}
	
	animation->setFrames(frames);

	anim_character = std::make_shared<AnimCharacter>();
	anim_character->setRoot(root);
	anim_character->setBones(bone_vec);
	anim_character->setJoints(joint_vec);
	anim_character->setAnimation(animation);
	status = IMPORT_SUCCESS;
	return anim_character;
}

// I really need joint information :<
// Incomplete
std::shared_ptr<AnimCharacter> AssetImporter::LoadBVHAssimp(
	std::string path, 
	IMPORT_STATUS & status)
{
	std::shared_ptr<AnimCharacter> anim_character = nullptr;
	status = IMPORT_STATUS::IMPORT_FAILED;
	if (path.find(".bvh") == string::npos)
	{
		cout << "Only Support (.bvh) Files" << endl;
		return nullptr;
	}
	
	Assimp::Importer importer;
	const auto scene = importer.ReadFile(
		path,
		aiProcess_CalcTangentSpace |
		aiProcess_Triangulate
	);
	std::cout << "Has Mesh: ";
	std::cout << ((scene->HasMeshes()) ? "true" : "false") << std::endl;
	std::cout << "Size: ";
	std::cout << scene->mNumMeshes << std::endl;
	std::cout << "Vertices: ";
	std::cout << scene->mMeshes[0]->mNumVertices << std::endl;
	std::cout << scene->mMeshes[0]->mNumFaces << std::endl;

	std::cout << "Has animation: ";
	std::cout << ((scene->HasAnimations()) ? "true" : "false") << std::endl;
	std::cout << "Size: ";
	std::cout << scene->mNumAnimations << std::endl;


	auto global_transform = scene->mRootNode->mTransformation;
	auto mesh = scene->mMeshes[0];
	
	for (unsigned int i = 0; i < scene->mNumAnimations; ++i)
	{

	}
	/*
		Load BVH
	*/
		
	anim_character = std::shared_ptr<AnimCharacter>();
	
	std::cout << "Load Sucess (Assimp)\t" << path << std::endl;
	status = IMPORT_STATUS::IMPORT_SUCCESS;

	return anim_character;
}

bool AssetImporter::OpenFile(std::string path)
{
	std::ifstream file(path.c_str(), std::ios::in);

	if (!file || !file.is_open())
	{
		return false;
	}

	// Access whole file content
	std::stringstream buffer;
	buffer << file.rdbuf();
	m_file_content = buffer.str();

	file.close();
	
	return true;
}

std::shared_ptr<Joint> AssetImporter::CreateJoint(std::string name)
{
	char buf[BUF_LENGTH];

	std::stringstream line_ss;
	std::string token;

	auto joint = std::make_shared<Joint>(name);

#ifdef _DEBUG
	assert(ss_.getline(buf, BUF_LENGTH)); //jump '{'
	assert(ss_.getline(buf, BUF_LENGTH));
#else 
	ss_.getline(buf, BUF_LENGTH); //jump '{'
	ss_.getline(buf, BUF_LENGTH);
#endif

	line_ss.clear();
	line_ss.str(buf);

	if (line_ss >> token && token == "OFFSET")
	{
		float x, y, z;
		line_ss >> x >> y >> z;
		glm::vec3 offset(x, y, z);
		joint->m_offset_from_parent = offset;
		joint->m_transform.m_translation = offset;
	}
	else
	{
		return nullptr;
	}

	ss_.getline(buf, BUF_LENGTH);
	line_ss.clear();
	line_ss.str(buf);

	if (line_ss >> token && token == "CHANNELS")
	{
		int num_channels = 0;
		line_ss >> num_channels;
		joint->m_num_channels = num_channels;

		while (line_ss >> token)
		{
			joint->m_channel_order.push_back(token);
		}
	}

	return joint;
}
void AssetImporter::CreateChannel(Channel& channel,std::string type)
{
	// Xposition Yposition Zposition Zrotation Xrotation Yrotation
	if (type == "Xposition")
		channel.type = Channel::Channel_Type::X_TRANSLATION;
	else if (type == "Yposition")
		channel.type = Channel::Channel_Type::Y_TRANSLATION;
	else if (type == "Zposition")
		channel.type = Channel::Channel_Type::Z_TRANSLATION;
	else if (type == "Xrotation")
		channel.type = Channel::Channel_Type::X_ROTATION;
	else if (type == "Yrotation")
		channel.type = Channel::Channel_Type::Y_ROTATION;
	else if (type == "Zrotation")
		channel.type = Channel::Channel_Type::Z_ROTATION;
}

std::shared_ptr<Bone> AssetImporter::CreateBones(std::shared_ptr<Joint> start, std::shared_ptr<Joint> end)
{
	// Based on bone's definition, start and end should both exist
	if (start == nullptr || end == nullptr)
		return nullptr;

	std::shared_ptr<Bone> bone = std::make_shared<Bone>(start, end);
	//bone->setMesh();

	glm::vec3 offset = end->m_transform.m_translation;
	glm::vec3 scale = glm::vec3(1, glm::length(offset), 1);
	
	Transform trans;
	
	/*
	// Compute bone rotation
	glm::vec3 norm_dir = glm::normalize(offset);
	glm::vec3 u = glm::vec3(0, 1, 0);
	auto rot_quat = glm::quat(u, norm_dir);

	trans.m_scale = scale;
	//trans.setRotation(rot_quat);

	trans.m_translation = start->m_transform.m_translation;
	*/
	trans.setParent(&(start->m_transform));
	
	bone->Initialize(trans);
	
	return bone;
}
