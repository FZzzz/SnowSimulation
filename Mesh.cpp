#include "Mesh.h"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/rotate_vector.hpp>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include "GLFWApp.h"

Mesh::Mesh() 
	: m_useSmoothNormal(false)//, m_render_option(RENDER_ENUM::RENDER_NORMAL)
{
}

Mesh::Mesh(const Mesh &other)
{
	*this = std::move(other);
}

Mesh::~Mesh()
{
#ifdef _DEBUG
	std::cout << "Mesh Dtor\n";
#endif
	Release();
}

bool Mesh::Initialize(std::shared_ptr<Shader> shader)
{
	if (!shader)
		return false;

	if (!m_shader)
		m_shader = shader;

	if (!m_shader->getProgram())
	{
		bool shader_status = false;
		if (!m_useSmoothNormal)
			shader_status = m_shader->SetupShader(ShaderEnum::SHADER_MODE_FLAT);
		else
			shader_status = m_shader->SetupShader(ShaderEnum::SHADER_MODE_PHONG);
		
		if (!shader_status)
		{
			fprintf(stderr, "shader setup failed -- please check your file existence");
			return false;
		}
	}

	RegistToManager();

	return true;
}

void Mesh::Release()
{

	glDeleteBuffers(3, m_vbo);
	glDeleteBuffers(1, &m_ebo);
	glDeleteVertexArrays(1, &m_vao);

	m_shader.reset();
	if (m_positions.size() > 0)
		m_positions.clear();
	if (m_normals.size() > 0)
		m_normals.clear();
	if (m_indices.size() > 0)
		m_indices.clear();
	if (m_colors.size() > 0)
		m_colors.clear();
	if (m_texcoord.size() > 0)
		m_texcoord.clear();
	if (m_triangleNormalIndex.size() > 0)
		m_triangleNormalIndex.clear();


	m_positions.shrink_to_fit();
	m_normals.shrink_to_fit();
	m_colors.shrink_to_fit();
	m_texcoord.shrink_to_fit();
	m_indices.shrink_to_fit();

}

/*
bool Mesh::LoadFromFile(std::string path)
{
	if (path.find(".obj") == string::npos)
	{
		cout << "Only Support (.obj) Files" << endl;
		return false;
	}

	ifstream fin;
	fin.open(path.c_str());

	if (!fin)
	{
		cout << "Cannot Open File" << path << "!!\n";
		return false;
	}

	string data_buf;
	string symb;
	bool start = true;

	while (fin >> symb)
	{
		if (symb.compare("#") == 0)
		{
			getline(fin, data_buf);
		}
		else if (symb.compare("v") == 0)
		{
			float x, y, z;

			getline(fin, data_buf);

			istringstream sin(data_buf.c_str());
			sin >> x >> y >> z;

			glm::vec3 position(x, y, z);
			
			m_positions.push_back(position);

		}
		else if (symb.compare("vn") == 0)
		{
			float x, y, z;

			getline(fin, data_buf);

			istringstream sin(data_buf.c_str());

			sin >> x >> y >> z;

			glm::vec3 normal(x, y, z);

			m_normals.push_back(normal);
		}
		else if (symb.compare("f") == 0)
		{
			//build indices and calculate flat normals; 
			//Triangle only Ov<
			getline(fin, data_buf);
			istringstream sin(data_buf.c_str());
			unsigned int index[20];
			int indicesCount = 0;


			if (data_buf.find("//") != std::string::npos) //found "//" in data_buf
			{
				string data1, data2, data3;

				unsigned int triangleId;

				sin >> data1 >> data2 >> data3;

				data1.replace(data1.find("//"), string("//").length(), " ");
				data2.replace(data2.find("//"), string("//").length(), " ");
				data3.replace(data3.find("//"), string("//").length(), " ");

				istringstream parseIndices1(data1.c_str());
				istringstream parseIndices2(data2.c_str());
				istringstream parseIndices3(data3.c_str());

				parseIndices1 >> index[0] >> triangleId;
				parseIndices2 >> index[1];
				parseIndices3 >> index[2];

				parseIndices1.clear();
				parseIndices2.clear();
				parseIndices3.clear();

				m_triangleNormalIndex.push_back(triangleId - 1);
				for (int i = 0; i < 3; i++)
				{
					m_indices.push_back(index[i] - 1);
				}
			}
			else if (data_buf.find("/") != std::string::npos) //found "/" in data_buf
			{
				string data1, data2, data3;

				unsigned int vt[3];
				unsigned int vn[3];

				sin >> data1 >> data2 >> data3;

				data1.replace(data1.find("/"), string("/").length(), " ");
				data2.replace(data2.find("/"), string("/").length(), " ");
				data3.replace(data3.find("/"), string("/").length(), " ");

				istringstream parseIndices1(data1.c_str());
				istringstream parseIndices2(data2.c_str());
				istringstream parseIndices3(data3.c_str());

				parseIndices1 >> index[0] >> vt[0] >> vn[0];
				parseIndices2 >> index[1] >> vt[1] >> vn[1];
				parseIndices3 >> index[2] >> vt[2] >> vn[2];

				parseIndices1.clear();
				parseIndices2.clear();
				parseIndices3.clear();

				m_triangleNormalIndex.push_back(vn[0] - 1);
				for (int i = 0; i < 3; i++)
				{
					m_indices.push_back(index[i] - 1);
				}
			}
			else
			{
				unsigned int data;
				while (sin >> data)
				{
					index[indicesCount] = data;
					indicesCount++;
				}

				for (int i = 1; i < indicesCount - 1; i++)
				{
					m_indices.push_back(index[0] - 1);
					m_indices.push_back(index[i] - 1);
					m_indices.push_back(index[i + 1] - 1);
				}
			}

		}
	}

	fin.close();
	std::cout << "Load Success!! " + path << endl;

	if (m_useSmoothNormal)
	{
		ComputeSmoothNormal(1);
	}
	SetupGLBuffers();
	//UploadToGPU();

	return true;
}

bool Mesh::LoadFromFileAssimp(std::string path)
{
	if (path.find(".obj") == string::npos)
	{
		cout << "Only Support (.obj) Files" << endl;
		return false;
	}

	Assimp::Importer importer;
	const auto scene = importer.ReadFile(
		path,
		aiProcess_CalcTangentSpace |
		aiProcess_Triangulate
	);

	if (!scene)
	{
		std::cout << importer.GetErrorString() << std::endl;
		return false;
	}

	if (scene->HasMeshes())
		std::cout << scene->mNumMeshes << std::endl;

	// Obj file will only contain one mesh....
	const aiVector3D Zero3D(0.0f, 0.0f, 0.0f);
	
	const auto mesh = scene->mMeshes[0];
	for (unsigned int i = 0; i < mesh->mNumVertices; ++i)
	{
		const auto* pos = &(mesh->mVertices[i]);
		const auto* normal = &(mesh->mNormals[i]);
		const auto* texcoord = mesh->HasTextureCoords(0) ? &(mesh->mTextureCoords[0][i]) : &Zero3D;

		m_positions.push_back( glm::vec3(pos->x, pos->y, pos->z) );
		m_normals.push_back( glm::vec3(normal->x, normal->y, normal->z) );
		m_texcoord.push_back( glm::vec2(texcoord->x, texcoord->y) );
	}

	for (unsigned int i = 0; i < mesh->mNumFaces; ++i)
	{
		const aiFace& face = mesh->mFaces[i];
#ifdef _DEBUG
		assert(face.mNumIndices == 3);
#endif 
		m_indices.push_back(face.mIndices[0]);
		m_indices.push_back(face.mIndices[1]);
		m_indices.push_back(face.mIndices[2]);
	}

	SetupGLBuffers();

	std::cout << "Load Sucess (Assimp)\t" << path << std::endl;

	return true;
}
*/

void Mesh::UploadToGPU()
{
	glBindVertexArray(m_vao);
	//upload vertices position to GPU

	if (m_positions.size() > 0)
	{
		glBindBuffer(GL_ARRAY_BUFFER, m_vbo[0]);
		glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3)* m_positions.size(),
			m_positions.data(), GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
		std::cout << "Has Vertex, Size:" << m_positions.size() << "\n";
	}

	if (m_normals.size() > 0)
	{
		glBindBuffer(GL_ARRAY_BUFFER, m_vbo[1]);
		glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3)* m_normals.size(),
			m_normals.data(), GL_STATIC_DRAW);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
		std::cout << "Has Normal, size: " << m_normals.size() << "\n";
	}

	if (m_texcoord.size() > 0)
	{
		glBindBuffer(GL_ARRAY_BUFFER, m_vbo[2]);
		glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2)* m_texcoord.size(),
			m_texcoord.data(), GL_STATIC_DRAW);
		glEnableVertexAttribArray(2);
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, 0);
		std::cout << "Using Texture\n";
	}

	//setup indices info
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_ebo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint) * m_indices.size(),
		m_indices.data(), GL_STATIC_DRAW);

	glBindVertexArray(0);
}

void Mesh::UpdatePositions()
{
	glBindVertexArray(m_vao);
	//upload vertices position to GPU

	if (m_positions.size() > 0)
	{
		glBindBuffer(GL_ARRAY_BUFFER, m_vbo[0]);
		glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3)* m_positions.size(),
			m_positions.data(), GL_STREAM_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
	}

	glBindVertexArray(0);
}

void Mesh::UpdateNormals()
{
	glBindVertexArray(m_vao);
	if (m_normals.size() > 0)
	{
		glBindBuffer(GL_ARRAY_BUFFER, m_vbo[1]);
		glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3)* m_normals.size(),
			m_normals.data(), GL_STREAM_DRAW);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
	}
	glBindVertexArray(0);
}

void Mesh::UpdateTextcoords()
{
	glBindVertexArray(m_vao);
	if (m_texcoord.size() > 0)
	{
		glBindBuffer(GL_ARRAY_BUFFER, m_vbo[2]);
		glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2)* m_texcoord.size(),
			m_texcoord.data(), GL_STREAM_DRAW);
		glEnableVertexAttribArray(2);
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, 0);
	}
	glBindVertexArray(0);
}

void Mesh::UpdateIndices()
{
	glBindVertexArray(m_vao);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_ebo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint) * m_indices.size(),
		m_indices.data(), GL_STREAM_DRAW);
	glBindVertexArray(0);
}

void Mesh::SetupGLBuffers()
{
	if (!m_shader)
	{
		std::cout << "No shader No render!\n";
		return;
	}
	m_shader->Use();
	glGenVertexArrays(1, &m_vao);
	glGenBuffers(3, m_vbo);
	glGenBuffers(1, &m_ebo);
	UploadToGPU();

}

void Mesh::ComputeSmoothNormal(unsigned int normal_type)
{
	if (!m_useSmoothNormal)
		return;

	switch (normal_type)
	{
	case 1:
	{
		ComputeVNormalByFace();
		break;
	}
	case 2:
	{
		ComputeVNormalByEdge();
		break;
	}
	default:
	{
		break;
	}
	}//switch end

}

void Mesh::ResetPositions(std::vector<glm::vec3>& pos)
{
	m_positions.clear();
	m_positions = pos;
	UpdatePositions();
}

void Mesh::ResetNormals(std::vector<glm::vec3>& normals)
{
	m_normals.clear();
	m_normals = std::move(normals);
	UpdateNormals();
}

void Mesh::ResetTexCoord(std::vector<glm::vec2>& texcoords)
{
	m_texcoord.clear();
	m_texcoord = std::move(texcoords);
	UpdatePositions();
}

void Mesh::ResetIndices(std::vector<unsigned int>& indices)
{
	m_indices.clear();
	m_indices = std::move(indices);
	UpdateIndices();
}

void Mesh::Render()
{
	if (!m_shader)
		return;
	if (m_indices.size() <= 0)
	{
		std::cout << "There's no any index to draw\n";
		return;
	}

	//m_shader->Use();
}

/*
void Mesh::Update()
{
	SetupGLBuffers();
}
*/

void Mesh::setPositions(std::vector<glm::vec3> pos)
{
	m_positions = pos;
}

void Mesh::setNormals(std::vector<glm::vec3>& normals)
{
	m_normals = std::move(normals);
}

void Mesh::setIndices(std::vector<unsigned int>& indices)
{
	m_indices = std::move(indices);
}

void Mesh::setTexCoord(std::vector<glm::vec2>& texcoords)
{
	m_texcoord = std::move(texcoords);
}

void Mesh::setShader(std::shared_ptr<Shader> shader)
{
	m_shader = shader;
}

void Mesh::setName(std::string name)
{
	m_name = name;
}

/*
void Mesh::setRenderOption(RENDER_ENUM render_option)
{
	m_render_option = render_option;
}
*/

void Mesh::ComputeVNormalByFace()
{
	if (m_normals.size() == 0)
	{
		std::cout << "Require given normals\n";
		return;
	}
	struct Normal_Acc
	{
		glm::vec3 value;
		int count=0;
	};

	std::cout << "Face Avg Computation\n";

	std::vector<Normal_Acc> normal_sum;
	normal_sum.resize(m_positions.size());

	{
		size_t idx = 0;
		for (size_t i = 0; i < m_indices.size(); i += 3)
		{
			/*
			std::cout << m_indices[i + 0] << "//" << m_triangleNormalIndex[idx] << " ";
			std::cout << m_indices[i + 1] << "//" << m_triangleNormalIndex[idx] << " ";
			std::cout << m_indices[i + 2] << "//" << m_triangleNormalIndex[idx] << " \n";
			*/
			unsigned int v1, v2, v3, tri_normal_id;
			v1 = m_indices[i + 0];
			v2 = m_indices[i + 1];
			v3 = m_indices[i + 2];
			tri_normal_id = m_triangleNormalIndex[idx];

			normal_sum[v1].value += m_normals[tri_normal_id];
			normal_sum[v1].count++;

			normal_sum[v2].value += m_normals[tri_normal_id];
			normal_sum[v2].count++;

			normal_sum[v3].value += m_normals[tri_normal_id];
			normal_sum[v3].count++;

			idx++;
		}

		for (size_t i = 0; i < normal_sum.size();++i)
		{
			/*
			std::cout << normal_sum[i].value.x << " " 
					  << normal_sum[i].value.y << " " 
					  << normal_sum[i].value.z << " " 
					  << normal_sum[i].count << "\n";
			*/
			normal_sum[i].value = normal_sum[i].value / static_cast<float>(normal_sum[i].count);
		}

		m_normals.clear();
		m_normals.resize(normal_sum.size());
		for (size_t i = 0; i < normal_sum.size(); ++i)
		{
			m_normals[i] = normal_sum[i].value;
			/*
			std::cout << normal_sum[i].value.x << " "
				<< normal_sum[i].value.y << " "
				<< normal_sum[i].value.z << "\n";
			*/
		}

	}


}

void Mesh::ComputeVNormalByEdge()
{
	std::cout << "Edge Relate Normal Avg\n";
}

void Mesh::RegistToManager()
{
	GLFWApp::getInstance()->getResourceManager()->AddMesh( shared_from_this() );
}
