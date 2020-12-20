/*! 
  @file rx_mesh.h

  @brief メッシュ構造の定義
 
  @author Makoto Fujisawa
  @date 2010
*/
// FILE --rx_mesh.h--

#ifndef _RX_MESH_H_
#define _RX_MESH_H_


//-----------------------------------------------------------------------------
// インクルードファイル
//-----------------------------------------------------------------------------
#include <sstream>

#include <string>
#include <vector>
#include <map>
#include <set>

#include <GL/glew.h>



using namespace std;

//using Vec2 = glm::vec2;
//using Vec3 = glm::vec3;
//using Vec4 = glm::vec4;

//-----------------------------------------------------------------------------
// 材質クラス
//-----------------------------------------------------------------------------
class rxMaterialOBJ
{
public:
	string name;		//!< 材質の名前

	Vec4 diffuse;		//!< 拡散色(GL_DIFFUSE)
	Vec4 specular;		//!< 鏡面反射色(GL_SPECULAR)
	Vec4 ambient;		//!< 環境光色(GL_AMBIENT)

	Vec4 color;			//!< 色(glColor)
	Vec4 emission;		//!< 放射色(GL_EMISSION)

	double shininess;	//!< 鏡面反射指数(GL_SHININESS)

	int illum;
	string tex_file;	//!< テクスチャファイル名
	unsigned int tex_name;	//!< テクスチャオブジェクト
};


typedef map<string, rxMaterialOBJ> rxMTL;


//-----------------------------------------------------------------------------
// メッシュクラス
//-----------------------------------------------------------------------------
// ポリゴン
class rxFace
{
public:
	vector<int> vert_idx;	//!< 頂点インデックス
	string material_name;	//!< 材質名
	vector<Vec2> texcoords;	//!< テクスチャ座標
	int attribute;			//!< 属性

public:
	rxFace() : attribute(0) {}

public:
	// オペレータによる頂点アクセス
	inline int& operator[](int i){ return vert_idx[i]; }
	inline int  operator[](int i) const { return vert_idx[i]; }

	// 関数による頂点アクセス
	inline int& at(int i){ return vert_idx.at(i); }
	inline int  at(int i) const { return vert_idx.at(i); }

	//! ポリゴン頂点数の変更
	void resize(int size)
	{
		vert_idx.resize(size);
	}

	//! ポリゴン頂点数の参照
	int size(void) const
	{
		return (int)vert_idx.size();
	}

	//! 初期化
	void clear(void)
	{
		vert_idx.clear();
		material_name = "";
		texcoords.clear();
	}
};


inline int CalMeshDiv(Vec3& minp, const Vec3 maxp, const int nmax, float& h, int n[3], float extend)
{
	Vec3 l = maxp - minp;
	minp -= 0.5f * extend * l;
	l *= 1.0 + extend;

	float max_l = 0;
	int max_axis = 0;
	for (int i = 0; i < 3; ++i) {
		if (l[i] > max_l) {
			max_l = l[i];
			max_axis = i;
		}
	}

	h = max_l / nmax;
	for (int i = 0; i < 3; ++i) {
		n[i] = (int)(l[i] / h) + 1;
	}

	return 0;
}



#endif // #ifndef _RX_MESH_H_
