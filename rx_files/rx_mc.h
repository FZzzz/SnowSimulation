﻿/*!
  @file rx_mc.h
	
  @brief 陰関数表面からのポリゴン生成(MC法)
	
	http://local.wasp.uwa.edu.au/~pbourke/geometry/polygonise/
 
  @author Raghavendra Chandrashekara (basesd on source code
			provided by Paul Bourke and Cory Gene Bloyd)
  @date   2010-03
*/


#ifndef _RX_MC_MESH_H_
#define _RX_MC_MESH_H_


//-----------------------------------------------------------------------------
// インクルードファイル
//-----------------------------------------------------------------------------
// C標準
#include <cstdlib>

// OpenGL
#include <GL/glew.h>

// STL
#include <map>
#include <vector>
#include <string>

#include <iostream>

//#include <glm/glm.hpp>
#include <cuda_runtime.h>
#include "rx_utility.h"
#include "rx_mesh.h"



//-----------------------------------------------------------------------------
// 名前空間
//-----------------------------------------------------------------------------
using namespace std;


//-----------------------------------------------------------------------------
// 定義
//-----------------------------------------------------------------------------
typedef unsigned int uint;
//using Vec3 = glm::vec3;


#ifndef RXREAL
	#define RXREAL float
#endif


struct RxVertexID
{
	uint newID;
	double x, y, z;
};

typedef std::map<uint, RxVertexID> ID2VertexID;

struct RxTriangle
{
	uint pointID[3];
};

typedef std::vector<RxTriangle> RxTriangleVector;

struct RxScalarField
{
	uint iNum[3];
	Vec3 fWidth;
	Vec3 fMin;
};


//-----------------------------------------------------------------------------
// rxMCMeshCPUクラス
//-----------------------------------------------------------------------------
class rxMCMeshCPU
{
public:
	// コンストラクタ
	rxMCMeshCPU();

	// デストラクタ
	~rxMCMeshCPU();
	
	//! 陰関数から三角形メッシュを生成
	bool CreateMesh(RXREAL (*func)(double, double, double, void*), void* func_ptr, Vec3 min_p, double h, int n[3], RXREAL threshold, 
					vector<Vec3> &vrts, vector<Vec3> &nrms, vector<rxFace> &face);

	//! 関数からサンプルボリュームを作成して等値面メッシュ生成
	void GenerateSurfaceV(const RxScalarField sf, RXREAL(*func)(double, double, double, void*), void* func_ptr, RXREAL threshold,
		vector<Vec3>& vrts, vector<Vec3>& nrms, vector<int>& tris);

	//! サンプルボリュームから三角形メッシュを生成
	bool CreateMeshV(RXREAL *field, Vec3 min_p, double h, int n[3], RXREAL threshold, 
					 vector<Vec3> &vrts, vector<Vec3> &nrms, vector<rxFace> &face);

	//! サンプルボリュームから等値面メッシュ生成
	void GenerateSurface(const RxScalarField sf, RXREAL *field, RXREAL threshold, 
						 vector<Vec3> &vrts, vector<Vec3> &nrms, vector<int> &tris);	

	//! 関数から等値面メッシュ生成
	void GenerateSurfaceF(const RxScalarField sf, RXREAL (*func)(double, double, double, void*), void* func_ptr, RXREAL threshold, 
						  vector<Vec3> &vrts, vector<Vec3> &nrms, vector<int> &tris);

	//! 等値面作成が成功したらtrueを返す
	bool IsSurfaceValid() const { return m_bValidSurface; }

	//! 作成下等値面メッシュの破棄
	void DeleteSurface();

	//! メッシュ化に用いたグリッドの大きさ(メッシュ作成していない場合は返値が-1)
	int GetVolumeLengths(double& fVolLengthX, double& fVolLengthY, double& fVolLengthZ);

	// 作成したメッシュの情報
	uint GetNumVertices(void) const { return m_nVertices; }
	uint GetNumTriangles(void) const { return m_nTriangles; }
	uint GetNumNormals(void) const { return m_nNormals; }

protected:
	// MARK:メンバ変数
	uint m_nVertices;	//!< 等値面メッシュの頂点数
	uint m_nNormals;	//!< 等値面メッシュの頂点法線数(作成されていれば 法線数=頂点数)
	uint m_nTriangles;	//!< 等値面メッシュの三角形ポリゴン数

	ID2VertexID m_i2pt3idVertices;			//!< 等値面を形成する頂点のリスト
	RxTriangleVector m_trivecTriangles;		//!< 三角形ポリゴンを形成する頂点のリスト

	RxScalarField m_Grid;					//!< 分割グリッド情報

	// 陰関数値(スカラー値)取得用変数(どちらかのみ用いる)
	const RXREAL* m_ptScalarField;				//!< スカラー値を保持するサンプルボリューム
	RXREAL (*m_fpScalarFunc)(double, double, double, void*);	//!< スカラー値を返す関数ポインタ
	void *m_pScalarFuncPtr;

	RXREAL m_tIsoLevel;							//!< 閾値

	bool m_bValidSurface;					//!< メッシュ生成成功の可否


	// メッシュ構築用のテーブル
	static const uint m_edgeTable[256];
	static const int m_triTable[256][16];



	// MARK:protectedメンバ関数

	//! エッジID
	uint GetEdgeID(uint nX, uint nY, uint nZ, uint nEdgeNo);

	//! 頂点ID
	uint GetVertexID(uint nX, uint nY, uint nZ);

	// エッジ上の等値点を計算
	RxVertexID CalculateIntersection(uint nX, uint nY, uint nZ, uint nEdgeNo);
	RxVertexID CalculateIntersectionF(uint nX, uint nY, uint nZ, uint nEdgeNo);

	//! グリッドエッジ両端の陰関数値から線型補間で等値点を計算
	RxVertexID Interpolate(double fX1, double fY1, double fZ1, double fX2, double fY2, double fZ2, RXREAL tVal1, RXREAL tVal2);

	//! 頂点，メッシュ幾何情報を出力形式で格納
	void RenameVerticesAndTriangles(vector<Vec3> &vrts, uint &nvrts, vector<int> &tris, uint &ntris);

	//! 頂点法線計算
	void CalculateNormals(const vector<Vec3> &vrts, uint nvrts, const vector<int> &tris, uint ntris, 
						  vector<Vec3> &nrms, uint &nnrms);

};


// Not using
//-----------------------------------------------------------------------------
// rxMCMeshGPUクラス
//-----------------------------------------------------------------------------
class rxMCMeshGPU
{
protected:
	// MC法用
	uint3 m_u3GridSize;				//!< グリッド数(nx,ny,nz)
	uint3 m_u3GridSizeMask;			//!< グリッド/インデックス変換時のマスク
	uint3 m_u3GridSizeShift;		//!< グリッド/インデックス変換時のシフト量

	float3 m_f3VoxelMin;			//!< グリッド最小位置
	float3 m_f3VoxelMax;			//!< グリッド最大位置
	float3 m_f3VoxelH;				//!< グリッド幅
	uint m_uNumVoxels;				//!< 総グリッド数
	uint m_uMaxVerts;				//!< 最大頂点数
	uint m_uNumActiveVoxels;		//!< メッシュが存在するボクセル数
	uint m_uNumVrts;				//!< 総頂点数
	uint m_uNumTris;				//!< 総メッシュ数

	// デバイスメモリ
	float *g_dfVolume;				//!< 陰関数データを格納するグリッド
	float *g_dfNoise;				//!< ノイズ値を格納するグリッド(描画時の色を決定するのに使用)
	uint *m_duVoxelVerts;			//!< グリッドに含まれるメッシュ頂点数
	uint *m_duVoxelVertsScan;		//!< グリッドに含まれるメッシュ頂点数(Scan)
	uint *m_duCompactedVoxelArray;	//!< メッシュを含むグリッド情報

	uint *m_duVoxelOccupied;		//!< ポリゴンが内部に存在するボクセルのリスト
	uint *m_duVoxelOccupiedScan;	//!< ポリゴンが内部に存在するボクセルのリスト(prefix scan)

#ifdef RX_CUMC_USE_GEOMETRY
	// 幾何情報を生成するときに必要な変数
	uint3 m_u3EdgeSize[3];			//!< エッジ数(nx,ny,nz)
	uint m_uNumEdges[4];			//!< 総エッジ数
#endif

#ifdef RX_CUMC_USE_GEOMETRY
	// 幾何情報を生成するときに必要な変数
	uint *m_duVoxelCubeIdx;			//!< グリッド8頂点の陰関数値が閾値以上かどうかを各ビットに格納した変数

	uint *m_duEdgeOccupied;			//!< エッジにメッシュ頂点を含むかどうか(x方向，y方向, z方向の順)
	uint *m_duEdgeOccupiedScan;		//!< エッジにメッシュ頂点を含むかどうか(Scan)
	float *m_dfEdgeVrts;			//!< エッジごとに算出した頂点情報
	float *m_dfCompactedEdgeVrts;	//!< 隙間をつめた頂点情報
	uint *m_duIndexArray;			//!< ポリゴンの幾何情報
	float *m_dfVertexNormals;		//!< 頂点法線
	uint *m_duVoxelTriNum;			//!< グリッドごとの三角形メッシュ数
	uint *m_duVoxelTriNumScan;		//!< グリッドごとの三角形メッシュ数(Scan)
#else
	// 幾何情報を必要としないときのみ用いる
	float4 *m_df4Vrts;				//!< ポリゴン頂点座標
	float4 *m_df4Nrms;				//!< ポリゴン頂点法線
#endif

	// ホストメモリ
	float4 *m_f4VertPos;			//!< 頂点座標
#ifdef RX_CUMC_USE_GEOMETRY
	uint3 *m_u3TriIdx;				//!< メッシュインデックス
	uint *m_uScan;					//!< デバッグ用
#else
	float4 *m_f4VertNrm;			//!< 頂点法線
#endif

	int m_iVertexStore;
	bool m_bSet;


	// 陰関数値(スカラー値)取得用変数(どちらかのみ用いる)
	const float* m_ptScalarField;				//!< スカラー値を保持するサンプルボリューム
	float (*m_fpScalarFunc)(double, double, double);	//!< スカラー値を返す関数ポインタ

	float m_tIsoLevel;							//!< 閾値

	bool m_bValidSurface;					//!< メッシュ生成成功の可否

public:
	// コンストラクタ
	rxMCMeshGPU();

	// デストラクタ
	~rxMCMeshGPU();
	
	//! 陰関数から三角形メッシュを生成
	bool CreateMesh(float (*func)(double, double, double), Vec3 min_p, double h, int n[3], float threshold, string method, 
					vector<Vec3> &vrts, vector<Vec3> &nrms, vector<rxFace> &face);

	//! サンプルボリュームから三角形メッシュを生成
	bool CreateMeshV(Vec3 minp, double h, int n[3], float threshold, uint &nvrts, uint &ntris);
	
	//! 配列の確保
	bool Set(Vec3 vMin, Vec3 vH, int n[3], unsigned int vm);

	//! 配列の削除
	void Clean(void);

	//! FBOにデータを設定
	bool SetDataToFBO(GLuint uVrtVBO, GLuint uNrmVBO, GLuint uTriVBO);

	//! ホスト側配列にデータを設定
	bool SetDataToArray(vector<Vec3> &vrts, vector<Vec3> &nrms, vector<rxFace> &face);

	//! サンプルボリュームセット
	void   SetSampleVolumeFromHost(float *hVolume);
	float* GetSampleVolumeDevice(void);

	void   SetSampleNoiseFromHost(float *hVolume);
	float* GetSampleNoiseDevice(void);

	// 最大頂点数
	uint GetMaxVrts(void){ return m_uMaxVerts; }

	// 最大頂点数計算用係数
	void SetVertexStore(int vs){ m_iVertexStore = vs; }

	
#ifdef RX_CUMC_USE_GEOMETRY
	//! 頂点データ(デバイスメモリ)
	float* GetVrtDev(void){ return (float*)m_dfCompactedEdgeVrts; }

	//! メッシュデータ(デバイスメモリ)
	uint* GetIdxDev(void){ return (uint*)m_duIndexArray; }

	//! 法線データ(デバイスメモリ)
	float* GetNrmDev(void){ return (float*)m_dfVertexNormals; }

	//! 頂点データ(ホストメモリ)
	float GetVertex(int idx, int dim)
	{
		if(dim == 0){
			return m_f4VertPos[idx].x;
		}
		else if(dim == 1){
			return m_f4VertPos[idx].y;
		}
		else{
			return m_f4VertPos[idx].z;
		}
	}
	void GetVertex2(int idx, float *x, float *y, float *z)
	{
		*x = m_f4VertPos[idx].x;
		*y = m_f4VertPos[idx].y;
		*z = m_f4VertPos[idx].z;
	}

	//! メッシュデータ(ホストメモリ)
	void GetTriIdx(int idx, unsigned int *vidx0, unsigned int *vidx1, unsigned int *vidx2)
	{
		*vidx0 = m_u3TriIdx[idx].x;
		*vidx1 = m_u3TriIdx[idx].y;
		*vidx2 = m_u3TriIdx[idx].z;
	}
#else // #ifdef RX_CUMC_USE_GEOMETRY

	//! 頂点データ(デバイスメモリ)
	float* GetVrtDev(void)
	{
		return (float*)m_df4Vrts;
	}

	//! 法線データ(デバイスメモリ)
	float* GetNrmDev(void)
	{
		return (float*)m_df4Nrms;
	}

#endif // #ifdef RX_CUMC_USE_GEOMETRY
};


/*!
 * メッシュ生成(サンプルボリューム作成)
 * @param[in] sf 分割グリッド情報
 * @param[in] func 陰関数値取得関数ポインタ
 * @param[in] threshold 閾値
 * @param[out] vrts メッシュ頂点
 * @param[out] nrms メッシュ頂点法線
 * @param[out] tris メッシュ幾何情報(頂点接続情報)
 */
static void GenerateValueArray(RXREAL **field, RXREAL (*func)(void*, double, double, double), void* func_ptr, const RxScalarField sf)
{
	int nx, ny, nz;
	nx = sf.iNum[0]+1;
	ny = sf.iNum[1]+1;
	nz = sf.iNum[2]+1;

	Vec3 minp = sf.fMin;
	Vec3 d = sf.fWidth;

	if(*field) delete [] *field;
	*field = new RXREAL[nx*ny*nz];
	for(int k = 0; k < nz; ++k){
		for(int j = 0; j < ny; ++j){
			for(int i = 0; i < nx; ++i){
				int idx = k*nx*ny+j*nx+i;
				Vec3 pos = minp+Vec3(i, j, k)*d;

				RXREAL val = func(func_ptr, pos[0], pos[1], pos[2]);
				(*field)[idx] = val;
			}
		}
	}
}



#endif // _RX_MC_MESH_H_

