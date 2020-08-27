#ifndef _SIMULATION_H_
#define _SIMULATION_H_

#include <stdint.h>
#include <memory>
#include <vector>
#include "Constraints.h"
#include "ConstraintSolver.h"
#include "Collider.h"
#include "Rigidbody.h"
#include "ParticleSystem.h"
#include "NeighborSearch.h"

struct SimWorldDesc
{
	SimWorldDesc(float g, float damping) : gravity(g), global_velocity_damping(damping) {};
	
	float gravity;
	float global_velocity_damping;
};

struct SimParams
{
	float3 gravity;
	float  global_damping;
	
	// pbf sph coeffcient
	float  epsilon;
	float  effective_radius;
	float  particle_radius;
	float  rest_density;

	// dem coefficient
	float  boundary_damping;
	float  static_friction;
	float  kinematic_friction;

	float  sor_coeff;

	float3 world_origin;
	float3 cell_size;

	uint3  grid_size;
	uint   num_cells;
};

class Simulation
{

public:
	Simulation();
	Simulation(SimWorldDesc desc);
	~Simulation();

	void Initialize(PBD_MODE mode, std::shared_ptr<ParticleSystem> particle_system);
	bool Step(float dt);

	bool StepCUDA(float dt);

	void AddCollider(Collider* collider);
	void AddStaticConstraint(Constraint* constraint);
	void AddStaticConstraints(std::vector<Constraint*> constraints);
	void SetSolverIteration(uint32_t iter_count);


	void Pause();

	// setters
	void setGravity(float gravity);
	void setClipLength(int length);

	// getters
	inline float getGravity() { return m_world_desc.gravity; };
	inline SimParams* getSimParams() { return m_sim_params; };
	inline bool  isPause() { return m_pause; };
private:

	void SetupSimParams();

	void InitializeBoundaryParticles();
	void InitializeBoundaryCudaData();
	void GenerateParticleCube(glm::vec3 half_extends, glm::vec3 origin, int opt, bool use_jitter);

	void PredictPositions(float dt);
	void FindNeighborParticles(float effective_radius);
	
	void ComputeRestDensity();
	void ComputeDensity(float effective_radius);
	void ComputeLambdas(float effective_radius);
	void ComputeSPHParticlesCorrection(float effective_radius, float dt);
	void UpdatePredictPosition();
	
	void CollisionDetection(float dt);
	void HandleCollisionResponse();
	void GenerateCollisionConstraint();
	
	bool ProjectConstraints(const float &dt);
	
	void AddCollisionConstraint(Constraint* constraint);
	void ApplySolverResults(float dt);

	SimWorldDesc m_world_desc;
	bool m_initialized;
	bool m_first_frame;
	bool m_pause;

	/*fluid data*/
	SimParams* m_sim_params;
	SimParams* m_d_sim_params;
	float m_rest_density;
	float m_volume;
	float m_particle_mass;
	
	std::shared_ptr<ConstraintSolver> m_solver;
	std::shared_ptr<ParticleSystem> m_particle_system;
	std::shared_ptr<NeighborSearch> m_neighbor_searcher;

	std::vector<Collider*> m_colliders;
	std::vector<Rigidbody*> m_rigidbodies;
	std::vector<Constraint*> m_static_constraints;
	std::vector<Constraint*> m_collision_constraints;

	/* The collision table record who contact with whom */
	std::vector<std::vector<Collider*>> m_collision_table;
	
	/*Simulation control parameters*/
	uint32_t m_iterations;
	int m_clip_length; // clip length of how many frames at the each run
	

};

#endif
