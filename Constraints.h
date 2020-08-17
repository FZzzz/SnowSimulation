#ifndef _CONSTRAINTS_H_
#define _CONSTRAINTS_H_

#include <vector>
#include <functional>
#include "Particle.h"

enum class CONSTRAINT_TYPE
{
	CONSTRAINT_DISTANCE,
	CONSTRAINT_BEND
};

class Constraint
{
public:

	Constraint() = delete;
	Constraint(size_t numOfRigidbodies);
	~Constraint();

	void InitLambda() { m_lambda = 0.0f; };
	void ComputeCompliance(const float &dt);

	virtual bool SolvePBDConstraint() = 0;
	virtual bool SolveXPBDConstraint() = 0;
	
	virtual float ConstraintFunction() = 0;
	virtual std::vector<std::vector<float>> GradientFunction() = 0;
	
	// setters
	void setStiffness(float stiffness);
	void setCompliance(float compliance);

	// getter
	virtual CONSTRAINT_TYPE getConstraintType() = 0;

	/* 
	 * Constraint caches particles but not owning it
	 * TODO: Rewrite this (Not generic solution) 
	 */
	std::vector<Particle_Ptr> m_particles;
	float m_lambda;

	// stiffness is the value between 0-1
	float m_stiffness;

	// XPBD 
	float m_compliance;
	// compliance_tmp = compliance / (dt * dt)
	float m_compliance_tmp;

};

class DistanceConstraint final: public Constraint
{
public:
	
	// Not allow to use C++ default constructor
	DistanceConstraint() = delete;
	/*
	 * @param p1 First particle
	 * @param p2 Second paritcle
	 * @param d  Rest length of two particle
	 */
	DistanceConstraint(Particle_Ptr p0, Particle_Ptr p1, float rest_length);
	~DistanceConstraint();

	virtual bool SolvePBDConstraint();
	virtual bool SolveXPBDConstraint();

	virtual float ConstraintFunction();
	virtual std::vector<std::vector<float>> GradientFunction();


	// getters
	CONSTRAINT_TYPE getConstraintType() { return CONSTRAINT_TYPE::CONSTRAINT_DISTANCE; };

private: 

	float m_rest_length;

};


class BendConstraint final: public Constraint
{
public:
	
	// Not allow to use C++ default constructor
	BendConstraint() = delete;
	/*
	 * @param p1 First particle
	 * @param p2 Second paritcle
	 * @param d  Initial distance of two particle
	 */
	BendConstraint(Particle* p1, Particle* p2, float d);
	~BendConstraint();

	virtual bool SolvePBDConstraint();
	virtual bool SolveXPBDConstraint();

	virtual float ConstraintFunction();
	virtual std::vector<std::vector<float>> GradientFunction();

	CONSTRAINT_TYPE getConstraintType() { return CONSTRAINT_TYPE::CONSTRAINT_BEND; };

};

#endif
