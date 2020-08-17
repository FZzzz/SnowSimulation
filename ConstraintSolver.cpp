#include "ConstraintSolver.h"

ConstraintSolver::ConstraintSolver(PBD_MODE mode)
	: m_solver_iteration(1), m_mode(mode)
{
}

ConstraintSolver::~ConstraintSolver()
{
}

bool ConstraintSolver::SolveConstraints(
	const float &dt,
	std::vector<Constraint*>& static_constraints, 
	std::vector<Constraint*>& collision_constraints)
{
	if (m_mode == PBD_MODE::PBD)
		SolvePBDConstraints(static_constraints, collision_constraints);
	else if (m_mode == PBD_MODE::XPBD)
		SolveXPBDConstraints(dt, static_constraints, collision_constraints);

	return true;
}

void ConstraintSolver::setSolverIteration(uint32_t iteration_num)
{
	m_solver_iteration = iteration_num;
}

void ConstraintSolver::setPBDMode(PBD_MODE mode)
{
	m_mode = mode;
}

bool ConstraintSolver::SolvePBDConstraints(
	std::vector<Constraint*>& static_constraints,
	std::vector<Constraint*>& collision_constraints)
{
	for (uint32_t i = 0; i < m_solver_iteration; ++i)
	{
		for (Constraint* c : static_constraints)
		{
			if (!c->SolvePBDConstraint())
				return false;
		}
	}
	
	return true;
}

bool ConstraintSolver::SolveXPBDConstraints(
	const float &dt, 
	std::vector<Constraint*>& static_constraints, 
	std::vector<Constraint*>& collision_constraints)
{
	// Initialize Lambda
	for (Constraint* c : static_constraints)
	{
		c->InitLambda();
		c->ComputeCompliance(dt);
	}

	for (uint32_t i = 0; i < m_solver_iteration; ++i)
	{
		for (Constraint* c : static_constraints)
		{
			if (!c->SolveXPBDConstraint())
				return false;
		}
	}
	return true;
}
