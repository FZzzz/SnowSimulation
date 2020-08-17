#ifndef _CONSTRAINT_SOLVER_H_
#define _CONSTRAINT_SOLVER_H_

#include <vector>
#include <cstdint>
#include "Constraints.h"

enum class PBD_MODE
{
	PBD,
	XPBD
};

class ConstraintSolver
{
public:
	ConstraintSolver() = delete;
	ConstraintSolver(PBD_MODE mode);
	~ConstraintSolver();
	
	bool SolveConstraints(
		const float &dt,
		std::vector<Constraint*>& static_constraints, 
		std::vector<Constraint*>& collision_constraints
	);
	
	// setter
	void setSolverIteration(uint32_t iteration_num);
	void setPBDMode(PBD_MODE mode);

	//getter
	uint32_t getSolverIteration() { return m_solver_iteration; }
	PBD_MODE getPBDMode() { return m_mode; }

private:

	bool SolvePBDConstraints(
		std::vector<Constraint*>& static_constraints,
		std::vector<Constraint*>& collision_constraints
	);

	bool SolveXPBDConstraints(
		const float &dt,
		std::vector<Constraint*>& static_constraints,
		std::vector<Constraint*>& collision_constraints
	);


	uint32_t m_solver_iteration;
	PBD_MODE m_mode;


};

#endif