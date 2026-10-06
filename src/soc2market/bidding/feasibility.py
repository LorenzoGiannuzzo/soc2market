"""
Feasibility checks on a solved PuLP problem.

A reused MultiBESSMILPOptimizer returns, from its second call onwards, a solution that
violates the constraints of the problem it was asked to solve (initial SOC and SOC
dynamics of the first day are kept). These helpers make that failure mode impossible to
miss: every daily solve of this repository is checked against its own constraints.
"""

import pulp


def constraint_violations(problem, tol=1e-6):
    """Return (magnitude, name) for every constraint violated by more than tol, largest first."""
    out = []
    for name, c in problem.constraints.items():
        try:
            val = c.value()
        except TypeError:
            out.append((float("inf"), f"{name} (variable without value)"))
            continue
        if c.sense == pulp.LpConstraintEQ:
            v = abs(val)
        elif c.sense == pulp.LpConstraintLE:
            v = max(val, 0.0)
        else:
            v = max(-val, 0.0)
        if v > tol:
            out.append((v, name))
    return sorted(out, reverse=True)


def max_violation(problem):
    """Largest constraint violation of the current solution (0.0 if fully feasible)."""
    viol = constraint_violations(problem, tol=0.0)
    return viol[0][0] if viol else 0.0
