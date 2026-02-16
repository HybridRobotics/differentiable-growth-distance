// Copyright 2025 Akshay Thirugnanam
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * @author Akshay Thirugnanam (akshay_t@berkeley.edu)
 * @brief Differentiable growth distance algorithm for two and three-dimensional
 * convex sets.
 */

#ifndef DGD_DGD_H_
#define DGD_DGD_H_

#include "dgd/data_types.h"
#include "dgd/geometry/convex_set.h"
#include "dgd/geometry/halfspace.h"
#include "dgd/output.h"
#include "dgd/settings.h"

namespace dgd {

/*
 * Growth distance algorithm.
 */

/**
 * @brief Growth distance algorithm for 2D and 3D compact convex sets.
 *
 * @attention When using warm-start, the following properties must be ensured:
 * The same output must be reused from the previous function call;
 * The output must not be used for other pairs of sets in between function
 * calls;
 * The order of the sets must be the same.
 *
 * @param[in]     set1,set2  Compact convex sets.
 * @param[in]     tf1,tf2    Rigid body transformations for the sets.
 * @param[in]     settings   Settings.
 * @param[in,out] out        Output.
 * @param         warm_start Whether to use previous output for warm start.
 * @return        Growth distance lower bound.
 */
template <int dim>
Real GrowthDistance(const ConvexSet<dim>* set1, const Transformr<dim>& tf1,
                    const ConvexSet<dim>* set2, const Transformr<dim>& tf2,
                    const Settings& settings, Output<dim>& out,
                    bool warm_start = false);

/**
 * @brief Growth distance algorithm for a compact convex set and a half-space.
 *
 * @attention out.s1, out.s2, out.bc, and out.z2 are not set.
 *
 * @note CoincidentCenters status is returned if the center of the convex set
 * lies in the half-space.
 *
 * @note Warm start only accelerates the support function computation.
 *
 * @param[in]     set1       Compact convex set.
 * @param[in]     set2       Half-space.
 * @param[in]     tf1,tf2    Rigid body transformations for the sets.
 * @param[in]     settings   Settings.
 * @param[in,out] out        Output.
 * @param         warm_start Whether to use previous output for warm start.
 * @return        Growth distance.
 */
template <int dim>
Real GrowthDistance(const ConvexSet<dim>* set1, const Transformr<dim>& tf1,
                    const Halfspace<dim>* set2, const Transformr<dim>& tf2,
                    const Settings& settings, Output<dim>& out,
                    bool warm_start = false);

/**
 * @brief Growth distance algorithm for compact convex sets using the cutting
 * plane method.
 *
 * @see GrowthDistance
 */
template <int dim>
Real GrowthDistanceCp(const ConvexSet<dim>* set1, const Transformr<dim>& tf1,
                      const ConvexSet<dim>* set2, const Transformr<dim>& tf2,
                      const Settings& settings, Output<dim>& out,
                      bool warm_start = false);

/**
 * @brief Growth distance algorithm for compact convex sets using the trust
 * region Newton method.
 *
 * @attention Warm start is disabled for this solver.
 *
 * @see GrowthDistance
 */
template <int dim>
Real GrowthDistanceTrn(const ConvexSet<dim>* set1, const Transformr<dim>& tf1,
                       const ConvexSet<dim>* set2, const Transformr<dim>& tf2,
                       const Settings& settings, Output<dim>& out,
                       bool warm_start = false);

/*
 * Collision detection algorithm.
 */

/**
 * @brief Collision detection algorithm for 2D and 3D compact convex sets.
 *
 * Returns true if the centers coincide or if the sets intersect;
 * false if a separating plane has been found or if the maximum number of
 * iterations have been reached.
 *
 * @attention When using warm-start, the following properties must be ensured:
 * The same output must be reused from the previous function call;
 * The output must not be used for other pairs of sets in between function
 * calls;
 * The order of the sets must be the same.
 *
 * @param[in]     set1,set2  Compact convex sets.
 * @param[in]     tf1,tf2    Rigid body transformations for the sets.
 * @param[in]     settings   Settings.
 * @param[in,out] out        Output.
 * @param         warm_start Whether to use previous output for warm start.
 * @return        true, if the sets are colliding; false, otherwise.
 */
template <int dim>
bool DetectCollision(const ConvexSet<dim>* set1, const Transformr<dim>& tf1,
                     const ConvexSet<dim>* set2, const Transformr<dim>& tf2,
                     const Settings& settings, Output<dim>& out,
                     bool warm_start = false);

/**
 * @brief Collision detection algorithm for a compact convex set and a
 * half-space.
 *
 * @attention out.s1, out.s2, out.bc, and out.z2 are not set.
 *
 * @note CoincidentCenters status is returned if the center of the convex set
 * lies in the half-space.
 *
 * @note Warm start only accelerates the support function computation.
 *
 * @param[in]     set1       Compact convex set.
 * @param[in]     set2       Half-space.
 * @param[in]     tf1,tf2    Rigid body transformations for the sets.
 * @param[in]     settings   Settings.
 * @param[in,out] out        Output.
 * @param         warm_start Whether to use previous output for warm start.
 * @return        true, if the sets are colliding; false, otherwise.
 */
template <int dim>
bool DetectCollision(const ConvexSet<dim>* set1, const Transformr<dim>& tf1,
                     const Halfspace<dim>* set2, const Transformr<dim>& tf2,
                     const Settings& settings, Output<dim>& out,
                     bool warm_start = false);

/*
 * KKT solution set null space algorithm.
 */

/**
 * @brief KKT solution set null space algorithm for 2D and 3D compact convex
 * sets.
 *
 * The KKT solution set null space determines the uniqueness of the primal and
 * dual optimal solutions (not including positive scaling of the optimal normal
 * vector).
 *
 * @param[in]     set1,set2  Compact convex sets.
 * @param[in]     tf1,tf2    Rigid body transformations for the sets.
 * @param[in]     settings   Settings.
 * @param[in,out] bundle     Output bundle.
 * @return        KKT solution set nullity; 0 if the solution is not optimal.
 */
template <int dim>
int ComputeKktNullspace(const ConvexSet<dim>* set1, const Transformr<dim>& tf1,
                        const ConvexSet<dim>* set2, const Transformr<dim>& tf2,
                        const Settings& settings,
                        const OutputBundle<dim>& bundle);

/**
 * @brief KKT solution set null space algorithm for a compact convex set and a
 * half-space.
 *
 * The KKT solution set null space determines the uniqueness of the primal and
 * dual optimal solutions (not including positive scaling of the optimal normal
 * vector).
 *
 * @param[in]     set1       Compact convex set.
 * @param[in]     set2       Half-space.
 * @param[in]     tf1,tf2    Rigid body transformations for the sets.
 * @param[in]     settings   Settings.
 * @param[in,out] bundle     Output bundle.
 * @return        KKT solution set nullity; 0 if the solution is not optimal.
 */
template <int dim>
int ComputeKktNullspace(const ConvexSet<dim>* set1, const Transformr<dim>& tf1,
                        const Halfspace<dim>* set2, const Transformr<dim>& tf2,
                        const Settings& settings,
                        const OutputBundle<dim>& bundle);

/*
 * Growth distance derivative algorithm.
 */

/**
 * @name Growth distance derivative functions
 * @brief Derivative of the growth distance function for 2D and 3D convex sets
 * (including half-spaces) with respect to rigid body motions.
 *
 * @attention bundle.output must contain a valid optimal solution from a prior
 * call to GrowthDistance with the same set pair and rigid body transforms. If
 * the solution status is not Optimal, the derivative is set to zero.
 *
 * @attention The derivative depends on the twist frame of reference specified
 * in settings.twist_frame (Spatial, Hybrid, or Body).
 *
 * @param[in]     set1,set2     Convex sets.
 * @param[in]     state1,state2 Kinematic states for the sets.
 * @param[in]     settings      Settings.
 * @param[in,out] bundle        Output bundle.
 * @return        Derivative of the growth distance.
 */
///@{
template <int dim>
Real GdDerivative(const ConvexSet<dim>* set1, const KinematicState<dim>& state1,
                  const ConvexSet<dim>* set2, const KinematicState<dim>& state2,
                  const Settings& settings, const OutputBundle<dim>& bundle);

template <int dim>
Real GdDerivative(const ConvexSet<dim>* set1, const KinematicState<dim>& state1,
                  const Halfspace<dim>* set2, const KinematicState<dim>& state2,
                  const Settings& settings, const OutputBundle<dim>& bundle);
///@}

/**
 * @name Growth distance gradient functions
 * @brief Gradient of the growth distance function for 2D and 3D convex sets
 * (including half-spaces) with respect to rigid body motions.
 *
 * @attention bundle.output must contain a valid optimal solution from a prior
 * call to GrowthDistance with the same set pair and rigid body transforms. If
 * the solution status is not Optimal, the gradient is set to zero.
 *
 * @attention The gradient depends on the twist frame of reference specified
 * in settings.twist_frame (Spatial, Hybrid, or Body).
 *
 * @param[in]     set1,set2 Convex sets.
 * @param[in]     tf1,tf2   Rigid body transformations for the sets.
 * @param[in]     settings  Settings.
 * @param[in,out] bundle    Output bundle.
 */
///@{
template <int dim>
void GdGradient(const ConvexSet<dim>* set1, const Transformr<dim>& tf1,
                const ConvexSet<dim>* set2, const Transformr<dim>& tf2,
                const Settings& settings, const OutputBundle<dim>& bundle);

template <int dim>
void GdGradient(const ConvexSet<dim>* set1, const Transformr<dim>& tf1,
                const Halfspace<dim>* set2, const Transformr<dim>& tf2,
                const Settings& settings, const OutputBundle<dim>& bundle);
///@}

}  // namespace dgd

#endif  // DGD_DGD_H_
