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
 * @brief Differentiable growth distance algorithm implementation.
 */

#include "dgd/dgd.h"

#include <iostream>

#include "dgd/data_types.h"
#include "dgd/geometry/convex_set.h"
#include "dgd/geometry/halfspace.h"
#include "dgd/output.h"
#include "dgd/settings.h"
#include "dgd/solvers/bundle_scheme_impl.h"
#include "dgd/solvers/derivative_impl.h"
#include "dgd/solvers/dgd_halfspace_impl.h"

namespace dgd {

/*
 * Growth distance algorithm.
 */

template <int dim>
Real GrowthDistance(const ConvexSet<dim>* set1, const Transformr<dim>& tf1,
                    const ConvexSet<dim>* set2, const Transformr<dim>& tf2,
                    const Settings& settings, Output<dim>& out,
                    bool warm_start) {
  if ((warm_start) || (set1->IsPolytopic() && set2->IsPolytopic())) {
    return GrowthDistanceCp(set1, tf1, set2, tf2, settings, out, warm_start);
  } else {
    return GrowthDistanceTrn(set1, tf1, set2, tf2, settings, out, warm_start);
  }
}

template <int dim>
Real GrowthDistance(const ConvexSet<dim>* set1, const Transformr<dim>& tf1,
                    const Halfspace<dim>* set2, const Transformr<dim>& tf2,
                    const Settings& settings, Output<dim>& out,
                    bool warm_start) {
  return GrowthDistanceHalfspaceTpl(set1, tf1, set2, tf2, settings, out,
                                    warm_start);
}

template <int dim>
Real GrowthDistanceCp(const ConvexSet<dim>* set1, const Transformr<dim>& tf1,
                      const ConvexSet<dim>* set2, const Transformr<dim>& tf2,
                      const Settings& settings, Output<dim>& out,
                      bool warm_start) {
  return GrowthDistanceCpTpl(set1, tf1, set2, tf2, settings, out, warm_start);
}

template <int dim>
Real GrowthDistanceTrn(const ConvexSet<dim>* set1, const Transformr<dim>& tf1,
                       const ConvexSet<dim>* set2, const Transformr<dim>& tf2,
                       const Settings& settings, Output<dim>& out,
                       bool /*warm_start*/) {
  return GrowthDistanceTrnTpl(set1, tf1, set2, tf2, settings, out, false);
}

/*
 * Collision detection algorithm.
 */

template <int dim>
bool DetectCollision(const ConvexSet<dim>* set1, const Transformr<dim>& tf1,
                     const ConvexSet<dim>* set2, const Transformr<dim>& tf2,
                     const Settings& settings, Output<dim>& out,
                     bool warm_start) {
  return DetectCollisionTpl(set1, tf1, set2, tf2, settings, out, warm_start);
}

template <int dim>
bool DetectCollision(const ConvexSet<dim>* set1, const Transformr<dim>& tf1,
                     const Halfspace<dim>* set2, const Transformr<dim>& tf2,
                     const Settings& settings, Output<dim>& out,
                     bool warm_start) {
  return DetectCollisionHalfspaceTpl(set1, tf1, set2, tf2, settings, out,
                                     warm_start);
}

/*
 * KKT solution set null space algorithm.
 */

template <int dim>
int ComputeKktNullspace(const ConvexSet<dim>* set1, const Transformr<dim>& tf1,
                        const ConvexSet<dim>* set2, const Transformr<dim>& tf2,
                        const Settings& settings, Output<dim>& out) {
  return ComputeKktNullspaceTpl(set1, tf1, set2, tf2, settings, out);
}

template <int dim>
int ComputeKktNullspace(const ConvexSet<dim>* set1, const Transformr<dim>& tf1,
                        const Halfspace<dim>* set2, const Transformr<dim>& tf2,
                        const Settings& settings, Output<dim>& out) {
  return ComputeKktNullspaceHalfspaceTpl(set1, tf1, set2, tf2, settings, out);
}

/*
 * Explicit instantiations.
 */

template Real GrowthDistance(const ConvexSet<2>*, const Transformr<2>&,
                             const ConvexSet<2>*, const Transformr<2>&,
                             const Settings&, Output<2>&, bool);
template Real GrowthDistance(const ConvexSet<3>*, const Transformr<3>&,
                             const ConvexSet<3>*, const Transformr<3>&,
                             const Settings&, Output<3>&, bool);

template Real GrowthDistance(const ConvexSet<2>*, const Transformr<2>&,
                             const Halfspace<2>*, const Transformr<2>&,
                             const Settings&, Output<2>&, bool);
template Real GrowthDistance(const ConvexSet<3>*, const Transformr<3>&,
                             const Halfspace<3>*, const Transformr<3>&,
                             const Settings&, Output<3>&, bool);

template Real GrowthDistanceCp(const ConvexSet<2>*, const Transformr<2>&,
                               const ConvexSet<2>*, const Transformr<2>&,
                               const Settings&, Output<2>&, bool);
template Real GrowthDistanceCp(const ConvexSet<3>*, const Transformr<3>&,
                               const ConvexSet<3>*, const Transformr<3>&,
                               const Settings&, Output<3>&, bool);

template Real GrowthDistanceTrn(const ConvexSet<2>*, const Transformr<2>&,
                                const ConvexSet<2>*, const Transformr<2>&,
                                const Settings&, Output<2>&, bool);
template Real GrowthDistanceTrn(const ConvexSet<3>*, const Transformr<3>&,
                                const ConvexSet<3>*, const Transformr<3>&,
                                const Settings&, Output<3>&, bool);

template bool DetectCollision(const ConvexSet<2>*, const Transformr<2>&,
                              const ConvexSet<2>*, const Transformr<2>&,
                              const Settings&, Output<2>&, bool);
template bool DetectCollision(const ConvexSet<3>*, const Transformr<3>&,
                              const ConvexSet<3>*, const Transformr<3>&,
                              const Settings&, Output<3>&, bool);

template bool DetectCollision(const ConvexSet<2>*, const Transformr<2>&,
                              const Halfspace<2>*, const Transformr<2>&,
                              const Settings&, Output<2>&, bool);
template bool DetectCollision(const ConvexSet<3>*, const Transformr<3>&,
                              const Halfspace<3>*, const Transformr<3>&,
                              const Settings&, Output<3>&, bool);

template int ComputeKktNullspace(const ConvexSet<2>*, const Transformr<2>&,
                                 const ConvexSet<2>*, const Transformr<2>&,
                                 const Settings&, Output<2>&);
template int ComputeKktNullspace(const ConvexSet<3>*, const Transformr<3>&,
                                 const ConvexSet<3>*, const Transformr<3>&,
                                 const Settings&, Output<3>&);

template int ComputeKktNullspace(const ConvexSet<2>*, const Transformr<2>&,
                                 const Halfspace<2>*, const Transformr<2>&,
                                 const Settings&, Output<2>&);
template int ComputeKktNullspace(const ConvexSet<3>*, const Transformr<3>&,
                                 const Halfspace<3>*, const Transformr<3>&,
                                 const Settings&, Output<3>&);

}  // namespace dgd
