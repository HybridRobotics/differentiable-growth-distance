#ifndef DGD_BENCHMARKS_INTERNAL_HELPERS_MATH_UTILS_H_
#define DGD_BENCHMARKS_INTERNAL_HELPERS_MATH_UTILS_H_

#include "dgd/data_types.h"
#include "dgd/utils/random.h"

namespace dgd {

namespace bench {

// Sets random rigid body transforms for a pair of rigid bodies.
template <int hdim>
inline void SetRandomTransforms(Rng& rng, Matr<hdim, hdim>& tf1,
                                Matr<hdim, hdim>& tf2, Real range_from,
                                Real range_to) {
  tf1 = rng.RandomTransform<hdim - 1>(range_from, range_to);
  tf2 = rng.RandomTransform<hdim - 1>(range_from, range_to);
}

// Sets a random displacement.
template <int dim>
inline void SetRandomDisplacement(Rng& rng, Vecr<dim>& dx, Matr<dim, dim>& drot,
                                  Real dx_max, Real ang_max) {
  dx = rng.Random(dx_max) * rng.RandomUnitVector<dim>();
  drot = rng.RandomRotation<dim>(ang_max);
}

// Updates a rigid body transform using a screw.
template <int dim>
inline void UpdateTransform(Matr<dim + 1, dim + 1>& tf, const Vecr<dim>& dx,
                            const Matr<dim, dim>& drot) {
  Affine(tf) += dx;
  Linear(tf) *= drot;
}

}  // namespace bench

}  // namespace dgd

#endif  // DGD_BENCHMARKS_INTERNAL_HELPERS_MATH_UTILS_H_
