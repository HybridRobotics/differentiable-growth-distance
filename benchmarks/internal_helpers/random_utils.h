#ifndef DGD_BENCHMARKS_INTERNAL_HELPERS_RANDOM_UTILS_H_
#define DGD_BENCHMARKS_INTERNAL_HELPERS_RANDOM_UTILS_H_

#include "dgd/data_types.h"
#include "dgd/utils/random.h"

namespace dgd {

namespace internal {

// Sets random rigid body transforms for a pair of rigid bodies.
template <int hdim>
inline void SetRandomTransforms(Rng& rng, Matr<hdim, hdim>& tf1,
                                Matr<hdim, hdim>& tf2, Real range_from,
                                Real range_to) {
  rng.RandomTransform(range_from, range_to, tf1);
  rng.RandomTransform(range_from, range_to, tf2);
}

// Sets a random displacement.
template <int dim>
inline void SetRandomDisplacement(Rng& rng, Vecr<dim>& dx, Matr<dim, dim>& drot,
                                  Real dx_max, Real ang_max) {
  rng.RandomUnitVector(dx);
  dx *= rng.Random(dx_max);
  rng.RandomRotation(drot, ang_max);
}

// Updates the rigid body transform using the screw.
template <int dim>
inline void UpdateTransform(Matr<dim + 1, dim + 1>& tf, const Vecr<dim>& dx,
                            const Matr<dim, dim>& drot) {
  Affine(tf) += dx;
  Linear(tf) *= drot;
}

}  // namespace internal

}  // namespace dgd

#endif  // DGD_BENCHMARKS_INTERNAL_HELPERS_RANDOM_UTILS_H_
