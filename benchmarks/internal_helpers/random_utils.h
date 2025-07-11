#ifndef DGD_BENCHMARKS_INTERNAL_HELPERS_RANDOM_UTILS_H_
#define DGD_BENCHMARKS_INTERNAL_HELPERS_RANDOM_UTILS_H_

#include "dgd/data_types.h"
#include "dgd/utils/random.h"

namespace dgd {

namespace internal {

// Sets random rigid body transforms for a pair of rigid bodies.
template <int dimh>
inline void SetRandomRigidBodyTransforms(Rng& rng, Matr<dimh, dimh>& tf1,
                                         Matr<dimh, dimh>& tf2, Real range_from,
                                         Real range_to) {
  rng.RandomRigidBodyTransform(range_from, range_to, tf1);
  rng.RandomRigidBodyTransform(range_from, range_to, tf2);
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
inline void UpdateRigidBodyTransform(Matr<dim + 1, dim + 1>& tf,
                                     const Vecr<dim>& dx,
                                     const Matr<dim, dim>& drot) {
  tf.template block<dim, 1>(0, dim) += dx;
  tf.template block<dim, dim>(0, 0) *= drot;
}

}  // namespace internal

}  // namespace dgd

#endif  // DGD_BENCHMARKS_INTERNAL_HELPERS_RANDOM_UTILS_H_
