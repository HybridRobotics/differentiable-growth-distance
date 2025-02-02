#ifndef DGD_DATA_TYPES_H_
#define DGD_DATA_TYPES_H_

#include <Eigen/Core>
#include <limits>

namespace dgd {

typedef float Real;

template <int dim>
using Vecf = Eigen::Vector<Real, dim>;
typedef Vecf<2> Vec2f;
typedef Vecf<3> Vec3f;

constexpr Real kInf{std::numeric_limits<Real>::infinity()};

}  // namespace dgd

#endif  // DGD_DATA_TYPES_H_
