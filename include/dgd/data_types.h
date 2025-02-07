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

template <typename Derived>
using MatrixBase = Eigen::MatrixBase<Derived>;

template <int dim>
using MatXf = Eigen::Matrix<Real, dim, -1>;
typedef MatXf<2> Mat2Xf;
typedef MatXf<3> Mat3Xf;

constexpr Real kInf{std::numeric_limits<Real>::infinity()};
constexpr Real kEps{std::numeric_limits<Real>::epsilon()};

}  // namespace dgd

#endif  // DGD_DATA_TYPES_H_
