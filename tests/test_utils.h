#ifndef DGD_TEST_UTILS_H_
#define DGD_TEST_UTILS_H_

#include <gtest/gtest.h>

#include <array>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

#include "dgd/data_types.h"
#include "dgd/geometry/geometry_2d.h"
#include "dgd/geometry/geometry_3d.h"
#include "dgd/graham_scan.h"
#include "dgd/mesh_loader.h"
#include "dgd/output.h"
#include "dgd/settings.h"
#include "dgd/utils/random.h"

namespace dgd {
namespace test {

// ---------------------------------------------------------------------------
// Tolerances
// ---------------------------------------------------------------------------

/// Absolute tolerance for scalar comparisons (support values, distances, etc.).
inline const Real kTol = kSqrtEps;

/// Tolerance for quantities involving numerical gradients.
inline const Real kTolGrad = std::sqrt(kSqrtEps);

/// Tolerance for quantities involving numerical Hessians.
/// Note: The FD Hessian with step h has truncation error O(h^2 * |f^(4)|).
inline const Real kTolHess = Real(2.0) * kTolGrad;

// ---------------------------------------------------------------------------
// Assertion predicates (for use with EXPECT_PRED3 / EXPECT_PRED4)
// ---------------------------------------------------------------------------

/// Returns true if ||v1 - v2||_inf < tol.
template <int dim>
bool VectorNear(const Vecr<dim>& v1, const Vecr<dim>& v2, Real tol) {
  return (v1 - v2).template lpNorm<Eigen::Infinity>() < tol;
}

/// Returns true if ||m1 - m2||_inf < tol.
template <int dim>
bool MatrixNear(const Matr<dim, dim>& m1, const Matr<dim, dim>& m2, Real tol) {
  return (m1 - m2).template lpNorm<Eigen::Infinity>() < tol;
}

/// Returns true if ||m1 - m2||_inf < tol.
template <int dim>
bool JacobianNear(const Matr<dim, SeDim<dim>()>& m1,
                  const Matr<dim, SeDim<dim>()>& m2, Real tol) {
  return (m1 - m2).template lpNorm<Eigen::Infinity>() < tol;
}

// ---------------------------------------------------------------------------
// Math utilities
// ---------------------------------------------------------------------------

/// Checks if a matrix is an orthonormal basis.
template <int row, int col>
bool IsOrthonormalBasis(const Matr<row, col>& basis, int dim, Real tol) {
  for (int i = 0; i < dim; ++i) {
    if (std::abs(basis.col(i).norm() - Real(1.0)) > tol) return false;
    for (int j = i + 1; j < dim; ++j) {
      if (std::abs(basis.col(i).dot(basis.col(j))) > tol) return false;
    }
  }
  return true;
}

/// Twist frame types.
constexpr std::array<TwistFrame, 3> kTwistFrames = {
    TwistFrame::Spatial, TwistFrame::Hybrid, TwistFrame::Body};

/// Integrates a twist into a transformation.
template <int dim>
Transformr<dim> IntegrateTransform(const Transformr<dim>& tf,
                                   const Twistr<dim>& tw,
                                   TwistFrame twist_frame) {
  // Convert twist to hybrid frame.
  Twistr<dim> tw_h = tw;
  if (twist_frame == TwistFrame::Spatial) {
    Linear(tw_h) = VelocityAtPoint<dim>(tw, Affine(tf));
    Angular(tw_h) = Angular(tw);
  } else if (twist_frame == TwistFrame::Body) {
    Linear(tw_h) = Linear(tf) * Linear(tw);
    if constexpr (dim == 3) {
      Angular(tw_h) = Linear(tf) * Angular(tw);
    } else {
      Angular(tw_h) = Angular(tw);
    }
  }

  const auto w_h = Angular(tw_h);
  // First-order approximation for rotation.
  Rotationr<dim> dR;
  if constexpr (dim == 2) {
    const Real c = std::cos(w_h), s = std::sin(w_h);
    dR << c, -s, s, c;
  } else {
    const Real theta = w_h.norm();
    if (theta < kEps) {
      dR = Rotation3r::Identity() + Hat(w_h);
    } else {
      dR = AngleAxisToRotation(w_h / theta, theta);
    }
  }

  // Compute updated transformation.
  Transformr<dim> tf_new = Transformr<dim>::Identity();
  Linear(tf_new) = dR * Linear(tf);
  Affine(tf_new) = Affine(tf) + Linear(tw_h);

  return tf_new;
}

// ---------------------------------------------------------------------------
// Printing utilities
// ---------------------------------------------------------------------------

/// Prints a formatted vector.
template <typename T, int dim>
inline void PrintVector(const Vec<T, dim>& v) {
  for (int i = 0; i < dim - 1; ++i) std::cout << v(i) << ", ";
  std::cout << v(dim - 1) << std::endl;
}

/// Prints a formatted matrix.
template <int dim, bool csv = true>
inline void PrintMatrix(const Matr<dim, dim>& m) {
  const std::string prefix = csv ? "  " : "  (";
  const std::string suffix = csv ? "," : ")";
  for (int i = 0; i < dim; ++i) {
    std::cout << prefix;
    for (int j = 0; j < dim - 1; ++j) std::cout << m(i, j) << ", ";
    std::cout << m(i, dim - 1) << suffix << std::endl;
  }
}

/// Prints the growth distance problem setup.
template <int dim>
void PrintSetup(const ConvexSet<dim>* set1, const Transformr<dim>& tf1,
                const ConvexSet<dim>* set2, const Transformr<dim>& tf2,
                const Output<dim>& out) {
  std::cout << "--- Solution Output ---" << std::endl;
  constexpr int max_precision = std::numeric_limits<dgd::Real>::max_digits10;
  std::cout << std::fixed << std::setprecision(max_precision);
  std::cout << "Transform 1:" << std::endl;
  PrintMatrix(tf1);
  std::cout << "Transform 2:" << std::endl;
  PrintMatrix(tf2);
  std::cout << "Set 1: ";
  set1->PrintInfo();
  std::cout << "Set 2: ";
  set2->PrintInfo();
  std::cout << "Output: " << std::endl
            << "  Status: " << SolutionStatusName(out.status) << std::endl
            << "  GD (lower): " << out.growth_dist_lb << std::endl
            << "  GD (upper): " << out.growth_dist_ub << std::endl
            << "  #Iter: " << out.iter << std::endl;
  std::cout << "  bc: ";
  PrintVector(out.bc);
  std::cout << "  Idx (s1): ";
  PrintVector(out.idx_s1);
  std::cout << "  z1: ";
  PrintVector(out.z1);
  std::cout << "  Idx (s2): ";
  PrintVector(out.idx_s2);
  std::cout << "  z2: ";
  PrintVector(out.z2);
  std::cout << "  normal: ";
  PrintVector(out.normal);
  std::cout.unsetf(std::ios_base::fixed);
  std::cout << std::setprecision(6);
}

// ---------------------------------------------------------------------------
// Point generation utilities
// ---------------------------------------------------------------------------

/// Generates `size` uniformly spaced points on the unit circle.
inline std::vector<Vec2r> UniformCirclePoints(int size) {
  assert(size >= 2);
  std::vector<Vec2r> pts(size);
  const Real dtheta = Real(2.0) * kPi / static_cast<Real>(size);
  for (int i = 0; i < size; ++i) {
    const Real theta = dtheta * static_cast<Real>(i);
    pts[i] = Vec2r(std::cos(theta), std::sin(theta));
  }
  return pts;
}

/// Generates `size_xy * size_z` points on the unit sphere.
inline std::vector<Vec3r> UniformSpherePoints(int size_xy, int size_z,
                                              Real z_off = Real(0.0)) {
  assert(size_xy >= 2 && size_z >= 2);
  std::vector<Vec3r> pts;
  pts.reserve(size_xy * size_z);

  const Real dtheta = Real(2.0) * kPi / static_cast<Real>(size_xy);
  const Real phi_lo = -kPi / Real(2.0) + z_off;
  const Real phi_hi = kPi / Real(2.0) - z_off;
  const Real dphi = (phi_hi - phi_lo) / static_cast<Real>(size_z - 1);

  for (int j = 0; j < size_z; ++j) {
    const Real phi = phi_lo + dphi * static_cast<Real>(j);
    const Real cp = std::cos(phi);
    const Real sp = std::sin(phi);
    for (int i = 0; i < size_xy; ++i) {
      const Real theta = dtheta * static_cast<Real>(i);
      pts.emplace_back(cp * std::cos(theta), cp * std::sin(theta), sp);
    }
  }
  return pts;
}

// ---------------------------------------------------------------------------
// Convex-set factory (Ellipse–Polygon in 2D, Cone–Mesh in 3D)
// ---------------------------------------------------------------------------

/// Convex set smart pointer type.
template <int dim>
using ConvexSetPtr = std::unique_ptr<ConvexSet<dim>>;

/// Creates a pair of convex sets for testing.
template <int dim>
void MakeConvexSetPair(ConvexSetPtr<dim>& set1, ConvexSetPtr<dim>& set2,
                       Real margin1, Real margin2, int npts = 200);

/// Ellipse–Polygon pair specialization.
template <>
inline void MakeConvexSetPair<2>(ConvexSetPtr<2>& set1, ConvexSetPtr<2>& set2,
                                 Real margin1, Real margin2, int npts) {
  const Real hlx = Real(3.0), hly = Real(2.0);
  set1 = std::make_unique<Ellipse>(hlx, hly, margin1);

  Rng rng;
  rng.SetSeed(42);
  const Real len = Real(2.0);
  std::vector<Vec2r> pts, vert;
  for (int i = 0; i < npts; ++i) {
    Vec2r v;
    v << rng.Random(), rng.Random();
    v *= len / (v.template lpNorm<6>() + kEps);
    pts.push_back(v);
  }
  GrahamScan(pts, vert);
  const Real inradius = ComputePolygonInradius(vert, Vec2r::Zero());
  set2 = std::make_unique<Polygon>(vert, inradius, margin2);
}

/// Cone–Mesh pair specialization.
template <>
inline void MakeConvexSetPair<3>(ConvexSetPtr<3>& set1, ConvexSetPtr<3>& set2,
                                 Real margin1, Real margin2, int npts) {
  const Real ha = kPi / Real(6.0), radius = Real(1.0);
  const Real height = radius / std::tan(ha);
  set1 = std::make_unique<Cone>(radius, height, margin1);

  Rng rng;
  rng.SetSeed(42);
  const Real len = Real(2.0);
  MeshLoader ml{};
  std::vector<Vec3r> pts, vert;
  std::vector<int> graph;
  for (int i = 0; i < npts; ++i) {
    Vec3r v;
    v << rng.Random(), rng.Random(), rng.Random();
    v *= len / (v.template lpNorm<4>() + kEps);
    pts.push_back(v);
  }
  ml.ProcessPoints(pts);
  const bool valid = ml.MakeVertexGraph(vert, graph);
  ASSERT_TRUE(valid);
  Vec3r interior;
  const Real inradius = ml.ComputeInradius(interior);
  for (auto& v : vert) v -= interior;
  set2 = std::make_unique<Mesh>(vert, graph, inradius, margin2);
}

}  // namespace test
}  // namespace dgd

#endif  // DGD_TEST_UTILS_H_
