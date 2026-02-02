#include <gtest/gtest.h>

#include <cassert>
#include <cmath>
#include <type_traits>
#include <utility>
#include <vector>

#include "dgd/data_types.h"
#include "dgd/geometry/geometry_2d.h"
#include "dgd/geometry/geometry_3d.h"
#include "dgd/graham_scan.h"
#include "dgd/mesh_loader.h"
#include "dgd/utils/numerical_differentiation.h"
#include "dgd/utils/random.h"

namespace {

using namespace dgd;

// Generates uniformly distributed points on a circle.
void UniformCirclePoints(MatXr<2>& pts, int size) {
  assert(size >= 2);
  pts.resize(2, size);

  const VecXr ang = VecXr::LinSpaced(size + 1, Real(0.0), Real(2.0) * kPi);
  pts.row(0) = ang.head(size).transpose().array().cos();
  pts.row(1) = ang.head(size).transpose().array().sin();
}

// Generates points on a sphere.
void UniformSpherePoints(MatXr<3>& pts, int size_xy, int size_z,
                         Real z_off = Real(0.0)) {
  assert((size_xy >= 2) && (size_z >= 2));
  pts.resize(3, size_xy * size_z);

  const VecXr ang_xy =
      VecXr::LinSpaced(size_xy + 1, Real(0.0), Real(2.0) * kPi);
  MatXr<2> pts_xy(2, size_xy);
  pts_xy.row(0) = ang_xy.head(size_xy).transpose().array().cos();
  pts_xy.row(1) = ang_xy.head(size_xy).transpose().array().sin();

  const VecXr ang_z = VecXr::LinSpaced(size_z, -kPi / Real(2.0) + z_off,
                                       kPi / Real(2.0) - z_off);
  for (int i = 0; i < size_z; ++i) {
    pts.block(0, size_xy * i, 2, size_xy) = pts_xy * std::cos(ang_z(i));
    pts.block(2, size_xy * i, 1, size_xy) =
        VecXr::Constant(size_xy, std::sin(ang_z(i))).transpose();
  }
}

// Computes the actual and numerical Jacobians of the support point function.
template <int dim>
bool ComputeSupportPointJacobian(const ConvexSet<dim>* set, const Vecr<dim>& n,
                                 Matr<dim, dim>& Dsp, Matr<dim, dim>& Dsp_num) {
  // Compute the actual Jacobian.
  SupportFunctionDerivatives<dim> deriv;
  set->SupportFunction(n.normalized(), deriv);
  if (!deriv.differentiable) return false;
  Dsp = deriv.Dsp;

  // Compute the numerical Jacobian.
  NumericalDifferentiator nd{};
  nd.Jacobian(
      [set](const Eigen::Ref<const VecXr>& x, Eigen::Ref<VecXr> y) -> void {
        Vecr<dim> n = x.head<dim>().normalized(), sp;
        set->SupportFunction(n, sp);
        y = sp;
      },
      n.normalized(), Dsp_num);
  return true;
}

// Assertion functions
const Real kTol = kSqrtEps;
const Real kTolJac = std::sqrt(kSqrtEps);

template <int dim>
bool AssertVectorEQ(const Vecr<dim>& v1, const Vecr<dim>& v2, Real tol) {
  return (v1 - v2).template lpNorm<Eigen::Infinity>() < tol;
}

template <int dim>
bool AssertMatrixEQ(const Matr<dim, dim>& m1, const Matr<dim, dim>& m2,
                    Real tol) {
  return (m1 - m2).template lpNorm<Eigen::Infinity>() < tol;
}

// Support function tests
// 2D convex set tests
//  Ellipse test
TEST(EllipseTest, SupportFunction) {
  const Real hlx = Real(3.0), hly = Real(2.0), margin = Real(0.0);
  auto set = Ellipse(hlx, hly, margin);

  EXPECT_EQ(set.inradius(), hly + margin);

  Real sv;
  Vec2r sp, spt, n;
  Matr<2, 2> Dsp, Dsp_num;
  MatXr<2> pts;
  UniformCirclePoints(pts, 16);
  const Vec2r len(hlx, hly);
  for (int i = 0; i < pts.cols(); ++i) {
    n = (pts.col(i).array() / len.array()).matrix();
    n = n / n.lpNorm<Eigen::Infinity>();
    spt = (pts.col(i).array() * len.array()).matrix() + margin * n;
    sv = set.SupportFunction(n, sp);
    EXPECT_NEAR(sv, n.dot(spt), kTol);
    ASSERT_PRED3(AssertVectorEQ<2>, sp, spt, kTol);
    if (ComputeSupportPointJacobian(&set, n, Dsp, Dsp_num)) {
      EXPECT_PRED3(AssertMatrixEQ<2>, Dsp, Dsp_num, kTolJac);
    }
  }
}

// Polygon test
TEST(PolygonTest, SupportFunction) {
  Rng rng;
  rng.SetSeed();
  const int npts = 100;
  const Real margin = Real(0.0), len = Real(5.0);

  std::vector<Vec2r> pts(npts), vert;
  for (int i = 0; i < npts; ++i) {
    pts[i] << rng.Random(len), rng.Random(len);
  }
  GrahamScan(pts, vert);
  Real inradius = ComputePolygonInradius(vert, Vec2r::Zero());

  auto set = Polygon(std::move(vert), inradius, margin);

  EXPECT_EQ(set.inradius(), inradius + margin);

  Real sv;
  Vec2r sp;
  MatXr<2> normals;
  UniformCirclePoints(normals, 16);
  for (int i = 0; i < normals.cols(); ++i) {
    sv = set.SupportFunction(normals.col(i), sp);
    ASSERT_GE(sv, inradius);
  }
}

//  Rectangle test
TEST(RectangleTest, SupportFunction) {
  const Real hlx = Real(3.0), hly = Real(2.0), margin = Real(0.0);
  auto set = Rectangle(hlx, hly, margin);

  EXPECT_EQ(set.inradius(), hly + margin);

  Real sv;
  Vec2r sp, spt, n;
  for (int i = 0; i < 4; ++i) {
    n = Vec2r(std::pow(Real(-1.0), i % 2), std::pow(Real(-1.0), (i / 2) % 2));
    spt = Vec2r(hlx * n(0), hly * n(1));
    spt += margin * n;
    sv = set.SupportFunction(n, sp);
    EXPECT_NEAR(sv, n.dot(spt), kTol);
    EXPECT_PRED3(AssertVectorEQ<2>, sp, spt, kTol);
  }
}

// 3D convex set tests
//  Cone test
TEST(ConeTest, SupportFunction) {
  const Real ha = kPi / Real(6.0), radius = Real(1.0), margin = Real(0.0);
  const Real height = radius / std::tan(ha);
  const Real rho = height / (Real(1.0) + Real(1.0) / std::sin(ha));
  auto set = Cone(radius, height, margin);

  EXPECT_NEAR(set.inradius(), rho + margin, kTol);
  EXPECT_NEAR(set.offset(), rho, kTol);

  Real sv;
  Vec3r sp, spt, n;
  Matr<3, 3> Dsp, Dsp_num;
  MatXr<3> pts;
  // Using size_z = 9 ensures that normal is not orthogonal
  // to the cone surface (for ha = 30 deg).
  UniformSpherePoints(pts, 16, 9, Real(1e-1));
  for (int i = 0; i < pts.cols(); ++i) {
    n = pts.col(i);
    n = n / n.lpNorm<Eigen::Infinity>();
    if (n.topRows<2>().norm() * std::tan(ha) < n(2)) {
      spt = Vec3r(Real(0.0), Real(0.0), height - rho);
    } else {
      spt.topRows<2>() = radius * n.topRows<2>().normalized();
      spt(2) = -rho;
    }
    spt += margin * n;
    sv = set.SupportFunction(n, sp);
    EXPECT_NEAR(sv, n.dot(spt), kTol);
    ASSERT_PRED3(AssertVectorEQ<3>, sp, spt, kTol);
    if (ComputeSupportPointJacobian(&set, n, Dsp, Dsp_num)) {
      EXPECT_PRED3(AssertMatrixEQ<3>, Dsp, Dsp_num, kTolJac);
    }
  }
}

// Cuboid test
TEST(CuboidTest, SupportFunction) {
  const Real hlx = Real(3.0), hly = Real(2.0), hlz = Real(1.5);
  const Real margin = Real(0.0);
  auto set = Cuboid(hlx, hly, hlz, margin);

  EXPECT_EQ(set.inradius(), hlz + margin);

  Real sv;
  Vec3r sp, spt, n;
  for (int i = 0; i < 8; ++i) {
    n = Vec3r(Real(std::pow(Real(-1.0), i % 2)),
              Real(std::pow(Real(-1.0), (i / 2) % 2)),
              Real(std::pow(Real(-1.0), (i / 4) % 2)));
    spt = Vec3r(hlx * n(0), hly * n(1), hlz * n(2));
    spt += margin * n;
    sv = set.SupportFunction(n, sp);
    EXPECT_NEAR(sv, n.dot(spt), kTol);
    ASSERT_PRED3(AssertVectorEQ<3>, sp, spt, kTol);
  }
}

// Cylinder test
TEST(CylinderTest, SupportFunction) {
  const Real hlx = Real(2.0), radius = Real(2.5), margin = Real(0.0);
  auto set = Cylinder(hlx, radius, margin);

  EXPECT_EQ(set.inradius(), hlx + margin);

  Real sv;
  Vec3r sp, spt, n;
  Matr<3, 3> Dsp, Dsp_num;
  MatXr<2> pts;
  UniformCirclePoints(pts, 16);
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < pts.cols(); ++j) {
      n(0) = Real(std::pow(Real(-1.0), i));
      n.bottomRows<2>() = pts.col(j);
      spt(0) = hlx * Real(std::pow(Real(-1.0), i));
      spt.bottomRows<2>() = radius * pts.col(j);
      spt += margin * n;
      sv = set.SupportFunction(n, sp);
      EXPECT_NEAR(sv, n.dot(spt), kTol);
      ASSERT_PRED3(AssertVectorEQ<3>, sp, spt, kTol);
      if (ComputeSupportPointJacobian(&set, n, Dsp, Dsp_num)) {
        EXPECT_PRED3(AssertMatrixEQ<3>, Dsp, Dsp_num, kTolJac);
      }
    }
}

//  Ellipsoid test
TEST(EllipsoidTest, SupportFunction) {
  const Real hlx = Real(3.0), hly = Real(2.0), hlz = Real(1.5),
             margin = Real(0.0);
  auto set = Ellipsoid(hlx, hly, hlz, margin);

  EXPECT_EQ(set.inradius(), hlz + margin);

  Real sv;
  Vec3r sp, spt, n;
  Matr<3, 3> Dsp, Dsp_num;
  MatXr<3> pts;
  UniformSpherePoints(pts, 16, 9);
  const Vec3r len(hlx, hly, hlz);
  for (int i = 0; i < pts.cols(); ++i) {
    n = (pts.col(i).array() / len.array()).matrix();
    n = n / n.lpNorm<Eigen::Infinity>();
    spt = (pts.col(i).array() * len.array()).matrix() + margin * n;
    sv = set.SupportFunction(n, sp);
    EXPECT_NEAR(sv, n.dot(spt), kTol);
    EXPECT_PRED3(AssertVectorEQ<3>, sp, spt, kTol);
    if (ComputeSupportPointJacobian(&set, n, Dsp, Dsp_num)) {
      EXPECT_PRED3(AssertMatrixEQ<3>, Dsp, Dsp_num, kTolJac);
    }
  }
}

//  Frustum test
TEST(FrustumTest, SupportFunction) {
  const Real margin = Real(0.0);
  std::vector<Frustum> sets;
  std::vector<Real> rb(8), rt(8), h(8);

  // Tall cylinder.
  Real radius = Real(1.0), height = Real(2.0);
  rb[0] = radius;
  rt[0] = radius;
  h[0] = height;
  sets.push_back(Frustum(radius, radius, height, margin));
  EXPECT_NEAR(sets[0].inradius(), radius + margin, kTol);
  EXPECT_NEAR(sets[0].offset(), radius, kTol);
  // Short cylinder.
  height = Real(0.5);
  rb[1] = radius;
  rt[1] = radius;
  h[1] = height;
  sets.push_back(Frustum(radius, radius, height, margin));
  EXPECT_NEAR(sets[1].inradius(), height / Real(2.0) + margin, kTol);
  EXPECT_NEAR(sets[1].offset(), height / Real(2.0), kTol);

  // Cone.
  Real ha = kPi / Real(6.0);
  height = radius / std::tan(ha);
  Real rho = height / (Real(1.0) + Real(1.0) / std::sin(ha));
  rb[2] = radius;
  rt[2] = Real(0.0);
  h[2] = height;
  sets.push_back(Frustum(radius, 0.0, height, margin));
  EXPECT_NEAR(sets[2].inradius(), rho + margin, kTol);
  EXPECT_NEAR(sets[2].offset(), rho, kTol);
  // Inverted cone.
  rb[3] = Real(0.0);
  rt[3] = radius;
  h[3] = height;
  sets.push_back(Frustum(Real(0.0), radius, height, margin));
  EXPECT_NEAR(sets[3].inradius(), rho + margin, kTol);
  EXPECT_NEAR(sets[3].offset(), height - rho, kTol);

  // Tall frustum with large base.
  Real height_cone = radius / std::tan(ha);
  height = height_cone / Real(2.0) + rho;
  Real small_radius = radius * (Real(1.0) - height / height_cone);
  rb[4] = radius;
  rt[4] = small_radius;
  h[4] = height;
  sets.push_back(Frustum(radius, small_radius, height, margin));
  EXPECT_NEAR(sets[4].inradius(), rho + margin, kTol);
  EXPECT_NEAR(sets[4].offset(), rho, kTol);
  // Tall frustum with small base.
  rb[5] = small_radius;
  rt[5] = radius;
  h[5] = height;
  sets.push_back(Frustum(small_radius, radius, height, margin));
  EXPECT_NEAR(sets[5].inradius(), rho + margin, kTol);
  EXPECT_NEAR(sets[5].offset(), height - rho, kTol);
  // Short frustum with large base.
  height = rho;
  small_radius = radius / height_cone * height;
  rb[6] = radius;
  rt[6] = small_radius;
  h[6] = height;
  sets.push_back(Frustum(radius, small_radius, height, margin));
  EXPECT_NEAR(sets[6].inradius(), height / Real(2.0) + margin, kTol);
  EXPECT_NEAR(sets[6].offset(), height / Real(2.0), kTol);
  // Short frustum with small base.
  rb[7] = small_radius;
  rt[7] = radius;
  h[7] = height;
  sets.push_back(Frustum(small_radius, radius, height, margin));
  EXPECT_NEAR(sets[7].inradius(), height / Real(2.0) + margin, kTol);
  EXPECT_NEAR(sets[7].offset(), height / Real(2.0), kTol);

  Real sv, tha, offset;
  Vec3r sp, spt, n;
  Matr<3, 3> Dsp, Dsp_num;
  MatXr<3> pts;
  UniformSpherePoints(pts, 16, 10, Real(1e-1));
  for (int k = 0; k < static_cast<int>(sets.size()); ++k) {
    const auto& set = sets[k];
    for (int i = 0; i < pts.cols(); ++i) {
      n = pts.col(i);
      n = n / n.lpNorm<Eigen::Infinity>();
      tha = (rb[k] - rt[k]) / h[k];
      offset = set.offset();
      if (n.topRows<2>().norm() * tha < n(2)) {
        spt.topRows<2>() = rt[k] * n.topRows<2>().normalized();
        spt(2) = h[k] - offset;
      } else {
        spt.topRows<2>() = rb[k] * n.topRows<2>().normalized();
        spt(2) = -offset;
      }
      spt += margin * n;
      sv = set.SupportFunction(n, sp);
      EXPECT_NEAR(sv, n.dot(spt), kTol);
      ASSERT_PRED3(AssertVectorEQ<3>, sp, spt, kTol);
      if (ComputeSupportPointJacobian(&set, n, Dsp, Dsp_num)) {
        EXPECT_PRED3(AssertMatrixEQ<3>, Dsp, Dsp_num, kTolJac);
      }
    }
  }
}

//  Mesh test
TEST(MeshTest, SupportFunction) {
  // Qhull computations can be unstable with float.
  if (typeid(Real) == typeid(float)) GTEST_SKIP();

  Rng rng;
  rng.SetSeed();
  const int nruns = 10;
  const int npts = 400;
  const Real inradius = Real(0.25), margin = Real(0.0);

  MeshLoader ml{};
  std::vector<Vec3r> pts(npts), vert;
  std::vector<int> graph;
  MatXr<3> normals;
  UniformSpherePoints(normals, 100, 10);
  Vec3r sp, spt, n;
  Real sv, svt;
  for (int i = 0; i < nruns; ++i) {
    for (int j = 0; j < npts; ++j) {
      pts[j] = rng.RandomUnitVector<3>();
    }
    ml.ProcessPoints(pts);
    bool valid = ml.MakeVertexGraph(vert, graph);

    ASSERT_TRUE(valid);

    auto polytope = Polytope(vert, inradius, margin);
    Mesh mesh(std::move(vert), std::move(graph), inradius, margin);

    // Support function test.
    for (int j = 0; j < normals.cols(); ++j) {
      n = normals.col(j);
      n = n / n.lpNorm<Eigen::Infinity>();
      svt = polytope.SupportFunction(n, spt);
      sv = mesh.SupportFunction(n, sp);
      ASSERT_NEAR(sv, svt, kTol);
    }
  }
}

// Polytope test
TEST(PolytopeTest, SupportFunction) {
  // Qhull computations can be unstable with float.
  if (typeid(Real) == typeid(float)) GTEST_SKIP();

  Rng rng;
  rng.SetSeed();
  const int npts = 1000;
  const Real margin = Real(0.0), len = Real(5.0);

  std::vector<Real> pts(3 * npts);
  for (int i = 0; i < 3 * npts; ++i) pts[i] = rng.Random(len);

  MeshLoader ml{};
  ml.ProcessPoints(pts);
  std::vector<Vec3r> vert;
  std::vector<int> graph;
  ml.MakeVertexGraph(vert, graph);
  Vec3r interior_point = Vec3r::Zero();
  Real inradius = ml.ComputeInradius(interior_point);

  auto set = Polytope(std::move(vert), inradius, margin);

  EXPECT_LE(set.inradius(), inradius + margin);

  Real sv;
  Vec3r sp;
  MatXr<3> normals;
  UniformSpherePoints(normals, 16, 9);
  for (int i = 0; i < normals.cols(); ++i) {
    sv = set.SupportFunction(normals.col(i), sp);
    ASSERT_GE(sv, inradius);
  }
}

// XD convex set tests
struct SetNameGenerator {
  template <typename T>
  static std::string GetName(int) {
    if constexpr (std::is_same_v<T, Stadium>) return "Stadium";
    if constexpr (std::is_same_v<T, Capsule>) return "Capsule";
    if constexpr (std::is_same_v<T, Circle>) return "Circle";
    if constexpr (std::is_same_v<T, Sphere>) return "Sphere";
    return "Unknown";
  }
};

template <class C>
class CapsuleSupportFunctionTest : public testing::Test {
 protected:
  CapsuleSupportFunctionTest() {}
  ~CapsuleSupportFunctionTest() {}
};

using CapsuleTypes = testing::Types<Stadium, Capsule>;
TYPED_TEST_SUITE(CapsuleSupportFunctionTest, CapsuleTypes, SetNameGenerator);

//  Sphere test
template <class C>
class SphereSupportFunctionTest : public testing::Test {
 protected:
  SphereSupportFunctionTest() {}
  ~SphereSupportFunctionTest() {}
};

using SphereTypes = testing::Types<Circle, Sphere>;
TYPED_TEST_SUITE(SphereSupportFunctionTest, SphereTypes, SetNameGenerator);

//  Capsule test
TYPED_TEST(CapsuleSupportFunctionTest, SupportFunction) {
  constexpr int dim = TypeParam::dimension();
  const Real hlx = Real(2.0), radius = Real(2.5), margin = Real(0.25);
  auto set = TypeParam(hlx, radius, margin);

  EXPECT_EQ(set.inradius(), radius + margin);

  MatXr<3> pts;
  // Odd number of points avoids zero x-component of normal.
  const int size_xy = 17;
  UniformSpherePoints(pts, size_xy, 9);
  const int size = (dim == 2) ? size_xy : static_cast<int>(pts.cols());

  Real sv;
  Vecr<dim> sp, spt, n;
  Matr<dim, dim> Dsp, Dsp_num;
  for (int i = 0; i < size; ++i) {
    n = pts.col(i).topRows<dim>().normalized();
    spt = (radius + margin) * n;
    spt(0) += std::copysign(hlx, n(0));
    sv = set.SupportFunction(n, sp);
    EXPECT_NEAR(sv, n.dot(spt), kTol);
    ASSERT_PRED3(AssertVectorEQ<dim>, sp, spt, kTol);
    if (ComputeSupportPointJacobian(&set, n, Dsp, Dsp_num)) {
      EXPECT_PRED3(AssertMatrixEQ<dim>, Dsp, Dsp_num, kTolJac);
    }
  }
}

//  Sphere test
TYPED_TEST(SphereSupportFunctionTest, SupportFunction) {
  constexpr int dim = TypeParam::dimension();
  const Real radius = Real(0.25);
  auto set = TypeParam(radius);

  EXPECT_EQ(set.inradius(), radius);

  Vecr<dim> sp;
  Matr<dim, dim> Dsp, Dsp_num;
  Vecr<dim> n = Vecr<dim>::UnitX();
  Real sv = set.SupportFunction(n, sp);
  EXPECT_NEAR(sv, radius, kTol);
  EXPECT_PRED3(AssertVectorEQ<dim>, sp, radius * Vecr<dim>::UnitX(), kTol);
  if (ComputeSupportPointJacobian(&set, n, Dsp, Dsp_num)) {
    EXPECT_PRED3(AssertMatrixEQ<dim>, Dsp, Dsp_num, kTolJac);
  }
}

}  // namespace
