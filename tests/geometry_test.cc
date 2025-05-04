#include <gtest/gtest.h>

#include <cassert>
#include <cmath>
#include <vector>

#include "dgd/data_types.h"
#include "dgd/geometry/2d/ellipse.h"
#include "dgd/geometry/2d/polygon.h"
#include "dgd/geometry/2d/rectangle.h"
#include "dgd/geometry/3d/cone.h"
#include "dgd/geometry/3d/cuboid.h"
#include "dgd/geometry/3d/cylinder.h"
#include "dgd/geometry/3d/ellipsoid.h"
#include "dgd/geometry/3d/frustum.h"
#include "dgd/geometry/3d/mesh.h"
#include "dgd/geometry/3d/polytope.h"
#include "dgd/geometry/xd/capsule.h"
#include "dgd/geometry/xd/sphere.h"
#include "dgd/graham_scan.h"
#include "dgd/mesh_loader.h"
#include "dgd/utils.h"

namespace {

using namespace dgd;

void UniformCirclePoints(Mat2Xf& pts, int size) {
  assert(size >= 2);
  pts.resize(2, size);

  const VecXf ang{VecXf::LinSpaced(size + 1, 0.0, 2.0 * kPi)};
  pts.row(0) = ang.head(size).transpose().array().cos();
  pts.row(1) = ang.head(size).transpose().array().sin();
}

void UniformSpherePoints(Mat3Xf& pts, int size_xy, int size_z,
                         Real z_off = 0.0) {
  assert((size_xy >= 2) && (size_z >= 2));
  pts.resize(3, size_xy * size_z);

  const VecXf ang_xy{VecXf::LinSpaced(size_xy + 1, 0.0, 2.0 * kPi)};
  Mat2Xf pts_xy(2, size_xy);
  pts_xy.row(0) = ang_xy.head(size_xy).transpose().array().cos();
  pts_xy.row(1) = ang_xy.head(size_xy).transpose().array().sin();

  const VecXf ang_z{VecXf::LinSpaced(size_z, -kPi / Real(2.0) + z_off,
                                     kPi / Real(2.0) - z_off)};
  for (int i = 0; i < size_z; ++i) {
    pts.block(0, size_xy * i, 2, size_xy) = pts_xy * std::cos(ang_z(i));
    pts.block(2, size_xy * i, 1, size_xy) =
        VecXf::Constant(size_xy, std::sin(ang_z(i))).transpose();
  }
}

// Assertion functions
const Real kTol{kEpsSqrt};

template <int dim>
bool AssertVectorEQ(const Vecf<dim>& v1, const Vecf<dim>& v2, Real tol) {
  return (v1 - v2).template lpNorm<Eigen::Infinity>() < tol;
}

// Support function tests
// 2D convex set tests
//  Ellipse test
TEST(EllipseTest, SupportFunction) {
  const Real hlx{3.0}, hly{2.0}, margin{0.0};
  auto set{Ellipse(hlx, hly, margin)};

  EXPECT_EQ(set.Inradius(), hly + margin);

  Real sv;
  Vec2f sp, sp_, n;
  Mat2Xf pts;
  UniformCirclePoints(pts, 16);
  const Vec2f len(hlx, hly);
  for (int i = 0; i < pts.cols(); ++i) {
    n = (pts.col(i).array() / len.array()).matrix();
    n = n / n.lpNorm<Eigen::Infinity>();
    sp_ = (pts.col(i).array() * len.array()).matrix() + margin * n;
    sv = set.SupportFunction(n, sp);
    EXPECT_NEAR(sv, n.dot(sp_), kTol);
    ASSERT_PRED3(AssertVectorEQ<2>, sp, sp_, kTol);
  }
}

// Polygon test
TEST(PolygonTest, SupportFunction) {
  SetDefaultSeed();
  const int npts{100};
  const Real margin{0.0}, len{5.0};

  std::vector<Vec2f> pts, vert;
  for (int i = 0; i < npts; ++i)
    pts.push_back(Vec2f(Random(-len, len), Random(-len, len)));
  GrahamScan(pts, vert);
  Real inradius{ComputePolygonInradius(vert, Vec2f::Zero())};

  auto set{Polygon(vert, margin, inradius)};

  EXPECT_EQ(set.Inradius(), inradius + margin);

  Real sv;
  Vec2f sp;
  Mat2Xf normals;
  UniformCirclePoints(normals, 16);
  for (int i = 0; i < normals.cols(); ++i) {
    sv = set.SupportFunction(normals.col(i), sp);
    ASSERT_GE(sv, inradius);
  }
}

//  Rectangle test
TEST(RectangleTest, SupportFunction) {
  const Real hlx{3.0}, hly{2.0}, margin{0.0};
  auto set{Rectangle(hlx, hly, margin)};

  EXPECT_EQ(set.Inradius(), hly + margin);

  Real sv;
  Vec2f sp, sp_, n;
  for (int i = 0; i < 4; ++i) {
    n = Vec2f(std::pow(-1.0, i % 2), std::pow(-1.0, (i / 2) % 2));
    sp_ = Vec2f(hlx * n(0), hly * n(1));
    sp_ += margin * n;
    sv = set.SupportFunction(n, sp);
    EXPECT_NEAR(sv, n.dot(sp_), kTol);
    EXPECT_PRED3(AssertVectorEQ<2>, sp, sp_, kTol);
  }
}

// 3D convex set tests
//  Cone test
TEST(ConeTest, SupportFunction) {
  const Real ha{kPi / 6.0}, radius{1.0}, margin{0.0};
  const Real height{radius / std::tan(ha)};
  const Real rho{height / (Real(1.0) + Real(1.0) / std::sin(ha))};
  auto set{Cone(radius, height, margin)};

  EXPECT_NEAR(set.Inradius(), rho + margin, kTol);
  EXPECT_NEAR(set.Offset(), rho, kTol);

  Real sv;
  Vec3f sp, sp_, n;
  Mat3Xf pts;
  // Using size_z = 9 ensures that normal is not orthogonal
  // to the cone surface (for ha = 30 deg).
  UniformSpherePoints(pts, 16, 9, Real(1e-5));
  for (int i = 0; i < pts.cols(); ++i) {
    n = pts.col(i);
    n = n / n.lpNorm<Eigen::Infinity>();
    if (n.topRows<2>().norm() * std::tan(ha) < n(2))
      sp_ = Vec3f(0.0, 0.0, height - rho);
    else {
      sp_.topRows<2>() = radius * n.topRows<2>().normalized();
      sp_(2) = -rho;
    }
    sp_ += margin * n;
    sv = set.SupportFunction(n, sp);
    EXPECT_NEAR(sv, n.dot(sp_), kTol);
    ASSERT_PRED3(AssertVectorEQ<3>, sp, sp_, kTol);
  }
}

// Cuboid test
TEST(CuboidTest, SupportFunction) {
  const Real hlx{3.0}, hly{2.0}, hlz{1.5}, margin{0.0};
  auto set{Cuboid(hlx, hly, hlz, margin)};

  EXPECT_EQ(set.Inradius(), hlz + margin);

  Real sv;
  Vec3f sp, sp_, n;
  for (int i = 0; i < 8; ++i) {
    n = Vec3f(Real(std::pow(-1.0, i % 2)), Real(std::pow(-1.0, (i / 2) % 2)),
              Real(std::pow(-1.0, (i / 4) % 2)));
    sp_ = Vec3f(hlx * n(0), hly * n(1), hlz * n(2));
    sp_ += margin * n;
    sv = set.SupportFunction(n, sp);
    EXPECT_NEAR(sv, n.dot(sp_), kTol);
    ASSERT_PRED3(AssertVectorEQ<3>, sp, sp_, kTol);
  }
}

// Cylinder test
TEST(CylinderTest, SupportFunction) {
  const Real hlx{2.0}, radius{2.5}, margin{0.0};
  auto set{Cylinder(hlx, radius, margin)};

  EXPECT_EQ(set.Inradius(), hlx + margin);

  Real sv;
  Vec3f sp, sp_, n;
  Mat2Xf pts;
  UniformCirclePoints(pts, 16);
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < pts.cols(); ++j) {
      n(0) = Real(std::pow(-1.0, i));
      n.bottomRows<2>() = pts.col(j);
      sp_(0) = hlx * Real(std::pow(-1.0, i));
      sp_.bottomRows<2>() = radius * pts.col(j);
      sp_ += margin * n;
      sv = set.SupportFunction(n, sp);
      EXPECT_NEAR(sv, n.dot(sp_), kTol);
      ASSERT_PRED3(AssertVectorEQ<3>, sp, sp_, kTol);
    }
}

//  Ellipsoid test
TEST(EllipsoidTest, SupportFunction) {
  const Real hlx{3.0}, hly{2.0}, hlz{1.5}, margin{0.0};
  auto set{Ellipsoid(hlx, hly, hlz, margin)};

  EXPECT_EQ(set.Inradius(), hlz + margin);

  Real sv;
  Vec3f sp, sp_, n;
  Mat3Xf pts;
  UniformSpherePoints(pts, 16, 9);
  const Vec3f len(hlx, hly, hlz);
  for (int i = 0; i < pts.cols(); ++i) {
    n = (pts.col(i).array() / len.array()).matrix();
    n = n / n.lpNorm<Eigen::Infinity>();
    sp_ = (pts.col(i).array() * len.array()).matrix() + margin * n;
    sv = set.SupportFunction(n, sp);
    EXPECT_NEAR(sv, n.dot(sp_), kTol);
    EXPECT_PRED3(AssertVectorEQ<3>, sp, sp_, kTol);
  }
}

//  Frustum test
TEST(FrustumTest, SupportFunction) {
  const Real margin{0.0};
  std::vector<Frustum> sets;
  std::vector<Real> rb(8), rt(8), h(8);

  // Tall cylinder.
  Real radius{1.0}, height{2.0};
  rb[0] = radius;
  rt[0] = radius;
  h[0] = height;
  sets.push_back(Frustum(radius, radius, height, margin));
  EXPECT_NEAR(sets[0].Inradius(), radius + margin, kTol);
  EXPECT_NEAR(sets[0].Offset(), radius, kTol);
  // Short cylinder.
  height = 0.5;
  rb[1] = radius;
  rt[1] = radius;
  h[1] = height;
  sets.push_back(Frustum(radius, radius, height, margin));
  EXPECT_NEAR(sets[1].Inradius(), height / Real(2.0) + margin, kTol);
  EXPECT_NEAR(sets[1].Offset(), height / Real(2.0), kTol);

  // Cone.
  Real ha{kPi / 6.0};
  height = radius / std::tan(ha);
  Real rho{height / (Real(1.0) + Real(1.0) / std::sin(ha))};
  rb[2] = radius;
  rt[2] = 0.0;
  h[2] = height;
  sets.push_back(Frustum(radius, 0.0, height, margin));
  EXPECT_NEAR(sets[2].Inradius(), rho + margin, kTol);
  EXPECT_NEAR(sets[2].Offset(), rho, kTol);
  // Inverted cone.
  rb[3] = 0.0;
  rt[3] = radius;
  h[3] = height;
  sets.push_back(Frustum(0.0, radius, height, margin));
  EXPECT_NEAR(sets[3].Inradius(), rho + margin, kTol);
  EXPECT_NEAR(sets[3].Offset(), height - rho, kTol);

  // Tall frustum with large base.
  Real height_cone{radius / std::tan(ha)};
  height = height_cone / Real(2.0) + rho;
  Real small_radius{radius * (Real(1.0) - height / height_cone)};
  rb[4] = radius;
  rt[4] = small_radius;
  h[4] = height;
  sets.push_back(Frustum(radius, small_radius, height, margin));
  EXPECT_NEAR(sets[4].Inradius(), rho + margin, kTol);
  EXPECT_NEAR(sets[4].Offset(), rho, kTol);
  // Tall frustum with small base.
  rb[5] = small_radius;
  rt[5] = radius;
  h[5] = height;
  sets.push_back(Frustum(small_radius, radius, height, margin));
  EXPECT_NEAR(sets[5].Inradius(), rho + margin, kTol);
  EXPECT_NEAR(sets[5].Offset(), height - rho, kTol);
  // Short frustum with large base.
  height = rho;
  small_radius = radius / height_cone * height;
  rb[6] = radius;
  rt[6] = small_radius;
  h[6] = height;
  sets.push_back(Frustum(radius, small_radius, height, margin));
  EXPECT_NEAR(sets[6].Inradius(), height / Real(2.0) + margin, kTol);
  EXPECT_NEAR(sets[6].Offset(), height / Real(2.0), kTol);
  // Short frustum with small base.
  rb[7] = small_radius;
  rt[7] = radius;
  h[7] = height;
  sets.push_back(Frustum(small_radius, radius, height, margin));
  EXPECT_NEAR(sets[7].Inradius(), height / Real(2.0) + margin, kTol);
  EXPECT_NEAR(sets[7].Offset(), height / Real(2.0), kTol);

  Real sv, tha, offset;
  Vec3f sp, sp_, n;
  Mat3Xf pts;
  UniformSpherePoints(pts, 16, 10, Real(1e-5));
  for (int k = 0; k < static_cast<int>(sets.size()); ++k) {
    const auto& set = sets[k];
    for (int i = 0; i < pts.cols(); ++i) {
      n = pts.col(i);
      n = n / n.lpNorm<Eigen::Infinity>();
      tha = std::abs(rb[k] - rt[k]) / h[k];
      offset = set.Offset();
      if (n.topRows<2>().norm() * tha < n(2)) {
        sp_.topRows<2>() = rt[k] * n.topRows<2>().normalized();
        sp_(2) = h[k] - offset;
      } else {
        sp_.topRows<2>() = rb[k] * n.topRows<2>().normalized();
        sp_(2) = -offset;
      }
      sp_ += margin * n;
      sv = set.SupportFunction(n, sp);
      EXPECT_NEAR(sv, n.dot(sp_), kTol);
      ASSERT_PRED3(AssertVectorEQ<3>, sp, sp_, kTol);
    }
  }
}

//  Mesh test
TEST(MeshTest, SupportFunction) {
  // Qhull computations can be unstable with float.
  if (typeid(Real) == typeid(float)) GTEST_SKIP();

  SetDefaultSeed();
  const int nruns{10};
  const int npts{400};
  const Real inradius{0.25}, margin{0.0};

  MeshLoader ml{};
  std::vector<Vec3f> pts(npts), vert;
  std::vector<int> graph;
  Mat3Xf normals;
  UniformSpherePoints(normals, 100, 10);
  Vec3f sp, sp_, n;
  Real sv, sv_;
  for (int i = 0; i < nruns; ++i) {
    for (int j = 0; j < npts; ++j)
      pts[j] = Vec3f(Random(1.0), Random(1.0), Random(1.0)).normalized();
    ml.ProcessPoints(pts);
    bool valid{ml.MakeVertexGraph(vert, graph)};

    ASSERT_TRUE(valid);

    auto polytope{Polytope(vert, margin, inradius)};
    auto mesh{Mesh(vert, graph, margin, inradius)};

    // Support function test.
    for (int j = 0; j < normals.cols(); ++j) {
      n = normals.col(j);
      n = n / n.lpNorm<Eigen::Infinity>();
      sv_ = polytope.SupportFunction(n, sp_);
      sv = mesh.SupportFunction(n, sp);
      ASSERT_NEAR(sv, sv_, kTol);
    }
  }
}

// Polytope test
TEST(PolytopeTest, SupportFunction) {
  // Qhull computations can be unstable with float.
  if (typeid(Real) == typeid(float)) GTEST_SKIP();

  SetDefaultSeed();
  const int npts{1000};
  const Real margin{0.0}, len{5.0};

  std::vector<Real> pts;
  for (int i = 0; i < 3 * npts; ++i) pts.push_back(Random(-len, len));

  MeshLoader ml{};
  ml.ProcessPoints(pts);
  std::vector<Vec3f> vert;
  std::vector<int> graph;
  ml.MakeVertexGraph(vert, graph);
  Vec3f interior_point{Vec3f::Zero()};
  Real inradius{ml.ComputeInradius(interior_point)};

  auto set{Polytope(vert, margin, inradius)};

  EXPECT_EQ(set.Inradius(), inradius + margin);

  Real sv;
  Vec3f sp, n;
  Mat3Xf normals;
  UniformSpherePoints(normals, 16, 9);
  for (int i = 0; i < normals.cols(); ++i) {
    sv = set.SupportFunction(normals.col(i), sp);
    ASSERT_GE(sv, inradius);
  }
}

// XD convex set tests
//  Capsule test
template <class C>
class CapsuleTest : public testing::Test {
 protected:
  CapsuleTest() {}

  ~CapsuleTest() {}
};

typedef testing::Types<Capsule<2>, Capsule<3>> CapsuleTypes;
TYPED_TEST_SUITE(CapsuleTest, CapsuleTypes);

TYPED_TEST(CapsuleTest, SupportFunction) {
  constexpr int dim{TypeParam::Dimension()};
  const Real hlx{2.0}, radius{2.5}, margin{0.25};
  auto set{TypeParam(hlx, radius, margin)};

  EXPECT_EQ(set.Inradius(), radius + margin);

  Mat3Xf pts;
  // Odd number of points avoids zero x-component of normal.
  const int size_xy{17};
  UniformSpherePoints(pts, size_xy, 9);
  const int size = (dim == 2) ? size_xy : static_cast<int>(pts.cols());

  Real sv;
  Vecf<dim> sp, sp_, n;
  for (int i = 0; i < size; ++i) {
    n = pts.col(i).topRows<dim>().normalized();
    sp_ = (radius + margin) * n;
    sp_(0) += std::copysign(hlx, n(0));
    sv = set.SupportFunction(n, sp);
    EXPECT_NEAR(sv, n.dot(sp_), kTol);
    ASSERT_PRED3(AssertVectorEQ<dim>, sp, sp_, kTol);
  }
}

//  Sphere test (to test compilation)
template <class C>
class SphereTest : public testing::Test {
 protected:
  SphereTest() {}

  ~SphereTest() {}
};

typedef testing::Types<Sphere<2>, Sphere<3>> SphereTypes;
TYPED_TEST_SUITE(SphereTest, SphereTypes);

TYPED_TEST(SphereTest, SupportFunction) {
  constexpr int dim{TypeParam::Dimension()};
  const Real radius{0.25};
  auto set{TypeParam(radius)};

  EXPECT_EQ(set.Inradius(), radius);

  Vecf<dim> sp;
  Real sv{set.SupportFunction(Vecf<dim>::UnitX(), sp)};
  EXPECT_NEAR(sv, radius, kTol);
  EXPECT_PRED3(AssertVectorEQ<dim>, sp, radius * Vecf<dim>::UnitX(), kTol);
}

}  // namespace
