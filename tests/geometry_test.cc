#include <gtest/gtest.h>

#include <Eigen/Core>
#include <cmath>

#include "dgd/data_types.h"
#include "dgd/growth_distance.h"
#include "dgd/sets.h"

namespace {

using namespace dgd;

void UniformCirclePoints(Mat2Xf& pts, int size) {
  assert(size >= 2);
  pts.resize(2, size);

  const VecXf ang{VecXf::LinSpaced(size, 0.0, 2.0 * kPi)};
  pts.row(0) = ang.transpose().array().cos();
  pts.row(1) = ang.transpose().array().sin();
}

void UniformSpherePoints(Mat3Xf& pts, int xy_size, int z_size,
                         Real z_off = 0.0) {
  assert((xy_size >= 2) && (z_size >= 2));
  pts.resize(3, xy_size * z_size);

  const VecXf xy_ang{VecXf::LinSpaced(xy_size, 0.0, 2.0 * kPi)};
  Mat2Xf xy_pts(2, xy_size);
  xy_pts.row(0) = xy_ang.transpose().array().cos();
  xy_pts.row(1) = xy_ang.transpose().array().sin();

  const VecXf z_ang{
      VecXf::LinSpaced(z_size, -kPi / 2.0 + z_off, kPi / 2.0 - z_off)};
  for (int i = 0; i < z_size; ++i) {
    pts.block(0, xy_size * i, 2, xy_size) = xy_pts * std::cos(z_ang(i));
    pts.block(2, xy_size * i, 1, xy_size) =
        VecXf::Constant(xy_size, std::sin(z_ang(i))).transpose();
  }
}

// Assertion functions
const Eigen::IOFormat kVecFmt(4, Eigen::DontAlignCols, ", ", "\n", "[", "]");

template <int dim>
bool AssertVectorEQ(const Vecf<dim>& v1, const Vecf<dim>& v2, Real tol) {
  return (v1 - v2).template lpNorm<Eigen::Infinity>() < tol;
}

// Support function tests
const Real kTol{kEpsSqrt};

// 2D convex set tests
//  Ellipse test
TEST(EllipseTest, SupportFunction) {
  const Real hlx{3.0}, hly{2.0}, margin{0.1};
  auto set{Ellipse(hlx, hly, margin)};

  EXPECT_EQ(set.GetInradius(), hly + margin);

  Real sv;
  Vec2f sp, sp_, n;
  Mat2Xf pts;
  UniformCirclePoints(pts, 17);
  const Vec2f len(hlx, hly);
  for (int i = 0; i < pts.cols(); ++i) {
    n = (pts.col(i).array() / len.array()).matrix().normalized();
    sp_ = (pts.col(i).array() * len.array()).matrix() + margin * n;
    sv = set.SupportFunction(n, sp);
    EXPECT_NEAR(sv, n.dot(sp_), kTol);
    EXPECT_PRED3(AssertVectorEQ<2>, sp, sp_, kTol);
  }
}

//  Rectangle test
TEST(RectangleTest, SupportFunction) {
  const Real hlx{3.0}, hly{2.0}, margin{0.1};
  auto set{Rectangle(hlx, hly, margin)};

  EXPECT_EQ(set.GetInradius(), hly + margin);

  Real sv;
  Vec2f sp, sp_, n;
  for (int i = 0; i < 4; ++i) {
    n = Vec2f(std::pow(-1.0, i % 2), std::pow(-1.0, (i / 2) % 2));
    sp_ = Vec2f(hlx * n(0), hly * n(1));
    n.normalize();
    sp_ += margin * n;
    sv = set.SupportFunction(n, sp);
    EXPECT_NEAR(sv, n.dot(sp_), kTol);
    EXPECT_PRED3(AssertVectorEQ<2>, sp, sp_, kTol);
  }
}

// 3D convex set tests
//  Cone test
TEST(ConeTest, SupportFunction) {
  const Real ha{kPi / 6.0}, radius{1.0}, margin{0.1};
  const Real height{radius / std::tan(ha)};
  const Real rho{height / (Real(1.0) + Real(1.0) / std::sin(ha))};
  auto set{Cone(radius, height, margin)};

  EXPECT_EQ(set.GetInradius(), rho + margin);
  EXPECT_EQ(set.GetOffset(), rho);

  Real sv;
  Vec3f sp, sp_, n;
  Mat3Xf pts;
  // Using z_size = 9 ensures that normal is not orthogonal
  // to the cone surface (for ha = 30 deg).
  UniformSpherePoints(pts, 17, 9, 1e-5);
  for (int i = 0; i < pts.cols(); ++i) {
    n = pts.col(i);
    if (n.topRows<2>().norm() * std::tan(ha) < n(2))
      sp_ = Vec3f(0.0, 0.0, height - rho);
    else {
      sp_.topRows<2>() = n.topRows<2>().normalized();
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
  const Real hlx{3.0}, hly{2.0}, hlz{1.5}, margin{0.1};
  auto set{Cuboid(hlx, hly, hlz, margin)};

  EXPECT_EQ(set.GetInradius(), hlz + margin);

  Real sv;
  Vec3f sp, sp_, n;
  for (int i = 0; i < 8; ++i) {
    n = Vec3f(std::pow(-1.0, i % 2), std::pow(-1.0, (i / 2) % 2),
              std::pow(-1.0, (i / 4) % 2));
    sp_ = Vec3f(hlx * n(0), hly * n(1), hlz * n(2));
    n.normalize();
    sp_ += margin * n;
    sv = set.SupportFunction(n, sp);
    EXPECT_NEAR(sv, n.dot(sp_), kTol);
    ASSERT_PRED3(AssertVectorEQ<3>, sp, sp_, kTol);
  }
}

// Cylinder test
TEST(CylinderTest, SupportFunction) {
  const Real hlx{2.0}, radius{2.5}, margin{0.1};
  auto set{Cylinder(hlx, radius, margin)};

  EXPECT_EQ(set.GetInradius(), hlx + margin);

  Real sv;
  Vec3f sp, sp_, n;
  Mat2Xf pts;
  UniformCirclePoints(pts, 17);
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < pts.cols(); ++j) {
      n(0) = std::pow(-1.0, i);
      n.bottomRows<2>() = pts.col(j);
      n.normalize();
      sp_(0) = hlx * std::pow(-1.0, i);
      sp_.bottomRows<2>() = radius * pts.col(j);
      sp_ += margin * n;
      sv = set.SupportFunction(n, sp);
      EXPECT_NEAR(sv, n.dot(sp_), kTol);
      ASSERT_PRED3(AssertVectorEQ<3>, sp, sp_, kTol);
    }
}

//  Ellipsoid test
TEST(EllipsoidTest, SupportFunction) {
  const Real hlx{3.0}, hly{2.0}, hlz{1.5}, margin{0.1};
  auto set{Ellipsoid(hlx, hly, hlz, margin)};

  EXPECT_EQ(set.GetInradius(), hlz + margin);

  Real sv;
  Vec3f sp, sp_, n;
  Mat3Xf pts;
  UniformSpherePoints(pts, 17, 9);
  const Vec3f len(hlx, hly, hlz);
  for (int i = 0; i < pts.cols(); ++i) {
    n = (pts.col(i).array() / len.array()).matrix().normalized();
    sp_ = (pts.col(i).array() * len.array()).matrix() + margin * n;
    sv = set.SupportFunction(n, sp);
    EXPECT_NEAR(sv, n.dot(sp_), kTol);
    EXPECT_PRED3(AssertVectorEQ<3>, sp, sp_, kTol);
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
  const Real hlx{2.0}, radius{2.5}, margin{0.1};
  auto set{TypeParam(hlx, radius, margin)};

  EXPECT_EQ(set.GetInradius(), radius + margin);

  Mat3Xf pts;
  // Even number of points avoids zero x-component of normal.
  const int xy_size{18};
  UniformSpherePoints(pts, xy_size, 9);
  const int size = (dim == 2) ? xy_size : static_cast<int>(pts.cols());

  Real sv;
  Vecf<dim> sp, sp_, n;
  for (int i = 0; i < size; ++i) {
    n = pts.col(i).topRows<dim>().normalized();
    sp_ = (radius + margin) * n;
    sp_(0) += std::copysign(hlx, n(0));
    sv = set.SupportFunction(n, sp);
    EXPECT_NEAR(sv, n.dot(sp_), kTol);
    EXPECT_PRED3(AssertVectorEQ<dim>, sp, sp_, kTol);
  }
}

}  // namespace
