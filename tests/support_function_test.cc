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
#include "test_utils.h"

namespace {

using namespace dgd;
using dgd::test::kTol;
using dgd::test::kTolGrad;
using dgd::test::MatrixNear;
using dgd::test::UniformCirclePoints;
using dgd::test::UniformSpherePoints;
using dgd::test::VectorNear;

// ---------------------------------------------------------------------------
// Numerical differentiation utilities
// ---------------------------------------------------------------------------

template <int dim>
bool ComputeSupportPointJacobian(const ConvexSet<dim>* set, const Vecr<dim>& n,
                                 Matr<dim, dim>& d_sp_n,
                                 Matr<dim, dim>& d_sp_n_num) {
  SupportFunctionDerivatives<dim> deriv;
  set->SupportFunction(n.normalized(), deriv);
  if (!deriv.differentiable) return false;
  d_sp_n = deriv.d_sp_n;

  NumericalDifferentiator nd{};
  nd.Jacobian(
      [set](const Eigen::Ref<const VecXr>& x, Eigen::Ref<VecXr> y) -> void {
        Vecr<dim> nn = x.head<dim>().normalized(), sp;
        set->SupportFunction(nn, sp);
        y = sp;
      },
      n.normalized(), d_sp_n_num);
  return true;
}

// ---------------------------------------------------------------------------
// 2D support function tests
// ---------------------------------------------------------------------------

TEST(SupportFunctionTest, Ellipse) {
  const Real hlx = Real(3.0), hly = Real(2.0), margin = Real(0.0);
  const Ellipse set(hlx, hly, margin);

  EXPECT_EQ(set.inradius(), hly + margin);

  Real sv;
  Vec2r sp, spt, n;
  Matr<2, 2> d_sp_n, d_sp_n_num;
  const auto pts = UniformCirclePoints(16);
  const Vec2r len(hlx, hly);
  for (const auto& pt : pts) {
    n = (pt.array() / len.array()).matrix();
    n /= n.lpNorm<Eigen::Infinity>();
    spt = (pt.array() * len.array()).matrix() + margin * n;
    sv = set.SupportFunction(n, sp);
    EXPECT_NEAR(sv, n.dot(spt), kTol);
    ASSERT_PRED3(VectorNear<2>, sp, spt, kTol);
    if (ComputeSupportPointJacobian(&set, n, d_sp_n, d_sp_n_num)) {
      EXPECT_PRED3(MatrixNear<2>, d_sp_n, d_sp_n_num, kTolGrad);
    }
  }
}

TEST(SupportFunctionTest, Polygon) {
  Rng rng;
  rng.SetSeed();
  const int npts = 100;
  const Real margin = Real(0.0), len = Real(5.0);

  std::vector<Vec2r> pts(npts), vert;
  for (int i = 0; i < npts; ++i) {
    pts[i] << rng.Random(len), rng.Random(len);
  }
  GrahamScan(pts, vert);
  const Real inradius = ComputePolygonInradius(vert, Vec2r::Zero());

  auto set = Polygon(std::move(vert), inradius, margin);

  EXPECT_EQ(set.inradius(), inradius + margin);

  Real sv;
  Vec2r sp;
  const auto normals = UniformCirclePoints(16);
  for (const auto& n : normals) {
    sv = set.SupportFunction(n, sp);
    ASSERT_GE(sv, inradius);
  }
}

TEST(SupportFunctionTest, Rectangle) {
  const Real hlx = Real(3.0), hly = Real(2.0), margin = Real(0.0);
  const auto set = Rectangle(hlx, hly, margin);

  EXPECT_EQ(set.inradius(), hly + margin);

  Real sv;
  Vec2r sp, spt, n;
  for (int i = 0; i < 4; ++i) {
    n = Vec2r(std::pow(Real(-1.0), i % 2), std::pow(Real(-1.0), (i / 2) % 2));
    spt = Vec2r(hlx * n(0), hly * n(1)) + margin * n;
    sv = set.SupportFunction(n, sp);
    EXPECT_NEAR(sv, n.dot(spt), kTol);
    EXPECT_PRED3(VectorNear<2>, sp, spt, kTol);
  }
}

// ---------------------------------------------------------------------------
// 3D support function tests
// ---------------------------------------------------------------------------

TEST(SupportFunctionTest, Cone) {
  const Real ha = kPi / Real(6.0), radius = Real(1.0), margin = Real(0.0);
  const Real height = radius / std::tan(ha);
  const Real rho = height / (Real(1.0) + Real(1.0) / std::sin(ha));
  const auto set = Cone(radius, height, margin);

  EXPECT_NEAR(set.inradius(), rho + margin, kTol);
  EXPECT_NEAR(set.offset(), rho, kTol);

  Real sv;
  Vec3r sp, spt, n;
  Matr<3, 3> d_sp_n, d_sp_n_num;
  const auto pts = UniformSpherePoints(16, 9, Real(1e-1));
  for (const auto& pt : pts) {
    n = pt;
    n /= n.lpNorm<Eigen::Infinity>();
    if (n.head<2>().norm() * std::tan(ha) < n(2)) {
      spt = Vec3r(Real(0.0), Real(0.0), height - rho);
    } else {
      spt.head<2>() = radius * n.head<2>().normalized();
      spt(2) = -rho;
    }
    spt += margin * n;
    sv = set.SupportFunction(n, sp);
    EXPECT_NEAR(sv, n.dot(spt), kTol);
    ASSERT_PRED3(VectorNear<3>, sp, spt, kTol);
    if (ComputeSupportPointJacobian(&set, n, d_sp_n, d_sp_n_num)) {
      EXPECT_PRED3(MatrixNear<3>, d_sp_n, d_sp_n_num, kTolGrad);
    }
  }
}

TEST(SupportFunctionTest, Cuboid) {
  const Real hlx = Real(3.0), hly = Real(2.0), hlz = Real(1.5);
  const Real margin = Real(0.0);
  const auto set = Cuboid(hlx, hly, hlz, margin);

  EXPECT_EQ(set.inradius(), hlz + margin);

  Real sv;
  Vec3r sp, spt, n;
  for (int i = 0; i < 8; ++i) {
    n = Vec3r(Real(std::pow(Real(-1.0), i % 2)),
              Real(std::pow(Real(-1.0), (i / 2) % 2)),
              Real(std::pow(Real(-1.0), (i / 4) % 2)));
    spt = Vec3r(hlx * n(0), hly * n(1), hlz * n(2)) + margin * n;
    sv = set.SupportFunction(n, sp);
    EXPECT_NEAR(sv, n.dot(spt), kTol);
    ASSERT_PRED3(VectorNear<3>, sp, spt, kTol);
  }
}

TEST(SupportFunctionTest, Cylinder) {
  const Real hlx = Real(2.0), radius = Real(2.5), margin = Real(0.0);
  const auto set = Cylinder(hlx, radius, margin);

  EXPECT_EQ(set.inradius(), hlx + margin);

  Real sv;
  Vec3r sp, spt, n;
  Matr<3, 3> d_sp_n, d_sp_n_num;
  const auto pts = UniformCirclePoints(16);
  for (int i = 0; i < 2; ++i) {
    for (const auto& pt : pts) {
      n(0) = Real(std::pow(Real(-1.0), i));
      n.tail<2>() = pt;
      spt(0) = hlx * Real(std::pow(Real(-1.0), i));
      spt.tail<2>() = radius * pt;
      spt += margin * n;
      sv = set.SupportFunction(n, sp);
      EXPECT_NEAR(sv, n.dot(spt), kTol);
      ASSERT_PRED3(VectorNear<3>, sp, spt, kTol);
      if (ComputeSupportPointJacobian(&set, n, d_sp_n, d_sp_n_num)) {
        EXPECT_PRED3(MatrixNear<3>, d_sp_n, d_sp_n_num, kTolGrad);
      }
    }
  }
}

TEST(SupportFunctionTest, Ellipsoid) {
  const Real hlx = Real(3.0), hly = Real(2.0), hlz = Real(1.5);
  const Real margin = Real(0.0);
  const auto set = Ellipsoid(hlx, hly, hlz, margin);

  EXPECT_EQ(set.inradius(), hlz + margin);

  Real sv;
  Vec3r sp, spt, n;
  Matr<3, 3> d_sp_n, d_sp_n_num;
  const auto pts = UniformSpherePoints(16, 9);
  const Vec3r len(hlx, hly, hlz);
  for (const auto& pt : pts) {
    n = (pt.array() / len.array()).matrix();
    n /= n.lpNorm<Eigen::Infinity>();
    spt = (pt.array() * len.array()).matrix() + margin * n;
    sv = set.SupportFunction(n, sp);
    EXPECT_NEAR(sv, n.dot(spt), kTol);
    EXPECT_PRED3(VectorNear<3>, sp, spt, kTol);
    if (ComputeSupportPointJacobian(&set, n, d_sp_n, d_sp_n_num)) {
      EXPECT_PRED3(MatrixNear<3>, d_sp_n, d_sp_n_num, kTolGrad);
    }
  }
}

TEST(SupportFunctionTest, Frustum) {
  const Real margin = Real(0.0);
  std::vector<Frustum> sets;
  std::vector<Real> rb(8), rt(8), h(8);

  Real radius = Real(1.0), height = Real(2.0);
  rb[0] = radius;
  rt[0] = radius;
  h[0] = height;
  sets.emplace_back(radius, radius, height, margin);
  EXPECT_NEAR(sets[0].inradius(), radius + margin, kTol);
  EXPECT_NEAR(sets[0].offset(), radius, kTol);

  height = Real(0.5);
  rb[1] = radius;
  rt[1] = radius;
  h[1] = height;
  sets.emplace_back(radius, radius, height, margin);
  EXPECT_NEAR(sets[1].inradius(), height / Real(2.0) + margin, kTol);
  EXPECT_NEAR(sets[1].offset(), height / Real(2.0), kTol);

  const Real ha = kPi / Real(6.0);
  height = radius / std::tan(ha);
  const Real rho = height / (Real(1.0) + Real(1.0) / std::sin(ha));
  rb[2] = radius;
  rt[2] = Real(0.0);
  h[2] = height;
  sets.emplace_back(radius, 0.0, height, margin);
  EXPECT_NEAR(sets[2].inradius(), rho + margin, kTol);
  EXPECT_NEAR(sets[2].offset(), rho, kTol);

  rb[3] = Real(0.0);
  rt[3] = radius;
  h[3] = height;
  sets.emplace_back(Real(0.0), radius, height, margin);
  EXPECT_NEAR(sets[3].inradius(), rho + margin, kTol);
  EXPECT_NEAR(sets[3].offset(), height - rho, kTol);

  const Real height_cone = radius / std::tan(ha);
  height = height_cone / Real(2.0) + rho;
  const Real small_radius = radius * (Real(1.0) - height / height_cone);
  rb[4] = radius;
  rt[4] = small_radius;
  h[4] = height;
  sets.emplace_back(radius, small_radius, height, margin);
  EXPECT_NEAR(sets[4].inradius(), rho + margin, kTol);
  EXPECT_NEAR(sets[4].offset(), rho, kTol);

  rb[5] = small_radius;
  rt[5] = radius;
  h[5] = height;
  sets.emplace_back(small_radius, radius, height, margin);
  EXPECT_NEAR(sets[5].inradius(), rho + margin, kTol);
  EXPECT_NEAR(sets[5].offset(), height - rho, kTol);

  height = rho;
  const Real sr2 = radius / height_cone * height;
  rb[6] = radius;
  rt[6] = sr2;
  h[6] = height;
  sets.emplace_back(radius, sr2, height, margin);
  EXPECT_NEAR(sets[6].inradius(), height / Real(2.0) + margin, kTol);
  EXPECT_NEAR(sets[6].offset(), height / Real(2.0), kTol);

  rb[7] = sr2;
  rt[7] = radius;
  h[7] = height;
  sets.emplace_back(sr2, radius, height, margin);
  EXPECT_NEAR(sets[7].inradius(), height / Real(2.0) + margin, kTol);
  EXPECT_NEAR(sets[7].offset(), height / Real(2.0), kTol);

  Real sv, tha, offset;
  Vec3r sp, spt, n;
  Matr<3, 3> d_sp_n, d_sp_n_num;
  const auto pts = UniformSpherePoints(16, 10, Real(1e-1));
  for (int k = 0; k < static_cast<int>(sets.size()); ++k) {
    const auto& set = sets[k];
    for (const auto& pt : pts) {
      n = pt;
      n /= n.lpNorm<Eigen::Infinity>();
      tha = (rb[k] - rt[k]) / h[k];
      offset = set.offset();
      if (n.head<2>().norm() * tha < n(2)) {
        spt.head<2>() = rt[k] * n.head<2>().normalized();
        spt(2) = h[k] - offset;
      } else {
        spt.head<2>() = rb[k] * n.head<2>().normalized();
        spt(2) = -offset;
      }
      spt += margin * n;
      sv = set.SupportFunction(n, sp);
      EXPECT_NEAR(sv, n.dot(spt), kTol);
      ASSERT_PRED3(VectorNear<3>, sp, spt, kTol);
      if (ComputeSupportPointJacobian(&set, n, d_sp_n, d_sp_n_num)) {
        EXPECT_PRED3(MatrixNear<3>, d_sp_n, d_sp_n_num, kTolGrad);
      }
    }
  }
}

TEST(SupportFunctionTest, Mesh) {
  if (typeid(Real) == typeid(float)) GTEST_SKIP();

  Rng rng;
  rng.SetSeed();
  const int nruns = 10, npts = 400;
  const Real inradius = Real(0.25), margin = Real(0.0);

  MeshLoader ml{};
  std::vector<Vec3r> pts(npts), vert;
  std::vector<int> graph;
  const auto normals = UniformSpherePoints(100, 10);
  Vec3r sp, spt, n;
  Real sv, svt;
  for (int i = 0; i < nruns; ++i) {
    for (int j = 0; j < npts; ++j) {
      pts[j] = rng.RandomUnitVector<3>();
    }
    ml.ProcessPoints(pts);
    const bool valid = ml.MakeVertexGraph(vert, graph);
    ASSERT_TRUE(valid);

    auto polytope = Polytope(vert, inradius, margin);
    Mesh mesh(std::move(vert), std::move(graph), inradius, margin);

    for (const auto& nn : normals) {
      n = nn;
      n /= n.lpNorm<Eigen::Infinity>();
      svt = polytope.SupportFunction(n, spt);
      sv = mesh.SupportFunction(n, sp);
      ASSERT_NEAR(sv, svt, kTol);
    }
  }
}

TEST(SupportFunctionTest, Polytope) {
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
  const Real inradius = ml.ComputeInradius(interior_point);

  auto set = Polytope(std::move(vert), inradius, margin);

  EXPECT_LE(set.inradius(), inradius + margin);

  Real sv;
  Vec3r sp;
  const auto normals = UniformSpherePoints(16, 9);
  for (const auto& n : normals) {
    sv = set.SupportFunction(n, sp);
    ASSERT_GE(sv, inradius);
  }
}

// ---------------------------------------------------------------------------
// Typed tests for CapsuleImpl and SphereImpl support functions
// ---------------------------------------------------------------------------

template <int dim>
void CapsuleImplSupportFunctionTest() {
  const Real hlx = Real(2.0), radius = Real(2.5), margin = Real(0.25);
  const auto set = CapsuleImpl<dim>(hlx, radius, margin);

  EXPECT_EQ(set.inradius(), radius + margin);

  const auto pts = UniformSpherePoints(17, 9);
  const int size = (dim == 2) ? 17 : static_cast<int>(pts.size());

  Real sv;
  Vecr<dim> sp, spt, n;
  Matr<dim, dim> d_sp_n, d_sp_n_num;
  for (int i = 0; i < size; ++i) {
    n = pts[i].head<dim>().normalized();
    spt = (radius + margin) * n;
    spt(0) += std::copysign(hlx, n(0));
    sv = set.SupportFunction(n, sp);
    EXPECT_NEAR(sv, n.dot(spt), kTol);
    ASSERT_PRED3(VectorNear<dim>, sp, spt, kTol);
    if (ComputeSupportPointJacobian(&set, n, d_sp_n, d_sp_n_num)) {
      EXPECT_PRED3(MatrixNear<dim>, d_sp_n, d_sp_n_num, kTolGrad);
    }
  }
}

TEST(SupportFunctionTest, Stadium) { CapsuleImplSupportFunctionTest<2>(); }

TEST(SupportFunctionTest, Capsule) { CapsuleImplSupportFunctionTest<3>(); }

template <int dim>
void SphereImplSupportFunctionTest() {
  const Real radius = Real(0.25);
  const auto set = SphereImpl<dim>(radius);

  EXPECT_EQ(set.inradius(), radius);

  Vecr<dim> sp;
  Matr<dim, dim> d_sp_n, d_sp_n_num;
  const Vecr<dim> n = Vecr<dim>::UnitX();
  const Real sv = set.SupportFunction(n, sp);
  EXPECT_NEAR(sv, radius, kTol);
  EXPECT_PRED3(VectorNear<dim>, sp, radius * Vecr<dim>::UnitX(), kTol);
  if (ComputeSupportPointJacobian(&set, n, d_sp_n, d_sp_n_num)) {
    EXPECT_PRED3(MatrixNear<dim>, d_sp_n, d_sp_n_num, kTolGrad);
  }
}

TEST(SupportFunctionTest, Circle) { SphereImplSupportFunctionTest<2>(); }

TEST(SupportFunctionTest, Sphere) { SphereImplSupportFunctionTest<3>(); }

}  // namespace
