#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <type_traits>
#include <utility>
#include <vector>

#include "dgd/data_types.h"
#include "dgd/geometry/geometry_2d.h"
#include "dgd/geometry/geometry_3d.h"
#include "dgd/graham_scan.h"
#include "dgd/settings.h"
#include "dgd/solvers/minimum_distance_impl.h"
#include "dgd/utils/numerical_differentiation.h"
#include "dgd/utils/random.h"
#include "dgd/utils/transformations.h"
#include "test_utils.h"

namespace {

using namespace dgd;
using dgd::test::kTolHess;
using dgd::test::MatrixNear;
using dgd::test::UniformCirclePoints;
using dgd::test::UniformSpherePoints;
using dgd::test::VectorNear;

const Real kStepSize = std::pow(kEps, Real(0.25));
const Real kRelTol = Real(1.0) + std::pow(kEps, Real(0.75));
const Real kDistTol = std::pow(kEps, Real(0.75));

// ---------------------------------------------------------------------------
// Math utilities
// ---------------------------------------------------------------------------

/// Tolerance scaling for distances.
inline Real DistTolerance(Real d, Real p) {
  return kDistTol * (std::max(Real(1.0), d) + Real(2.0)) +
         Real(2.0) * kRelTol * p;
}

/// Tolerance scaling for witness points.
inline Real WitnessTolerance(Real d, Real p) {
  return std::sqrt(DistTolerance(d, p));
}

inline Real LogSample(Real low, Real high, Rng& rng) {
  return std::exp2(rng.Random(std::log2(low), std::log2(high)));
}

// ---------------------------------------------------------------------------
// Numerical differentiation utilities
// ---------------------------------------------------------------------------

template <int dim>
bool ComputeProjectionDerivative(const ConvexSet<dim>* set, const Vecr<dim>& p,
                                 const Vecr<dim>& pi, Matr<dim, dim>& d_pi_p,
                                 Matr<dim, dim>& d_pi_p_num,
                                 BasePointHint<dim>* hint = nullptr) {
  bool differentiable = set->ProjectionDerivative(p, pi, d_pi_p, hint);
  if (!differentiable) return false;

  const SphereImpl<dim> pt(Real(0.0));
  const Transformr<dim> tf = Transformr<dim>::Identity();
  Transformr<dim> tf_pt = Transformr<dim>::Identity();

  Settings settings{};
  settings.rel_tol = kRelTol;
  DistanceOutput<dim> out;

  // A uniform step for every coordinate avoids accuracy issues when some
  // components of p are near zero. Scaling by ||p|| keeps the solver-noise
  // contribution to the Hessian error at most (d/||p||)^2 * kTolGrad <
  // kTolGrad.
  const Real h = kStepSize * std::max(Real(1.0), p.norm());
  NumericalDifferentiator nd(h, false);
  nd.Hessian(
      [&](const VecXr& x) -> Real {
        Affine(tf_pt) = x.head<dim>();
        const Real d =
            MinimumDistanceTpl(&pt, tf_pt, set, tf, settings, out, kDistTol);
        return Real(0.5) * d * d;
      },
      p, d_pi_p_num);
  d_pi_p_num = Matr<dim, dim>::Identity() - d_pi_p_num;
  return true;
}

// ---------------------------------------------------------------------------
// Minimum distance tests
// ---------------------------------------------------------------------------

TEST(MinimumDistanceTest, CircleCircle) {
  const Real r1 = Real(1.5), r2 = Real(1.0);
  const Real p = Real(6.0);
  const Real d_true = p - r1 - r2;

  const Circle set1(r1);
  const Circle set2(r2);
  const Transform2r tf1 = Transform2r::Identity();
  Transform2r tf2 = Transform2r::Identity();
  Affine(tf2) = Vec2r(p, Real(0.0));

  Settings settings{};
  settings.rel_tol = kRelTol;
  DistanceOutput<2> out;
  const Real d =
      MinimumDistanceTpl(&set1, tf1, &set2, tf2, settings, out, kDistTol);

  const Real dtol = DistTolerance(d_true, p);
  const Real wtol = WitnessTolerance(d_true, p);

  ASSERT_EQ(out.status, SolutionStatus::Optimal);
  EXPECT_NEAR(d, d_true, dtol);
  EXPECT_NEAR(out.min_dist, d_true, dtol);

  EXPECT_PRED3(VectorNear<2>, out.normal, Vec2r::UnitX(), wtol);
  EXPECT_PRED3(VectorNear<2>, out.z1, Vec2r(r1, Real(0.0)), wtol);
  EXPECT_PRED3(VectorNear<2>, out.z2, Vec2r(p - r2, Real(0.0)), wtol);
}

TEST(MinimumDistanceTest, SphereSphere) {
  const Real r1 = Real(1.0), r2 = Real(2.0);
  const Real p = Real(7.0);
  const Real d_true = p - r1 - r2;

  const Sphere set1(r1);
  const Sphere set2(r2);
  const Transform3r tf1 = Transform3r::Identity();
  Transform3r tf2 = Transform3r::Identity();
  Affine(tf2) = p * Vec3r(Real(1.0), Real(0.5), Real(2.0)).normalized();

  Settings settings{};
  settings.rel_tol = kRelTol;
  DistanceOutput<3> out;
  const Real d =
      MinimumDistanceTpl(&set1, tf1, &set2, tf2, settings, out, kDistTol);

  const Real dtol = DistTolerance(d_true, p);
  const Real wtol = WitnessTolerance(d_true, p);

  ASSERT_EQ(out.status, SolutionStatus::Optimal);
  EXPECT_NEAR(d, d_true, dtol);
  EXPECT_NEAR(out.min_dist, d_true, dtol);

  const Vec3r n_true = Affine(tf2).normalized();
  EXPECT_PRED3(VectorNear<3>, out.normal, n_true, wtol);
  EXPECT_PRED3(VectorNear<3>, out.z1, r1 * n_true, wtol);
  EXPECT_PRED3(VectorNear<3>, out.z2, (p - r2) * n_true, wtol);
}

TEST(MinimumDistanceTest, CuboidSphere) {
  const Real hx = Real(2.0), hy = Real(1.5), hz = Real(1.0);
  const Real r = Real(0.5);
  const Vec3r h(hx, hy, hz);

  const Cuboid set1(hx, hy, hz);
  const Sphere set2(r);
  const Transform3r tf1 = Transform3r::Identity();
  Transform3r tf2 = Transform3r::Identity();

  Settings settings{};
  settings.rel_tol = kRelTol;
  DistanceOutput<3> out;

  Rng rng;
  rng.SetSeed(42);
  const auto pts = UniformSpherePoints(4, 10, Real(1e-2));

  for (const auto& pt : pts) {
    const Real p = h.norm() + r + rng.Random(Real(1.0), Real(2.0));
    const Vec3r center = p * pt;
    Affine(tf2) = center;

    const Real d =
        MinimumDistanceTpl(&set1, tf1, &set2, tf2, settings, out, kDistTol);

    ASSERT_EQ(out.status, SolutionStatus::Optimal);

    const Vec3r z1_true = center.cwiseMax(-h).cwiseMin(h);
    const Vec3r diff = center - z1_true;
    const Real dist_true = diff.norm() - r;
    const Vec3r n_true = diff.normalized();
    const Vec3r z2_true = center - r * n_true;

    const Real tol = DistTolerance(dist_true, p);
    const Real wtol = WitnessTolerance(dist_true, p);
    EXPECT_NEAR(d, dist_true, tol);
    EXPECT_PRED3(VectorNear<3>, out.z1, z1_true, wtol);
    EXPECT_PRED3(VectorNear<3>, out.normal, n_true, wtol);
    EXPECT_PRED3(VectorNear<3>, out.z2, z2_true, wtol);
  }
}

TEST(MinimumDistanceTest, CircleHalfspace) {
  const Real r = Real(1.5);
  const Real m_hs = Real(0.4);
  const Real cx = Real(3.0);

  const Circle set1(r);
  const Halfspace<2> set2(m_hs);
  Transform2r tf1 = Transform2r::Identity();
  const Transform2r tf2 = Transform2r::Identity();

  Settings settings{};
  settings.rel_tol = kRelTol;
  DistanceOutput<2> out;

  const Real cy_sep = Real(5.0);
  Affine(tf1) = Vec2r(cx, cy_sep);
  const Real d_sep = cy_sep - r - m_hs;

  const Real d =
      MinimumDistanceHalfspaceTpl(&set1, tf1, &set2, tf2, settings, out);

  const Real dtol = DistTolerance(d_sep, cy_sep);
  const Real wtol = WitnessTolerance(d_sep, cy_sep);

  ASSERT_EQ(out.status, SolutionStatus::Optimal);
  EXPECT_NEAR(d, d_sep, dtol);
  EXPECT_NEAR(out.min_dist, d_sep, dtol);
  EXPECT_PRED3(VectorNear<2>, out.normal, -Vec2r::UnitY(), wtol);
  EXPECT_PRED3(VectorNear<2>, out.z1, Vec2r(cx, cy_sep - r), wtol);
  EXPECT_PRED3(VectorNear<2>, out.z2, Vec2r(cx, m_hs), wtol);

  const Real cy_int = Real(0.5) * m_hs;
  Affine(tf1) = Vec2r(cx, cy_int);

  MinimumDistanceHalfspaceTpl(&set1, tf1, &set2, tf2, settings, out);

  ASSERT_EQ(out.status, SolutionStatus::Optimal);
  EXPECT_NEAR(out.min_dist, Real(0.0), kEps);
}

TEST(MinimumDistanceTest, CuboidHalfspace) {
  const Real hx = Real(1.0), hy = Real(2.0), hz = Real(1.5);
  const Real m_hs = Real(0.3);
  const Vec3r h(hx, hy, hz);

  const Cuboid set1(hx, hy, hz);
  const Halfspace<3> set2(m_hs);
  const Transform3r tf2 = Transform3r::Identity();

  Settings settings{};
  settings.rel_tol = kRelTol;
  DistanceOutput<3> out;

  const Vec3r n_world = -Vec3r::UnitZ();

  Rng rng;
  rng.SetSeed(7);
  const int nsamples = 20;

  for (int i = 0; i < nsamples; ++i) {
    Transform3r tf1 = Transform3r::Identity();
    Linear(tf1) = rng.RandomRotation<3>();

    const Vec3r n_body = Linear(tf1).transpose() * n_world;
    Vec3r z1_rel;
    for (int k = 0; k < 3; ++k) z1_rel(k) = std::copysign(h(k), n_body(k));
    z1_rel = Linear(tf1) * z1_rel;

    const Real delta = rng.Random(Real(0.5), Real(1.5));
    Affine(tf1)(0) = rng.Random(Real(-3.0), Real(3.0));
    Affine(tf1)(1) = rng.Random(Real(-3.0), Real(3.0));
    Affine(tf1)(2) = m_hs + delta - z1_rel(2);

    const Real d =
        MinimumDistanceHalfspaceTpl(&set1, tf1, &set2, tf2, settings, out);

    const Vec3r z1_true = Affine(tf1) + z1_rel;
    const Vec3r z2_true = Vec3r(z1_true(0), z1_true(1), m_hs);
    const Real d_true = delta;

    const Real dtol = DistTolerance(d_true, Affine(tf1).norm());
    const Real wtol = WitnessTolerance(d_true, Affine(tf1).norm());

    ASSERT_EQ(out.status, SolutionStatus::Optimal);
    EXPECT_NEAR(d, d_true, dtol);
    EXPECT_PRED3(VectorNear<3>, out.z1, z1_true, wtol);
    EXPECT_PRED3(VectorNear<3>, out.z2, z2_true, wtol);
    EXPECT_PRED3(VectorNear<3>, out.normal, n_world, wtol);
  }
}

// ---------------------------------------------------------------------------
// Projection tests
// ---------------------------------------------------------------------------

TEST(ProjectionTest, Circle) {
  const Real r = Real(2.0);

  const Circle set1(r);
  const Circle pt(Real(0.0));
  const Transform2r tf1 = Transform2r::Identity();
  Transform2r tf2 = Transform2r::Identity();

  Settings settings{};
  settings.rel_tol = kRelTol;
  DistanceOutput<2> out;

  Rng rng;
  rng.SetSeed(13);
  const auto pts = UniformCirclePoints(20);

  for (const auto& n : pts) {
    const Real p = r + rng.Random(Real(0.5), Real(2.0));
    Affine(tf2) = p * n;

    const Real d_true = p - r;
    const Real d =
        MinimumDistanceTpl(&set1, tf1, &pt, tf2, settings, out, kDistTol);

    const Real dtol = DistTolerance(d_true, p);
    const Real wtol = WitnessTolerance(d_true, p);

    ASSERT_EQ(out.status, SolutionStatus::Optimal);
    EXPECT_NEAR(d, d_true, dtol);
    EXPECT_PRED3(VectorNear<2>, out.z1, r * n, wtol);
    EXPECT_PRED3(VectorNear<2>, out.z2, p * n, wtol);
    EXPECT_PRED3(VectorNear<2>, out.normal, n, wtol);
  }
}

TEST(ProjectionTest, Cuboid) {
  const Real hx = Real(2.0), hy = Real(1.5), hz = Real(1.0);
  const Vec3r h(hx, hy, hz);

  const Cuboid set1(hx, hy, hz);
  const Sphere pt(Real(0.0));
  const Transform3r tf1 = Transform3r::Identity();
  Transform3r tf2 = Transform3r::Identity();

  Settings settings{};
  settings.rel_tol = kRelTol;
  DistanceOutput<3> out;

  Rng rng;
  rng.SetSeed(17);
  const auto pts = UniformSpherePoints(8, 20, Real(1e-2));

  for (const auto& n : pts) {
    const Real p = h.norm() + rng.Random(Real(1.0), Real(3.0));
    const Vec3r p2 = p * n;
    Affine(tf2) = p2;

    const Real d =
        MinimumDistanceTpl(&set1, tf1, &pt, tf2, settings, out, kDistTol);

    const Vec3r z1_true = p2.cwiseMax(-h).cwiseMin(h);
    const Vec3r diff = p2 - z1_true;
    const Real d_true = diff.norm();
    const Vec3r n_true = diff.normalized();

    const Real tol = DistTolerance(d_true, p);
    const Real wtol = WitnessTolerance(d_true, p);

    ASSERT_EQ(out.status, SolutionStatus::Optimal);
    EXPECT_NEAR(d, d_true, tol);
    EXPECT_PRED3(VectorNear<3>, out.z1, z1_true, wtol);
    EXPECT_PRED3(VectorNear<3>, out.z2, p2, wtol);
    EXPECT_PRED3(VectorNear<3>, out.normal, n_true, wtol);
  }
}

// ---------------------------------------------------------------------------
// 2D projection derivative tests
// ---------------------------------------------------------------------------

TEST(ProjectionDerivativeTest, Ellipse) {
  const Real hlx = Real(3.0), hly = Real(2.0);
  const Vec2r qe(Real(1.0) / (hlx * hlx), Real(1.0) / (hly * hly));

  const auto dirs = UniformCirclePoints(16);

  Rng rng;
  rng.SetSeed(31);

  const Real margins[2] = {Real(0.0), Real(0.5)};
  Matr<2, 2> d_pi_p, d_pi_p_num;

  for (Real m : margins) {
    const Ellipse set(hlx, hly, m);
    for (const auto& u : dirs) {
      Vec2r pi(hlx * u(0), hly * u(1));
      const Vec2r n = Vec2r(pi(0) * qe(0), pi(1) * qe(1)).normalized();
      pi += m * n;
      const Vec2r p = pi + LogSample(Real(5e-1), Real(5.0), rng) * n;

      const bool diff =
          ComputeProjectionDerivative(&set, p, pi, d_pi_p, d_pi_p_num);
      ASSERT_TRUE(diff);

      EXPECT_PRED3(MatrixNear<2>, d_pi_p, d_pi_p_num, kTolHess);
    }
  }
}

TEST(ProjectionDerivativeTest, Polygon) {
  const int nv = 6;
  const Real radius = Real(1.0);
  std::vector<Vec2r> vert(nv);
  const Real dtheta = Real(2.0) * kPi / static_cast<Real>(nv);
  for (int i = 0; i < nv; ++i) {
    const Real theta = dtheta * static_cast<Real>(i);
    vert[i] = Vec2r(radius * std::cos(theta), radius * std::sin(theta));
  }
  const Real inradius = ComputePolygonInradius(vert, Vec2r::Zero());

  Matr<2, 2> d_pi_p, d_pi_p_num;
  Rng rng;
  rng.SetSeed(101);

  const Real margins[2] = {Real(0.0), Real(0.3)};

  for (Real m : margins) {
    const Polygon set(vert, inradius, m);

    // Surface 1: edges.
    for (int i = 0; i < nv; ++i) {
      const Vec2r& v0 = vert[i];
      const Vec2r& v1 = vert[(i + 1) % nv];
      const Real f = rng.CoinFlip() * rng.Random(Real(0.05), Real(0.95));
      const Vec2r n = Vec2r(v1(1) - v0(1), v0(0) - v1(0)).normalized();
      const Vec2r pi = v1 + f * (v0 - v1) + m * n;
      const Vec2r p = pi + LogSample(Real(5e-1), Real(5.0), rng) * n;

      BasePointHint<2> hint;
      hint.idx = Vec2i(i, (i + 1) % nv);
      hint.bc = Vec2r(f, Real(1.0) - f);

      const bool diff = ComputeProjectionDerivative<2>(&set, p, pi, d_pi_p,
                                                       d_pi_p_num, &hint);
      if (f == Real(0.0)) {
        ASSERT_FALSE(diff);
        continue;
      }
      ASSERT_TRUE(diff);
      EXPECT_PRED3(MatrixNear<2>, d_pi_p, d_pi_p_num, kTolHess);
    }

    // Surface 2: vertices.
    for (int i = 0; i < nv; ++i) {
      const Real theta =
          dtheta * static_cast<Real>(i) + Real(0.125) * kPi * rng.Random();
      const Vec2r n = Vec2r(std::cos(theta), std::sin(theta));
      const Vec2r pi = vert[i] + m * n;
      const Vec2r p = pi + LogSample(Real(5e-1), Real(5.0), rng) * n;

      const bool diff =
          ComputeProjectionDerivative<2>(&set, p, pi, d_pi_p, d_pi_p_num);
      ASSERT_TRUE(diff);
      EXPECT_PRED3(MatrixNear<2>, d_pi_p, d_pi_p_num, kTolHess);
    }
  }
}

TEST(ProjectionDerivativeTest, Rectangle) {
  const Real hlx = Real(3.0), hly = Real(2.0);

  Matr<2, 2> d_pi_p, d_pi_p_num;
  Rng rng;
  rng.SetSeed(67);

  const Real margins[2] = {Real(0.0), Real(0.5)};

  for (Real m : margins) {
    const Rectangle set(hlx, hly, m);

    // Surface 1: left/right edges.
    for (Real sign : {Real(-1.0), Real(1.0)}) {
      const Vec2r n(sign, Real(0.0));
      for (int j = 0; j < 4; ++j) {
        const Vec2r pi(sign * (hlx + m), rng.Random(hly * Real(0.8)));
        const Vec2r p = pi + LogSample(Real(5e-1), Real(5.0), rng) * n;

        const bool diff =
            ComputeProjectionDerivative<2>(&set, p, pi, d_pi_p, d_pi_p_num);
        ASSERT_TRUE(diff);
        EXPECT_PRED3(MatrixNear<2>, d_pi_p, d_pi_p_num, kTolHess);
      }
    }

    // Surface 2: top/bottom edges.
    for (Real sign : {Real(-1.0), Real(1.0)}) {
      const Vec2r n(Real(0.0), sign);
      for (int j = 0; j < 4; ++j) {
        const Vec2r pi(rng.Random(hlx * Real(0.8)), sign * (hly + m));
        const Vec2r p = pi + LogSample(Real(5e-1), Real(5.0), rng) * n;

        const bool diff =
            ComputeProjectionDerivative<2>(&set, p, pi, d_pi_p, d_pi_p_num);
        ASSERT_TRUE(diff);
        EXPECT_PRED3(MatrixNear<2>, d_pi_p, d_pi_p_num, kTolHess);
      }
    }

    // Surface 3: vertices.
    for (Real sx : {Real(-1.0), Real(1.0)}) {
      for (Real sy : {Real(-1.0), Real(1.0)}) {
        const Vec2r v(sx * hlx, sy * hly);
        for (int j = 0; j < 6; ++j) {
          const Real theta = rng.Random(Real(0.1), kPi / Real(2.0) - Real(0.1));
          const Vec2r n(sx * std::cos(theta), sy * std::sin(theta));
          const Vec2r pi = v + m * n;
          const Vec2r p = pi + LogSample(Real(5e-1), Real(5.0), rng) * n;

          const bool diff =
              ComputeProjectionDerivative<2>(&set, p, pi, d_pi_p, d_pi_p_num);
          ASSERT_TRUE(diff);
          EXPECT_PRED3(MatrixNear<2>, d_pi_p, d_pi_p_num, kTolHess);
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// 3D projection derivative tests
// ---------------------------------------------------------------------------

TEST(ProjectionDerivativeTest, Ellipsoid) {
  const Real hlx = Real(3.0), hly = Real(2.0), hlz = Real(1.5);
  const Vec3r qe(Real(1.0) / (hlx * hlx), Real(1.0) / (hly * hly),
                 Real(1.0) / (hlz * hlz));

  const auto dirs = UniformSpherePoints(8, 10, Real(1e-2));

  Rng rng;
  rng.SetSeed(37);

  const Real margins[2] = {Real(0.0), Real(0.5)};
  Matr<3, 3> d_pi_p, d_pi_p_num;

  for (Real m : margins) {
    const Ellipsoid set(hlx, hly, hlz, m);
    for (const auto& u : dirs) {
      Vec3r pi(hlx * u(0), hly * u(1), hlz * u(2));
      const Vec3r n =
          Vec3r(pi(0) * qe(0), pi(1) * qe(1), pi(2) * qe(2)).normalized();
      pi += m * n;
      const Vec3r p = pi + LogSample(Real(5e-1), Real(5.0), rng) * n;

      const bool diff =
          ComputeProjectionDerivative(&set, p, pi, d_pi_p, d_pi_p_num);
      ASSERT_TRUE(diff);

      EXPECT_PRED3(MatrixNear<3>, d_pi_p, d_pi_p_num, kTolHess);
    }
  }
}

TEST(ProjectionDerivativeTest, Cuboid) {
  const Real hlx = Real(3.0), hly = Real(2.0), hlz = Real(1.5);

  Matr<3, 3> d_pi_p, d_pi_p_num;
  Rng rng;
  rng.SetSeed(79);

  const Real margins[2] = {Real(0.0), Real(0.5)};

  for (Real m : margins) {
    const Cuboid set(hlx, hly, hlz, m);
    const Vec3r h(hlx, hly, hlz);

    // Surface 1: faces.
    for (int axis = 0; axis < 3; ++axis) {
      for (Real sign : {Real(-1.0), Real(1.0)}) {
        Vec3r n = Vec3r::Zero();
        n(axis) = sign;
        for (int j = 0; j < 6; ++j) {
          Vec3r pi = Vec3r::Zero();
          pi(axis) = sign * (h(axis) + m);
          pi((axis + 1) % 3) = rng.Random(h((axis + 1) % 3) * Real(0.7));
          pi((axis + 2) % 3) = rng.Random(h((axis + 2) % 3) * Real(0.7));
          const Vec3r p = pi + LogSample(Real(5e-1), Real(5.0), rng) * n;

          const bool diff =
              ComputeProjectionDerivative<3>(&set, p, pi, d_pi_p, d_pi_p_num);
          ASSERT_TRUE(diff);
          EXPECT_PRED3(MatrixNear<3>, d_pi_p, d_pi_p_num, kTolHess);
        }
      }
    }

    // Surface 2: edges.
    for (int i0 = 0; i0 < 3; ++i0) {
      for (int i1 = i0 + 1; i1 < 3; ++i1) {
        const int i2 = 3 - i0 - i1;
        for (Real s0 : {Real(-1.0), Real(1.0)}) {
          for (Real s1 : {Real(-1.0), Real(1.0)}) {
            Vec3r n = Vec3r::Zero();
            n(i0) = s0;
            n(i1) = s1;
            n.normalize();
            Vec3r v = Vec3r::Zero();
            v(i0) = s0 * h(i0);
            v(i1) = s1 * h(i1);
            v(i2) = rng.Random(h(i2) * Real(0.8));
            const Vec3r pi = v + m * n;
            const Vec3r p = pi + LogSample(Real(5e-1), Real(5.0), rng) * n;

            const bool diff =
                ComputeProjectionDerivative<3>(&set, p, pi, d_pi_p, d_pi_p_num);
            ASSERT_TRUE(diff);
            EXPECT_PRED3(MatrixNear<3>, d_pi_p, d_pi_p_num, kTolHess);
          }
        }
      }
    }

    // Surface 3: vertices.
    for (Real sx : {Real(-1.0), Real(1.0)}) {
      for (Real sy : {Real(-1.0), Real(1.0)}) {
        for (Real sz : {Real(-1.0), Real(1.0)}) {
          const Vec3r v(sx * hlx, sy * hly, sz * hlz);
          const auto dirs = UniformSpherePoints(4, 5, Real(1e-1));
          for (const auto& n : dirs) {
            if (std::min({sx * n(0), sy * n(1), sz * n(2)}) <= Real(0.1)) {
              continue;
            }
            const Vec3r pi = v + m * n;
            const Vec3r p = pi + LogSample(Real(5e-1), Real(5.0), rng) * n;

            const bool diff =
                ComputeProjectionDerivative<3>(&set, p, pi, d_pi_p, d_pi_p_num);
            ASSERT_TRUE(diff);
            EXPECT_PRED3(MatrixNear<3>, d_pi_p, d_pi_p_num, kTolHess);
          }
        }
      }
    }
  }
}

TEST(ProjectionDerivativeTest, Cylinder) {
  const Real hlx = Real(2.0), radius = Real(2.5);

  Matr<3, 3> d_pi_p, d_pi_p_num;
  Rng rng;
  rng.SetSeed(83);

  const Real margins[2] = {Real(0.0), Real(0.5)};

  for (Real m : margins) {
    const Cylinder set(hlx, radius, m);
    const auto dirs = UniformCirclePoints(16);

    // Surface 1: cylindrical surface.
    for (const auto& u : dirs) {
      const Vec3r n(Real(0.0), u(0), u(1));
      const Vec3r pi(rng.Random(hlx * Real(0.8)), (radius + m) * u(0),
                     (radius + m) * u(1));
      const Vec3r p = pi + LogSample(Real(5e-1), Real(5.0), rng) * n;

      const bool diff =
          ComputeProjectionDerivative<3>(&set, p, pi, d_pi_p, d_pi_p_num);
      ASSERT_TRUE(diff);
      EXPECT_PRED3(MatrixNear<3>, d_pi_p, d_pi_p_num, kTolHess);
    }

    // Surface 2: left/right disks.
    for (Real side : {Real(-1.0), Real(1.0)}) {
      const Vec3r n(side, Real(0.0), Real(0.0));
      for (const auto& u : dirs) {
        const Real r = rng.Random(Real(0.1), radius * Real(0.9));
        const Vec3r pi = Vec3r(side * (hlx + m), r * u(0), r * u(1));
        const Vec3r p = pi + LogSample(Real(5e-1), Real(5.0), rng) * n;

        const bool diff =
            ComputeProjectionDerivative<3>(&set, p, pi, d_pi_p, d_pi_p_num);
        ASSERT_TRUE(diff);
        EXPECT_PRED3(MatrixNear<3>, d_pi_p, d_pi_p_num, kTolHess);
      }
    }

    // Surface 3: left/right circular edges.
    for (Real side : {Real(-1.0), Real(1.0)}) {
      for (const auto& u : dirs) {
        const Real theta = rng.Random(Real(0.1), kPi / Real(2.0) - Real(0.1));
        const Vec3r n(side * std::sin(theta), std::cos(theta) * u(0),
                      std::cos(theta) * u(1));
        const Vec3r v(side * hlx, radius * u(0), radius * u(1));
        const Vec3r pi = v + m * n;
        const Real k = rng.Random(std::log2(Real(5e-1)), std::log2(Real(5.0)));
        const Vec3r p = pi + std::exp2(k) * n;

        const bool diff =
            ComputeProjectionDerivative<3>(&set, p, pi, d_pi_p, d_pi_p_num);
        ASSERT_TRUE(diff);
        EXPECT_PRED3(MatrixNear<3>, d_pi_p, d_pi_p_num, kTolHess);
      }
    }
  }
}

TEST(ProjectionDerivativeTest, Cone) {
  const Real ha = kPi / Real(6.0), r = Real(1.5);
  const Real sha = std::sin(ha), cha = std::cos(ha), tha = std::tan(ha);
  const Real h = r / tha;

  Matr<3, 3> d_pi_p, d_pi_p_num;
  Rng rng;
  rng.SetSeed(89);

  const Real margins[2] = {Real(0.0), Real(0.3)};

  for (Real m : margins) {
    const Cone set(r, h, m);
    const Real rho = set.offset();

    const auto cdirs = UniformCirclePoints(16);

    // Surface 1: vertex.
    const auto sdirs = UniformSpherePoints(4, 5, Real(0.3));
    for (const auto& u : sdirs) {
      if (u(2) <= tha * u.head<2>().norm() + Real(0.05)) continue;

      const Vec3r pi(Real(0.0), Real(0.0), h - rho + m);
      const Vec3r p = pi + LogSample(Real(5e-1), Real(5.0), rng) * u;

      const bool diff =
          ComputeProjectionDerivative<3>(&set, p, pi, d_pi_p, d_pi_p_num);
      ASSERT_TRUE(diff);
      EXPECT_PRED3(MatrixNear<3>, d_pi_p, d_pi_p_num, kTolHess);
    }

    // Surface 2: cone surface.
    for (const auto& u : cdirs) {
      const Real z = -rho + rng.Random(Real(0.1), Real(0.9)) * h;
      const Real rz = r - tha * (z + rho);
      Vec3r n = Vec3r(cha * u(0), cha * u(1), sha).normalized();
      const Vec3r pi(rz * u(0) + m * n(0), rz * u(1) + m * n(1), z + m * n(2));
      const Vec3r p = pi + LogSample(Real(5e-1), Real(5.0), rng) * n;

      const bool diff =
          ComputeProjectionDerivative<3>(&set, p, pi, d_pi_p, d_pi_p_num);
      ASSERT_TRUE(diff);
      EXPECT_PRED3(MatrixNear<3>, d_pi_p, d_pi_p_num, kTolHess);
    }

    // Surface 3: base circular edge.
    for (const auto& u : cdirs) {
      const Real psi = rng.Random(Real(0.1), kPi / Real(2.0) - Real(0.1));
      const Vec3r n(std::cos(psi) * u(0), std::cos(psi) * u(1), -std::sin(psi));
      const Vec3r pi(r * u(0) + m * n(0), r * u(1) + m * n(1), -rho + m * n(2));
      const Vec3r p = pi + LogSample(Real(5e-1), Real(5.0), rng) * n;

      const bool diff =
          ComputeProjectionDerivative<3>(&set, p, pi, d_pi_p, d_pi_p_num);
      if (!diff) continue;
      EXPECT_PRED3(MatrixNear<3>, d_pi_p, d_pi_p_num, kTolHess);
    }

    // Surface 4: base disk.
    for (const auto& u : cdirs) {
      const Real rz = r * rng.Random(Real(0.1), Real(0.9));
      const Vec3r pi(rz * u(0), rz * u(1), -rho - m);
      const Vec3r p =
          pi - LogSample(Real(5e-1), Real(5.0), rng) * Vec3r::UnitZ();

      const bool diff =
          ComputeProjectionDerivative<3>(&set, p, pi, d_pi_p, d_pi_p_num);
      ASSERT_TRUE(diff);
      EXPECT_PRED3(MatrixNear<3>, d_pi_p, d_pi_p_num, kTolHess);
    }
  }
}

TEST(ProjectionDerivativeTest, Frustum) {
  Matr<3, 3> d_pi_p, d_pi_p_num;
  Rng rng;
  rng.SetSeed(97);

  const auto cdirs = UniformCirclePoints(5);
  const auto sdirs = UniformSpherePoints(4, 5, Real(0.3));

  // Sub-case parameters: {rb, rt, h}.
  const std::vector<std::array<Real, 3>> params = {
      {Real(2.0), Real(2.0), Real(3.0)},  // cylinder
      {Real(2.0), Real(1.0), Real(3.0)},  // normal frustum  (rb > rt > 0)
      {Real(1.0), Real(2.0), Real(3.0)},  // inverted frustum (rb < rt)
      {Real(2.0), Real(0.0), Real(3.0)},  // cone             (rt = 0)
      {Real(0.0), Real(2.0), Real(3.0)},  // inverted cone    (rb = 0)
  };

  for (const auto& param : params) {
    const Real rb = param[0], rt = param[1], h = param[2];
    const Real tha = (rb - rt) / h;
    const Real ha = std::atan(tha);
    const Real cha = std::cos(ha), sha = std::sin(ha);
    for (Real m : {Real(0.0), Real(0.3)}) {
      const Frustum set(rb, rt, h, m);
      const Real offset = set.offset();

      // --- Lambda helpers -------------------------------------------------

      // Test a point on the frustum surface.
      auto test_lateral = [&](Real rz, Real z) {
        for (const auto& u : cdirs) {
          Vec3r n = Vec3r(cha * u(0), cha * u(1), sha).normalized();
          const Vec3r pi = Vec3r(rz * u(0), rz * u(1), z) + m * n;
          const Vec3r p = pi + LogSample(Real(5e-1), Real(5.0), rng) * n;

          const bool diff =
              ComputeProjectionDerivative<3>(&set, p, pi, d_pi_p, d_pi_p_num);
          ASSERT_TRUE(diff);
          EXPECT_PRED3(MatrixNear<3>, d_pi_p, d_pi_p_num, kTolHess)
              << "Frustum lateral, rb=" << rb << " rt=" << rt << " m=" << m;
        }
      };

      // Test a point on the top/bottom disk.
      auto test_disk = [&](Real rd, Real z, Real ns) {
        for (const auto& u : cdirs) {
          const Real r = rd * rng.Random(Real(0.0), Real(0.85));
          const Vec3r pi(r * u(0), r * u(1), z + m * ns);
          const Vec3r p =
              pi + LogSample(Real(5e-1), Real(5.0), rng) * ns * Vec3r::UnitZ();

          const bool diff =
              ComputeProjectionDerivative<3>(&set, p, pi, d_pi_p, d_pi_p_num);
          ASSERT_TRUE(diff);
          EXPECT_PRED3(MatrixNear<3>, d_pi_p, d_pi_p_num, kTolHess)
              << "Frustum disk, rb=" << rb << " rt=" << rt << " m=" << m;
        }
      };

      // Test a vertex point.
      auto test_vertex = [&](Real z, Real tha_v) {
        for (const auto& n : sdirs) {
          const Real ru = n.head<2>().norm();
          if (tha_v > Real(0.0) && n(2) <= std::abs(tha_v) * ru + Real(0.05))
            continue;
          if (tha_v < Real(0.0) && n(2) >= -std::abs(tha_v) * ru - Real(0.05))
            continue;
          const Vec3r pi(m * n(0), m * n(1), z + m * n(2));
          const Vec3r p = pi + LogSample(Real(5e-1), Real(5.0), rng) * n;

          const bool diff =
              ComputeProjectionDerivative<3>(&set, p, pi, d_pi_p, d_pi_p_num);
          ASSERT_TRUE(diff);
          EXPECT_PRED3(MatrixNear<3>, d_pi_p, d_pi_p_num, kTolHess)
              << "Frustum vertex, rb=" << rb << " rt=" << rt << " m=" << m;
        }
      };

      // Test a circular edge.
      auto test_edge = [&](Real re, Real z, Real ns) {
        const Real psi_min = std::abs(ha) + Real(0.1);
        const Real psi_max = kPi / Real(2.0) - Real(0.1);
        if (psi_min >= psi_max) return;
        for (const auto& u : cdirs) {
          const Real psi = rng.Random(psi_min, psi_max);
          const Vec3r n(std::cos(psi) * u(0), std::cos(psi) * u(1),
                        ns * std::sin(psi));
          const Vec3r pi = Vec3r(re * u(0), re * u(1), z) + m * n;
          const Vec3r p = pi + LogSample(Real(5e-1), Real(5.0), rng) * n;

          const bool diff =
              ComputeProjectionDerivative<3>(&set, p, pi, d_pi_p, d_pi_p_num);
          ASSERT_TRUE(diff);
          EXPECT_PRED3(MatrixNear<3>, d_pi_p, d_pi_p_num, kTolHess)
              << "Frustum edge, rb=" << rb << " rt=" << rt << " m=" << m;
        }
      };

      // --- Run tests for each surface -------------------------------------

      // Surface 1: frustum lateral surface.
      for (int j = 0; j < 6; ++j) {
        const Real f =
            Real(0.15) + static_cast<Real>(j) * Real(0.70) / Real(5.0);
        const Real z = -offset + f * h;
        const Real rz = rb - tha * (z + offset);
        test_lateral(rz, z);
      }

      // Surface 2: frustum top.
      if (rt > Real(0.0)) {
        test_disk(rt, h - offset, Real(1.0));
      } else {
        test_vertex(h - offset, std::abs(tha));  // top apex: u(2) > |tha|*ru
      }

      // Surface 3: frustum base.
      if (rb > Real(0.0)) {
        test_disk(rb, -offset, Real(-1.0));
      } else {
        test_vertex(-offset, -std::abs(tha));  // base apex: u(2) < -|tha|*ru
      }

      // Surface 4: circular edges.
      if (rt > Real(0.0)) test_edge(rt, h - offset, Real(1.0));
      if (rb > Real(0.0)) test_edge(rb, -offset, Real(-1.0));
    }
  }
}

inline void PolytopeProjectionDerivativeTest(
    const ConvexSet<3>* set, const std::vector<Vec3r>& vert,
    const std::vector<Vec3r>& face_normals,
    const std::vector<std::vector<int>>& face_idx, Real m) {
  const int nsamples = 10;
  const int nvert = static_cast<int>(vert.size());
  const int nface = static_cast<int>(face_normals.size());

  Matr<3, 3> d_pi_p, d_pi_p_num;
  Rng rng;
  rng.SetSeed(107);

  BasePointHint<3> hint;

  auto neighbor_face = [&](int f, const std::array<int, 2>& v) -> int {
    for (int i = 0; i < nface; ++i) {
      std::array<bool, 2> exists = {false, false};
      if (i == f) continue;
      for (int j : face_idx[i]) {
        if (j == v[0]) exists[0] = true;
        if (j == v[1]) exists[1] = true;
      }
      if (exists[0] && exists[1]) return i;
    }
    return -1;
  };

  // Surface 1: faces.
  for (int i = 0; i < nsamples; ++i) {
    const int f = rng.RandomInt(0, nface - 1);
    const Vec3r& n = face_normals[f];
    const int nn = static_cast<int>(face_idx[f].size());
    const int v = rng.RandomInt(0, nn - 1);
    Vec3r bc;
    bc(0) = rng.CoinFlip(Real(0.67)) * rng.Random(Real(0.05), Real(0.95));
    bc(1) = rng.Random(Real(0.05), Real(0.95));
    bc(2) = rng.CoinFlip(Real(0.67)) * rng.Random(Real(0.05), Real(0.95));
    bc /= bc.sum();
    Vec3r pi = m * n;
    for (int j = 0; j < 3; ++j) pi += bc(j) * vert[face_idx[f][(v + j) % nn]];
    const Vec3r p = pi + LogSample(Real(5e-1), Real(5.0), rng) * n;

    hint.idx = Vec3i(face_idx[f][v], face_idx[f][(v + 1) % nn],
                     face_idx[f][(v + 2) % nn]);
    hint.bc = bc;

    const bool diff =
        ComputeProjectionDerivative<3>(set, p, pi, d_pi_p, d_pi_p_num, &hint);
    if ((bc(0) < kEps) || (bc(2) < kEps)) {
      ASSERT_FALSE(diff);
      continue;
    }
    ASSERT_TRUE(diff);
    EXPECT_PRED3(MatrixNear<3>, d_pi_p, d_pi_p_num, kTolHess);
  }

  // Surface 2: edges.
  for (int i = 0; i < nsamples; ++i) {
    const int f1 = rng.RandomInt(0, nface - 1);
    const int nn = static_cast<int>(face_idx[f1].size());
    const int j = rng.RandomInt(0, nn - 1);
    const int v1 = face_idx[f1][j];
    const int v2 = face_idx[f1][(j + 1) % nn];
    const int f2 = neighbor_face(f1, {v1, v2});
    ASSERT_GE(f2, 0);

    const Real fn =
        rng.CoinFlip(Real(0.67)) * rng.Random(Real(0.05), Real(0.95));
    const Vec3r n =
        (fn * face_normals[f1] + (Real(1.0) - fn) * face_normals[f2])
            .normalized();
    const Real bc =
        rng.CoinFlip(Real(0.67)) * rng.Random(Real(0.05), Real(0.95));
    Vec3r pi = bc * vert[v1] + (Real(1.0) - bc) * vert[v2] + m * n;
    const Vec3r p = pi + LogSample(Real(5e-1), Real(5.0), rng) * n;

    hint.idx = Vec3i(v1, v2, (v1 + 2) % nn);
    hint.bc = Vec3r(bc, Real(1.0) - bc, Real(0.0));

    const bool diff =
        ComputeProjectionDerivative<3>(set, p, pi, d_pi_p, d_pi_p_num, &hint);
    if ((bc < kEps) || (fn < kEps)) {
      ASSERT_FALSE(diff);
      continue;
    }
    ASSERT_TRUE(diff);
    EXPECT_PRED3(MatrixNear<3>, d_pi_p, d_pi_p_num, kTolHess);
  }

  // Surface 3: vertices.
  for (int i = 0; i < nvert; ++i) {
    const Vec3r& v = vert[i];
    const Vec3r n = v.normalized();
    const Vec3r pi = v + m * n;
    const Vec3r p = pi + LogSample(Real(5e-1), Real(5.0), rng) * n;

    const bool diff =
        ComputeProjectionDerivative<3>(set, p, pi, d_pi_p, d_pi_p_num);
    ASSERT_TRUE(diff);
    EXPECT_PRED3(MatrixNear<3>, d_pi_p, d_pi_p_num, kTolHess);
  }
}

TEST(ProjectionDerivativeTest, Mesh) {
  const std::vector<Vec3r> vert = {
      Vec3r(Real(1), Real(1), Real(1)),
      Vec3r(Real(1), -Real(1), -Real(1)),
      Vec3r(-Real(1), Real(1), -Real(1)),
      Vec3r(-Real(1), -Real(1), Real(1)),
  };
  const Real inradius = Real(1.0) / std::sqrt(Real(3.0));

  const std::vector<int> graph = {
      4, 4,         // nvert, nface
      0, 4, 8, 12,  // vert_edgeadr[0..3]
      1, 2, 3, -1,  // neighbours of v0
      0, 2, 3, -1,  // neighbours of v1
      0, 1, 3, -1,  // neighbours of v2
      0, 1, 2, -1,  // neighbours of v3
  };

  std::vector<Vec3r> face_normals(4);
  for (int i = 0; i < 4; ++i) {
    face_normals[i] = -vert[i] / std::sqrt(Real(3.0));  // opposite to vi
  }
  const std::vector<std::vector<int>> face_idx = {{
      {1, 2, 3},  // face 0: opposite to v0
      {0, 2, 3},  // face 1: opposite to v1
      {0, 1, 3},  // face 2: opposite to v2
      {0, 1, 2},  // face 3: opposite to v3
  }};

  const Real margins[2] = {Real(0.0), Real(0.3)};

  for (Real m : margins) {
    const Mesh set(vert, graph, inradius, m);

    PolytopeProjectionDerivativeTest(&set, vert, face_normals, face_idx, m);
  }
}

TEST(ProjectionDerivativeTest, Polytope) {
  const Real h = Real(1.0);
  const std::vector<Vec3r> vert = {
      Vec3r(h, h, h),  Vec3r(-h, h, h),  Vec3r(-h, -h, h),  Vec3r(h, -h, h),
      Vec3r(h, h, -h), Vec3r(-h, h, -h), Vec3r(-h, -h, -h), Vec3r(h, -h, -h),
  };
  const Real inradius = h;

  const std::vector<Vec3r> face_normals = {{
      Vec3r(0, 0, 1),
      Vec3r(0, 0, -1),
      Vec3r(0, 1, 0),
      Vec3r(0, -1, 0),
      Vec3r(1, 0, 0),
      Vec3r(-1, 0, 0),
  }};
  const std::vector<std::vector<int>> face_idx = {{
      {0, 1, 2, 3},  // +z face
      {7, 6, 5, 4},  // -z face
      {0, 4, 5, 1},  // +y face
      {3, 2, 6, 7},  // -y face
      {0, 3, 7, 4},  // +x face
      {2, 1, 5, 6},  // -x face
  }};

  const Real margins[2] = {Real(0.0), Real(0.3)};

  for (Real m : margins) {
    const Polytope set(vert, inradius, m);

    PolytopeProjectionDerivativeTest(&set, vert, face_normals, face_idx, m);
  }
}

// ---------------------------------------------------------------------------
// Typed tests for CapsuleImpl and SphereImpl projection derivative functions
// ---------------------------------------------------------------------------

TEST(ProjectionDerivativeTest, Stadium) {
  const Real hlx = Real(2.0), radius = Real(1.5);

  Matr<2, 2> d_pi_p, d_pi_p_num;
  Rng rng;
  rng.SetSeed(53);

  const Real margins[2] = {Real(0.0), Real(0.5)};

  for (Real m : margins) {
    const Stadium set(hlx, radius, m);
    const Real r = set.inradius();

    // Surface 1: flat surface.
    for (Real side : {Real(-1.0), Real(1.0)}) {
      const Vec2r n(Real(0.0), side);
      for (int j = 0; j < 10; ++j) {
        const Vec2r pi(rng.Random(hlx * Real(0.8)), side * r);
        const Vec2r p = pi + LogSample(Real(5e-1), Real(5.0), rng) * n;

        const bool diff =
            ComputeProjectionDerivative(&set, p, pi, d_pi_p, d_pi_p_num);
        ASSERT_TRUE(diff);
        EXPECT_PRED3(MatrixNear<2>, d_pi_p, d_pi_p_num, kTolHess);
      }
    }

    // Surface 2: hemispherical cap.
    const auto dirs = UniformCirclePoints(5);
    for (const auto& n : dirs) {
      const Vec2r c(std::copysign(hlx, n(0)), Real(0.0));
      const Vec2r pi = c + r * n;
      const Vec2r p = pi + LogSample(Real(5e-1), Real(5.0), rng) * n;

      const bool diff =
          ComputeProjectionDerivative(&set, p, pi, d_pi_p, d_pi_p_num);
      ASSERT_TRUE(diff);
      EXPECT_PRED3(MatrixNear<2>, d_pi_p, d_pi_p_num, kTolHess);
    }
  }
}

TEST(ProjectionDerivativeTest, Capsule) {
  const Real hlx = Real(2.0), radius = Real(1.5);

  Matr<3, 3> d_pi_p, d_pi_p_num;
  Rng rng;
  rng.SetSeed(71);

  const Real margins[2] = {Real(0.0), Real(0.5)};

  for (Real m : margins) {
    const Capsule set(hlx, radius, m);
    const Real r = set.inradius();

    // Surface 1: cylindrical surface.
    const auto cdirs = UniformCirclePoints(16);
    for (const auto& u : cdirs) {
      const Vec3r n(Real(0.0), u(0), u(1));
      const Vec3r pi(rng.Random(hlx * Real(0.8)), r * u(0), r * u(1));
      const Vec3r p = pi + LogSample(Real(5e-1), Real(5.0), rng) * n;

      const bool diff =
          ComputeProjectionDerivative<3>(&set, p, pi, d_pi_p, d_pi_p_num);
      ASSERT_TRUE(diff);
      EXPECT_PRED3(MatrixNear<3>, d_pi_p, d_pi_p_num, kTolHess);
    }

    // Surface 2: hemispherical caps.
    const auto sdirs = UniformSpherePoints(8, 9, Real(1e-2));
    for (const auto& n : sdirs) {
      if (std::abs(n(0)) < Real(0.1)) continue;

      const Vec3r c(std::copysign(hlx, n(0)), Real(0.0), Real(0.0));
      const Vec3r pi = c + r * n;
      const Vec3r p = pi + LogSample(Real(5e-1), Real(5.0), rng) * n;

      const bool diff =
          ComputeProjectionDerivative(&set, p, pi, d_pi_p, d_pi_p_num);
      ASSERT_TRUE(diff);
      EXPECT_PRED3(MatrixNear<3>, d_pi_p, d_pi_p_num, kTolHess);
    }
  }
}

template <int dim>
void SphereImplProjectionDerivativeTest() {
  Matr<dim, dim> d_pi_p, d_pi_p_num;

  Rng rng;
  rng.SetSeed(59);

  const Real radius = Real(2.0);
  const SphereImpl<dim> set(radius);

  const auto dirs = UniformSpherePoints(8, 9, Real(1e-2));
  const int size = (dim == 2) ? 8 : static_cast<int>(dirs.size());

  for (int i = 0; i < size; ++i) {
    const Vecr<dim> n = dirs[i].head<dim>().normalized();
    const Vecr<dim> pi = radius * n;
    const Vecr<dim> p = pi + LogSample(Real(5e-1), Real(5.0), rng) * n;

    const bool diff =
        ComputeProjectionDerivative<dim>(&set, p, pi, d_pi_p, d_pi_p_num);
    ASSERT_TRUE(diff);
    EXPECT_PRED3(MatrixNear<dim>, d_pi_p, d_pi_p_num, kTolHess);
  }
}

TEST(ProjectionDerivativeTest, Circle) {
  SphereImplProjectionDerivativeTest<2>();
}

TEST(ProjectionDerivativeTest, Sphere) {
  SphereImplProjectionDerivativeTest<3>();
}

}  // namespace
