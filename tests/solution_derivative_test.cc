#include <gtest/gtest.h>

#include <cmath>
#include <functional>
#include <memory>
#include <type_traits>
#include <typeinfo>

#include "dgd/data_types.h"
#include "dgd/dgd.h"
#include "dgd/geometry/geometry_2d.h"
#include "dgd/geometry/geometry_3d.h"
#include "dgd/output.h"
#include "dgd/settings.h"
#include "dgd/solvers/bundle_scheme_impl.h"
#include "dgd/utils/numerical_differentiation.h"
#include "dgd/utils/random.h"
#include "dgd/utils/transformations.h"
#include "test_utils.h"

namespace {

using namespace dgd;
using dgd::test::ConvexSetPtr;
using dgd::test::IntegrateTransform;
using dgd::test::JacobianNear;
using dgd::test::kTol;
using dgd::test::kTolGrad;
using dgd::test::kTwistFrames;
using dgd::test::MakeConvexSetPair;
using dgd::test::PrintSetup;
using dgd::test::VectorNear;

const Real kTolJac = Real(20.0) * kTolGrad;
const Real kStepSize = std::pow(kEps, Real(0.2));
const Real kRelTol = Real(1.0) + std::pow(kEps, Real(0.75));

// ---------------------------------------------------------------------------
// Numerical differentiation utilities
// ---------------------------------------------------------------------------

template <int dim, class C2>
Real ComputeSolutionDerivatives(const ConvexSet<dim>* set1,
                                const KinematicState<dim>& state1,
                                const C2* set2,
                                const KinematicState<dim>& state2,
                                TwistFrame twist_frame, Vecr<dim>& d_normal_num,
                                Vecr<dim>& d_z1_num, Vecr<dim>& d_z2_num) {
  static_assert((std::is_same_v<C2, ConvexSet<dim>>) ||
                    (std::is_same_v<C2, Halfspace<dim>>),
                "Invalid ConvexSet type");

  Settings settings;
  settings.rel_tol = kRelTol;
  const Real h =
      kStepSize *
      std::max(Real(1.0), (Affine(state2.tf) - Affine(state1.tf)).norm());
  const NumericalDifferentiator nd(h, false);

  auto perturbed_solve =
      [&](Real x) -> std::tuple<Vecr<dim>, Vecr<dim>, Vecr<dim>, Real> {
    const Twistr<dim> tw1 = x * state1.tw;
    const Twistr<dim> tw2 = x * state2.tw;

    const Transformr<dim> tf1_new =
        IntegrateTransform<dim>(state1.tf, tw1, twist_frame);
    const Transformr<dim> tf2_new =
        IntegrateTransform<dim>(state2.tf, tw2, twist_frame);

    Output<dim> tmp_out;
    GrowthDistance(set1, tf1_new, set2, tf2_new, settings, tmp_out);

    return {tmp_out.normal, tmp_out.z1, tmp_out.z2, tmp_out.growth_dist_ub};
  };

  VecXr jac(3 * dim + 1);
  nd.Jacobian(
      [&](const Eigen::Ref<const VecXr>& x, Eigen::Ref<VecXr> y) {
        auto [n, z1, z2, gd] = perturbed_solve(x(0));
        y.head<dim>() = n;
        y.segment<dim>(dim) = z1;
        y.segment<dim>(2 * dim) = z2;
        y(3 * dim) = gd;
      },
      Vecr<1>::Zero(), jac);
  d_normal_num = jac.head<dim>();
  d_z1_num = jac.segment<dim>(dim);
  d_z2_num = jac.segment<dim>(2 * dim);
  return jac(3 * dim);
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

template <int dim>
void ConvexSetSolutionDerivativeTest(const ConvexSet<dim>* set1,
                                     const ConvexSet<dim>* set2,
                                     Real jac_tol = kSqrtEps, int nsamples = 50,
                                     int rng_seed = 1) {
  constexpr int tw_dim = SeDim<dim>();

  Rng rng;
  rng.SetSeed(rng_seed);

  KinematicState<dim> state1, state2;

  Settings settings;
  settings.rel_tol = kRelTol;
  settings.jac_tol = jac_tol;
  Output<dim> out;
  DirectionalDerivative<dim> dd;
  TotalDerivative<dim> td;
  OutputBundle<dim> bundle{&out, &dd, &td};

  Vecr<dim> d_normal_num, d_z1_num, d_z2_num;
  Real d_gd, d_gd_num;
  int skipped_samples = 0;
  for (const auto twist_frame : kTwistFrames) {
    settings.twist_frame = twist_frame;

    for (int i = 0; i < nsamples; ++i) {
      Linear(state1.tf) = rng.RandomRotation<dim>();
      Affine(state1.tf) =
          rng.Random(Real(0.5), Real(5.0)) * rng.RandomUnitVector<dim>();
      Linear(state2.tf) = rng.RandomRotation<dim>();

      for (int j = 0; j < tw_dim; ++j) {
        state1.tw(j) = rng.Random(Real(1.0));
        state2.tw(j) = rng.Random(Real(1.0));
      }

      GrowthDistanceCpTpl<BcSolverType::LU>(set1, state1.tf, set2, state2.tf,
                                            settings, out);
      ASSERT_TRUE(out.status == SolutionStatus::Optimal ||
                  out.status == SolutionStatus::CoincidentCenters);
      if (out.status == SolutionStatus::CoincidentCenters) {
        ++skipped_samples;
        continue;
      }

      const bool diff = FactorizeKktSystem(set1, state1.tf, set2, state2.tf,
                                           settings, bundle);
      if (!diff) {
        ++skipped_samples;
        continue;
      }

      d_gd = GdDerivative(state1, state2, settings, bundle);
      GdSolutionDerivative(state1, state2, settings, bundle);
      EXPECT_NEAR(dd.d_gd, d_gd, kTol);

      d_gd_num =
          ComputeSolutionDerivatives(set1, state1, set2, state2, twist_frame,
                                     d_normal_num, d_z1_num, d_z2_num);

      EXPECT_NEAR(dd.d_gd, d_gd_num, kTolGrad);
      EXPECT_PRED3(VectorNear<dim>, dd.d_normal, d_normal_num, kTolJac);
      EXPECT_PRED3(VectorNear<dim>, dd.d_z1, d_z1_num, kTolJac);
      EXPECT_PRED3(VectorNear<dim>, dd.d_z2, d_z2_num, kTolJac);

      /*
      if ((dd.d_normal - d_normal_num).norm() > 0.1 ||
          (dd.d_z1 - d_z1_num).norm() > 0.1 ||
          (dd.d_z2 - d_z2_num).norm() > 0.1) {
        PrintSetup(set1, state1.tf, set2, state2.tf, out);
      }
      */
    }
  }

  /*
  std::cout << "Fraction of samples skipped: "
            << (Real(1.0) * skipped_samples) / (3 * nsamples) << std::endl;
  */
}

// ---------------------------------------------------------------------------
// 2D solution derivative tests
// ---------------------------------------------------------------------------

TEST(GdSolutionDerivativeTest, CircleCircle) {
  constexpr int tw_dim = SeDim<2>();
  const int nsamples = 10;

  const Real r1 = Real(1.5), r2 = Real(1.0), d = Real(5.0);
  Circle set1(r1);
  Circle set2(r2);

  KinematicState<2> state1, state2;
  Affine(state2.tf) << d, Real(0.0);

  const Real gd_true = d / (r1 + r2);

  Settings settings;
  settings.rel_tol = kRelTol;

  Output<2> out;
  DirectionalDerivative<2> dd;
  TotalDerivative<2> td;
  OutputBundle<2> bundle{&out, &dd, &td};

  GrowthDistance(&set1, state1.tf, &set2, state2.tf, settings, out);
  ASSERT_EQ(out.status, SolutionStatus::Optimal);
  EXPECT_NEAR(out.growth_dist_ub, gd_true, kTol);

  Rng rng;
  rng.SetSeed(42);

  for (const auto twist_frame : kTwistFrames) {
    settings.twist_frame = twist_frame;

    const bool diff = FactorizeKktSystem(&set1, state1.tf, &set2, state2.tf,
                                         settings, bundle);
    ASSERT_TRUE(diff);

    GdJacobian(state1.tf, state2.tf, settings, bundle);

    if (twist_frame == TwistFrame::Hybrid) {
      EXPECT_NEAR(td.d_gd_tf1(0), -Real(1.0) / (r1 + r2), kTolGrad);
      EXPECT_NEAR(td.d_gd_tf2(0), Real(1.0) / (r1 + r2), kTolGrad);
    }

    for (int i = 0; i < nsamples; ++i) {
      for (int j = 0; j < tw_dim; ++j) {
        state1.tw(j) = rng.Random(-Real(1.0), Real(1.0));
        state2.tw(j) = rng.Random(-Real(1.0), Real(1.0));
      }

      const Real d_gd = GdDerivative(state1, state2, settings, bundle);
      GdSolutionDerivative(state1, state2, settings, bundle);
      EXPECT_NEAR(dd.d_gd, d_gd, kTol);
    }
  }
}

TEST(GdSolutionDerivativeTest, EllipseEllipse) {
  if (typeid(Real) == typeid(float)) GTEST_SKIP();

  Ellipse set1(Real(2.0), Real(1.0), Real(0.05));
  Ellipse set2(Real(1.5), Real(0.8), Real(0.05));

  const int nsamples = 50;
  ConvexSetSolutionDerivativeTest(&set1, &set2, kSqrtEps, nsamples, 7);
}

TEST(GdSolutionDerivativeTest, EllipsePolygon) {
  if (typeid(Real) == typeid(float)) GTEST_SKIP();

  ConvexSetPtr<2> set1, set2;
  // The numerical derivatives can be inaccurate for large number of polygon
  // vertices.
  MakeConvexSetPair(set1, set2, Real(0.05), Real(0.05), /*npts=*/5);
  // The epsilon quantities are set large only for testing purposes.
  set2->set_eps_p(Real(1e-3));
  set2->set_eps_d(Real(1e-3));

  const int nsamples = 50;
  ConvexSetSolutionDerivativeTest(set1.get(), set2.get(), Real(1e-4), nsamples,
                                  30);
}

TEST(GdSolutionDerivativeTest, EllipseHalfspace) {
  constexpr int tw_dim = SeDim<2>();
  const int nsamples = 25;

  Ellipse set1(Real(2.0), Real(1.0), Real(0.05));
  Halfspace<2> set2(Real(0.05));

  const Real d = Real(1.0);
  KinematicState<2> state1, state2;
  Affine(state1.tf) = d * Vec2r::UnitY();

  Settings settings;
  settings.rel_tol = kRelTol;
  Output<2> out;
  DirectionalDerivative<2> dd;
  TotalDerivative<2> td;
  OutputBundle<2> bundle{&out, &dd, &td};

  Rng rng;
  rng.SetSeed(23);

  Vec2r d_normal_num, d_z1_num, d_z2_num;
  Real d_gd, d_gd_num;
  for (const auto twist_frame : kTwistFrames) {
    settings.twist_frame = twist_frame;

    for (int i = 0; i < nsamples; ++i) {
      Linear(state1.tf) = rng.RandomRotation<2>();

      for (int j = 0; j < tw_dim; ++j) {
        state1.tw(j) = rng.Random(Real(1.0));
        state2.tw(j) = rng.Random(Real(1.0));
      }

      GrowthDistance(&set1, state1.tf, &set2, state2.tf, settings, out);
      ASSERT_TRUE(out.status == SolutionStatus::Optimal ||
                  out.status == SolutionStatus::CoincidentCenters);
      if (out.status == SolutionStatus::CoincidentCenters) continue;

      const bool diff = FactorizeKktSystem(&set1, state1.tf, &set2, state2.tf,
                                           settings, bundle);
      ASSERT_TRUE(diff);

      d_gd = GdDerivative(state1, state2, settings, bundle);
      GdSolutionDerivative(state1, state2, settings, bundle);
      EXPECT_NEAR(dd.d_gd, d_gd, kTol);

      d_gd_num =
          ComputeSolutionDerivatives(&set1, state1, &set2, state2, twist_frame,
                                     d_normal_num, d_z1_num, d_z2_num);

      EXPECT_NEAR(dd.d_gd, d_gd_num, kTolGrad);
      EXPECT_PRED3(VectorNear<2>, dd.d_normal, d_normal_num, kTolJac);
      EXPECT_PRED3(VectorNear<2>, dd.d_z1, d_z1_num, kTolJac);
      EXPECT_PRED3(VectorNear<2>, dd.d_z2, d_z2_num, kTolJac);
    }
  }
}

// ---------------------------------------------------------------------------
// 3D solution derivative tests
// ---------------------------------------------------------------------------

TEST(GdSolutionDerivativeTest, SphereSphere) {
  constexpr int tw_dim = SeDim<3>();
  const int nsamples = 10;

  const Real r1 = Real(1.5), r2 = Real(1.0), d = Real(6.0);
  Sphere set1(r1);
  Sphere set2(r2);

  KinematicState<3> state1, state2;
  Affine(state2.tf) << d, Real(0.0), Real(0.0);

  const Real gd_true = d / (r1 + r2);

  Settings settings;
  settings.rel_tol = kRelTol;

  Output<3> out;
  DirectionalDerivative<3> dd;
  TotalDerivative<3> td;
  OutputBundle<3> bundle{&out, &dd, &td};

  GrowthDistance(&set1, state1.tf, &set2, state2.tf, settings, out);
  ASSERT_EQ(out.status, SolutionStatus::Optimal);
  EXPECT_NEAR(out.growth_dist_ub, gd_true, kTol);

  Rng rng;
  rng.SetSeed(42);

  for (const auto twist_frame : kTwistFrames) {
    settings.twist_frame = twist_frame;

    const bool diff = FactorizeKktSystem(&set1, state1.tf, &set2, state2.tf,
                                         settings, bundle);
    ASSERT_TRUE(diff);

    GdJacobian(state1.tf, state2.tf, settings, bundle);

    if (twist_frame == TwistFrame::Hybrid) {
      EXPECT_NEAR(td.d_gd_tf1(0), -Real(1.0) / (r1 + r2), kTolGrad);
      EXPECT_NEAR(td.d_gd_tf2(0), Real(1.0) / (r1 + r2), kTolGrad);
    }

    for (int i = 0; i < nsamples; ++i) {
      for (int j = 0; j < tw_dim; ++j) {
        state1.tw(j) = rng.Random(-Real(1.0), Real(1.0));
        state2.tw(j) = rng.Random(-Real(1.0), Real(1.0));
      }

      const Real d_gd = GdDerivative(state1, state2, settings, bundle);
      GdSolutionDerivative(state1, state2, settings, bundle);
      EXPECT_NEAR(dd.d_gd, d_gd, kTol);
    }
  }
}

TEST(GdSolutionDerivativeTest, CylinderCylinder) {
  constexpr TwistFrame kFrame = TwistFrame::Hybrid;

  const Real hlx1 = Real(1.0), r1 = Real(0.5);
  const Real hlx2 = Real(1.0), r2 = Real(0.3);
  const Real d = Real(4.0);

  Cylinder set1(hlx1, r1);
  Cylinder set2(hlx2, r2);

  KinematicState<3> state1, state2;
  Affine(state2.tf) = d * Vec3r::UnitZ();
  Linear(state2.tf) = EulerToRotation({Real(0.0), Real(0.0), Real(0.5) * kPi});

  // Optimal solutions.
  const Real gd_true = d / (r1 + r2);
  const Vec3r z1_true = r1 * Vec3r::UnitZ();
  const Vec3r z2_true = (d - r2) * Vec3r::UnitZ();
  const Vec3r normal_true = Vec3r::UnitZ();

  // Optimal solution Jacobians.
  Twist3r d_gd_tf1_true = Twist3r::Zero();
  d_gd_tf1_true(2) = -Real(1.0) / (r1 + r2);
  Twist3r d_gd_tf2_true = Twist3r::Zero();
  d_gd_tf2_true(2) = Real(1.0) / (r1 + r2);

  Matr<3, 6> d_z1_tf1_true = Matr<3, 6>::Zero();
  d_z1_tf1_true.leftCols<3>().setIdentity();
  d_z1_tf1_true(0, 0) -= Real(1.0) / gd_true;  // vx
  d_z1_tf1_true(0, 4) = -r2;                   // wy
  Matr<3, 6> d_z2_tf1_true = Matr<3, 6>::Zero();
  d_z2_tf1_true(1, 1) = Real(1.0) / gd_true;  // vy
  d_z2_tf1_true(0, 4) = -r2;                  // wy
  Matr<3, 6> d_normal_tf1_true = Matr<3, 6>::Zero();
  d_normal_tf1_true(0, 4) = Real(1.0);  // wy

  Matr<3, 6> d_z1_tf2_true = Matr<3, 6>::Zero();
  d_z1_tf2_true(0, 0) = Real(1.0) / gd_true;  // vx
  d_z1_tf2_true(1, 3) = -r1;                  // wx
  Matr<3, 6> d_z2_tf2_true = Matr<3, 6>::Zero();
  d_z2_tf2_true.leftCols<3>().setIdentity();
  d_z2_tf2_true(1, 1) -= Real(1.0) / gd_true;  // vy
  d_z2_tf2_true(1, 3) = -r1;                   // wx
  Matr<3, 6> d_normal_tf2_true = Matr<3, 6>::Zero();
  d_normal_tf2_true(1, 3) = -Real(1.0);  // wx

  Settings settings;
  settings.rel_tol = kRelTol;
  settings.twist_frame = kFrame;

  Output<3> out;
  DirectionalDerivative<3> dd;
  TotalDerivative<3> td;
  OutputBundle<3> bundle{&out, &dd, &td};

  GrowthDistance(&set1, state1.tf, &set2, state2.tf, settings, out);
  ASSERT_EQ(out.status, SolutionStatus::Optimal);
  EXPECT_NEAR(out.growth_dist_ub, gd_true, kTol);
  EXPECT_PRED3(VectorNear<3>, out.z1, z1_true, kTol);
  EXPECT_PRED3(VectorNear<3>, out.z2, z2_true, kTol);
  EXPECT_PRED3(VectorNear<3>, out.normal, normal_true, kTol);

  const bool diff =
      FactorizeKktSystem(&set1, state1.tf, &set2, state2.tf, settings, bundle);
  ASSERT_TRUE(diff);
  GdJacobian(state1.tf, state2.tf, settings, bundle);

  EXPECT_PRED3(VectorNear<6>, td.d_gd_tf1, d_gd_tf1_true, kTolGrad);
  EXPECT_PRED3(VectorNear<6>, td.d_gd_tf2, d_gd_tf2_true, kTolGrad);
  EXPECT_PRED3(JacobianNear<3>, td.d_z1_tf1, d_z1_tf1_true, kTolGrad);
  EXPECT_PRED3(JacobianNear<3>, td.d_z2_tf1, d_z2_tf1_true, kTolGrad);
  EXPECT_PRED3(JacobianNear<3>, td.d_normal_tf1, d_normal_tf1_true, kTolGrad);
  EXPECT_PRED3(JacobianNear<3>, td.d_z1_tf2, d_z1_tf2_true, kTolGrad);
  EXPECT_PRED3(JacobianNear<3>, td.d_z2_tf2, d_z2_tf2_true, kTolGrad);
  EXPECT_PRED3(JacobianNear<3>, td.d_normal_tf2, d_normal_tf2_true, kTolGrad);
}

TEST(GdSolutionDerivativeTest, EllipsoidEllipsoid) {
  if (typeid(Real) == typeid(float)) GTEST_SKIP();

  Ellipsoid set1(Real(2.0), Real(1.0), Real(0.7), Real(0.05));
  Ellipsoid set2(Real(1.5), Real(0.8), Real(1.2), Real(0.05));

  const int nsamples = 50;
  ConvexSetSolutionDerivativeTest(&set1, &set2, kSqrtEps, nsamples, 13);
}

TEST(GdSolutionDerivativeTest, ConeEllipsoid) {
  if (typeid(Real) == typeid(float)) GTEST_SKIP();

  const Real ha = kPi / Real(6.0), radius = Real(1.0);
  const Real height = radius / std::tan(ha);
  Cone set1(radius, height, Real(0.05));
  // The epsilon quantities are set large only for testing purposes.
  set1.set_eps_p(Real(1e-3));
  set1.set_eps_d(Real(1e-3));

  Ellipsoid set2(Real(1.5), Real(0.8), Real(1.2), Real(0.05));

  const int nsamples = 50;
  ConvexSetSolutionDerivativeTest(&set1, &set2, Real(1e-4), nsamples, 55);
}

TEST(GdSolutionDerivativeTest, EllipsoidMesh) {
  if (typeid(Real) == typeid(float)) GTEST_SKIP();

  ConvexSetPtr<3> set1, set2;
  // The numerical derivatives can be inaccurate for large number of polygon
  // vertices.
  MakeConvexSetPair<3>(set1, set2, Real(0.1), Real(0.1), 9);
  set1 =
      std::make_unique<Ellipsoid>(Real(2.0), Real(1.0), Real(0.7), Real(0.05));
  // The epsilon quantities are set large only for testing purposes.
  set2->set_eps_p(Real(0.09));
  set2->set_eps_d(Real(0.09));

  const int nsamples = 50;
  ConvexSetSolutionDerivativeTest(set1.get(), set2.get(), nsamples, 73);
}

TEST(GdSolutionDerivativeTest, FrustumHalfspace) {
  constexpr int tw_dim = SeDim<3>();
  const int nsamples = 50;

  const Real rt = Real(0.5), rb = Real(1.0), height = Real(2.0);
  Frustum set1(rt, rb, height, Real(0.1));
  // The epsilon quantities are set large only for testing purposes.
  set1.set_eps_p(Real(1e-3));
  set1.set_eps_d(Real(1e-3));

  Halfspace<3> set2(Real(0.1));

  const Real d = Real(1.0);
  KinematicState<3> state1, state2;
  Affine(state1.tf) = d * Vec3r::UnitZ();

  Settings settings;
  settings.rel_tol = kRelTol;
  Output<3> out;
  DirectionalDerivative<3> dd;
  TotalDerivative<3> td;
  OutputBundle<3> bundle{&out, &dd, &td};

  Rng rng;
  rng.SetSeed(23);

  Vec3r d_normal_num, d_z1_num, d_z2_num;
  Real d_gd, d_gd_num;
  for (const auto twist_frame : kTwistFrames) {
    settings.twist_frame = twist_frame;

    for (int i = 0; i < nsamples; ++i) {
      Linear(state1.tf) = rng.RandomRotation<3>();

      for (int j = 0; j < tw_dim; ++j) {
        state1.tw(j) = rng.Random(Real(0.1));
        state2.tw(j) = rng.Random(Real(0.1));
      }

      GrowthDistance(&set1, state1.tf, &set2, state2.tf, settings, out);
      ASSERT_TRUE(out.status == SolutionStatus::Optimal ||
                  out.status == SolutionStatus::CoincidentCenters);
      if (out.status == SolutionStatus::CoincidentCenters) continue;

      const bool diff = FactorizeKktSystem(&set1, state1.tf, &set2, state2.tf,
                                           settings, bundle);
      if (!diff) continue;

      d_gd = GdDerivative(state1, state2, settings, bundle);
      GdSolutionDerivative(state1, state2, settings, bundle);
      EXPECT_NEAR(dd.d_gd, d_gd, kTol);

      d_gd_num =
          ComputeSolutionDerivatives(&set1, state1, &set2, state2, twist_frame,
                                     d_normal_num, d_z1_num, d_z2_num);

      EXPECT_NEAR(dd.d_gd, d_gd_num, kTolGrad);
      EXPECT_PRED3(VectorNear<3>, dd.d_normal, d_normal_num, kTolJac);
      EXPECT_PRED3(VectorNear<3>, dd.d_z1, d_z1_num, kTolJac);
      EXPECT_PRED3(VectorNear<3>, dd.d_z2, d_z2_num, kTolJac);
    }
  }
}

}  // namespace
