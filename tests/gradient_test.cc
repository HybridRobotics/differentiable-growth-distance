#include <gtest/gtest.h>

#include <cmath>
#include <functional>
#include <memory>
#include <typeinfo>

#include "dgd/data_types.h"
#include "dgd/dgd.h"
#include "dgd/geometry/geometry_2d.h"
#include "dgd/geometry/geometry_3d.h"
#include "dgd/geometry/halfspace.h"
#include "dgd/output.h"
#include "dgd/settings.h"
#include "dgd/utils/numerical_differentiation.h"
#include "dgd/utils/random.h"
#include "dgd/utils/transformations.h"
#include "test_utils.h"

namespace {

using namespace dgd;
using dgd::test::ConvexSetPtr;
using dgd::test::IntegrateTransform;
using dgd::test::kTolGrad;
using dgd::test::kTwistFrames;
using dgd::test::MakeConvexSetPair;
using dgd::test::VectorNear;

// ---------------------------------------------------------------------------
// Numerical differentiation utilities
// ---------------------------------------------------------------------------

template <int dim>
std::pair<Twistr<dim>, Twistr<dim>> ComputeGdGradient(
    std::function<Real(const Transformr<dim>&, const Transformr<dim>&)> func,
    const Transformr<dim>& tf1, const Transformr<dim>& tf2,
    TwistFrame twist_frame) {
  constexpr int tw_dim = SeDim<dim>();

  auto integrated_func = [&](const Eigen::Ref<const VecXr>& tw) -> Real {
    const Twistr<dim> tw1 = tw.head<tw_dim>();
    const Twistr<dim> tw2 = tw.tail<tw_dim>();

    const Transformr<dim> tf1_new =
        IntegrateTransform<dim>(tf1, tw1, twist_frame);
    const Transformr<dim> tf2_new =
        IntegrateTransform<dim>(tf2, tw2, twist_frame);

    return func(tf1_new, tf2_new);
  };

  Vecr<2 * tw_dim> tw = Vecr<2 * tw_dim>::Zero();
  Vecr<2 * tw_dim> grad;

  NumericalDifferentiator nd;
  nd.Gradient(integrated_func, tw, grad);

  return {grad.template head<tw_dim>(), grad.template tail<tw_dim>()};
}

// ---------------------------------------------------------------------------
// 2D growth distance gradient tests
// ---------------------------------------------------------------------------

TEST(GdGradientTest, CircleCircle) {
  Circle set1(1.0);
  Circle set2(0.5);

  Transform2r tf1 = Transform2r::Identity();
  Affine(tf1) << 0.0, 0.0;

  Transform2r tf2 = Transform2r::Identity();
  Affine(tf2) << 3.0, 0.0;

  Settings settings;

  Output<2> out;
  DirectionalDerivative<2> dd;
  TotalDerivative<2> td;
  OutputBundle<2> bundle{&out, &dd, &td};

  Rng rng;
  rng.SetSeed(42);

  for (const auto twist_frame : kTwistFrames) {
    settings.twist_frame = twist_frame;

    GrowthDistance(&set1, tf1, &set2, tf2, settings, out);
    ComputeKktNullspace(&set1, tf1, &set2, tf2, settings, bundle);
    ASSERT_TRUE(dd.value_differentiable);

    GdGradient(tf1, tf2, settings, bundle);

    auto gd_func = [&](const Transform2r& t1, const Transform2r& t2) -> Real {
      Output<2> tmp_out;
      return GrowthDistance(&set1, t1, &set2, t2, settings, tmp_out);
    };

    auto [d_gd_tf1_num, d_gd_tf2_num] =
        ComputeGdGradient<2>(gd_func, tf1, tf2, settings.twist_frame);

    constexpr int tw_dim = SeDim<2>();
    EXPECT_PRED3(VectorNear<tw_dim>, td.d_gd_tf1, d_gd_tf1_num, kTolGrad);
    EXPECT_PRED3(VectorNear<tw_dim>, td.d_gd_tf2, d_gd_tf2_num, kTolGrad);

    // Test directional derivative with random twists
    KinematicState<2> state1, state2;
    state1.tf = tf1;
    state2.tf = tf2;

    for (int i = 0; i < 10; ++i) {
      for (int j = 0; j < tw_dim; ++j) {
        state1.tw(j) = rng.Random(-1.0, 1.0);
        state2.tw(j) = rng.Random(-1.0, 1.0);
      }

      GdDerivative(state1, state2, settings, bundle);

      const Real d_gd_num =
          td.d_gd_tf1.dot(state1.tw) + td.d_gd_tf2.dot(state2.tw);

      EXPECT_NEAR(dd.d_gd, d_gd_num, kTolGrad);
    }
  }
}

TEST(GdGradientTest, EllipsePolygon) {
  // Numerical derivative computations can be unstable with float.
  if (typeid(Real) == typeid(float)) GTEST_SKIP();

  const int nsamples = 50;

  ConvexSetPtr<2> set1, set2;
  MakeConvexSetPair<2>(set1, set2, Real(0.1), Real(0.1));

  Settings settings;

  Output<2> out;
  DirectionalDerivative<2> dd;
  TotalDerivative<2> td;
  OutputBundle<2> bundle{&out, &dd, &td};

  Rng rng;
  rng.SetSeed(42);

  for (const auto twist_frame : kTwistFrames) {
    settings.twist_frame = twist_frame;

    for (int i = 0; i < nsamples; ++i) {
      const auto tf1 = rng.RandomTransform<2>(-2.0, 2.0);
      const auto tf2 = rng.RandomTransform<2>(-2.0, 2.0);

      GrowthDistance(set1.get(), tf1, set2.get(), tf2, settings, out);
      ComputeKktNullspace(set1.get(), tf1, set2.get(), tf2, settings, bundle);
      if (!dd.value_differentiable) continue;

      GdGradient(tf1, tf2, settings, bundle);

      auto gd_func = [&](const Transform2r& t1, const Transform2r& t2) -> Real {
        Output<2> tmp_out;
        return GrowthDistance(set1.get(), t1, set2.get(), t2, settings,
                              tmp_out);
      };

      auto [d_gd_tf1_num, d_gd_tf2_num] =
          ComputeGdGradient<2>(gd_func, tf1, tf2, settings.twist_frame);

      constexpr int tw_dim = SeDim<2>();
      EXPECT_PRED3(VectorNear<tw_dim>, td.d_gd_tf1, d_gd_tf1_num, kTolGrad);
      EXPECT_PRED3(VectorNear<tw_dim>, td.d_gd_tf2, d_gd_tf2_num, kTolGrad);

      KinematicState<2> state1, state2;
      state1.tf = tf1;
      state2.tf = tf2;

      for (int j = 0; j < tw_dim; ++j) {
        state1.tw(j) = rng.Random(-1.0, 1.0);
        state2.tw(j) = rng.Random(-1.0, 1.0);
      }

      GdDerivative(state1, state2, settings, bundle);

      const Real d_gd_num =
          td.d_gd_tf1.dot(state1.tw) + td.d_gd_tf2.dot(state2.tw);

      EXPECT_NEAR(dd.d_gd, d_gd_num, kTolGrad);
    }
  }
}

// ---------------------------------------------------------------------------
// 3D growth distance gradient tests
// ---------------------------------------------------------------------------

TEST(GdGradientTest, SphereCuboid) {
  Sphere set1(1.0);
  Cuboid set2(1.0, 1.0, 1.0, 0.0);

  Transform3r tf1 = Transform3r::Identity();
  Affine(tf1) << 0.0, 0.0, 0.0;

  Transform3r tf2 = Transform3r::Identity();
  Affine(tf2) << 3.0, 0.0, 0.0;

  Settings settings;

  Output<3> out;
  DirectionalDerivative<3> dd;
  TotalDerivative<3> td;
  OutputBundle<3> bundle{&out, &dd, &td};

  Rng rng;
  rng.SetSeed(42);

  for (const auto twist_frame : kTwistFrames) {
    settings.twist_frame = twist_frame;

    GrowthDistance(&set1, tf1, &set2, tf2, settings, out);
    ComputeKktNullspace(&set1, tf1, &set2, tf2, settings, bundle);
    ASSERT_TRUE(dd.value_differentiable);

    GdGradient(tf1, tf2, settings, bundle);

    auto gd_func = [&](const Transform3r& t1, const Transform3r& t2) -> Real {
      Output<3> tmp_out;
      return GrowthDistance(&set1, t1, &set2, t2, settings, tmp_out);
    };

    auto [d_gd_tf1_num, d_gd_tf2_num] =
        ComputeGdGradient<3>(gd_func, tf1, tf2, settings.twist_frame);

    constexpr int tw_dim = SeDim<3>();
    EXPECT_PRED3(VectorNear<tw_dim>, td.d_gd_tf1, d_gd_tf1_num, kTolGrad);
    EXPECT_PRED3(VectorNear<tw_dim>, td.d_gd_tf2, d_gd_tf2_num, kTolGrad);

    // Test directional derivative with random twists
    KinematicState<3> state1, state2;
    state1.tf = tf1;
    state2.tf = tf2;

    for (int i = 0; i < 10; ++i) {
      for (int j = 0; j < tw_dim; ++j) {
        state1.tw(j) = rng.Random(-1.0, 1.0);
        state2.tw(j) = rng.Random(-1.0, 1.0);
      }

      GdDerivative(state1, state2, settings, bundle);

      const Real d_gd_num =
          td.d_gd_tf1.dot(state1.tw) + td.d_gd_tf2.dot(state2.tw);

      EXPECT_NEAR(dd.d_gd, d_gd_num, kTolGrad);
    }
  }
}

TEST(GdGradientTest, ConeMesh) {
  // Numerical derivative computations can be unstable with float.
  if (typeid(Real) == typeid(float)) GTEST_SKIP();

  const int nsamples = 50;

  ConvexSetPtr<3> set1, set2;
  MakeConvexSetPair<3>(set1, set2, Real(0.1), Real(0.1));

  Settings settings;

  Output<3> out;
  DirectionalDerivative<3> dd;
  TotalDerivative<3> td;
  OutputBundle<3> bundle{&out, &dd, &td};

  Rng rng;
  rng.SetSeed(42);

  for (const auto twist_frame : kTwistFrames) {
    settings.twist_frame = twist_frame;

    for (int i = 0; i < nsamples; ++i) {
      const auto tf1 = rng.RandomTransform<3>(-2.0, 2.0);
      const auto tf2 = rng.RandomTransform<3>(-2.0, 2.0);

      GrowthDistance(set1.get(), tf1, set2.get(), tf2, settings, out);
      ComputeKktNullspace(set1.get(), tf1, set2.get(), tf2, settings, bundle);
      if (!dd.value_differentiable) continue;

      GdGradient(tf1, tf2, settings, bundle);

      auto gd_func = [&](const Transform3r& t1, const Transform3r& t2) -> Real {
        Output<3> tmp_out;
        return GrowthDistance(set1.get(), t1, set2.get(), t2, settings,
                              tmp_out);
      };

      auto [d_gd_tf1_num, d_gd_tf2_num] =
          ComputeGdGradient<3>(gd_func, tf1, tf2, settings.twist_frame);

      constexpr int tw_dim = SeDim<3>();
      EXPECT_PRED3(VectorNear<tw_dim>, td.d_gd_tf1, d_gd_tf1_num, kTolGrad);
      EXPECT_PRED3(VectorNear<tw_dim>, td.d_gd_tf2, d_gd_tf2_num, kTolGrad);

      KinematicState<3> state1, state2;
      state1.tf = tf1;
      state2.tf = tf2;

      for (int j = 0; j < tw_dim; ++j) {
        state1.tw(j) = rng.Random(-1.0, 1.0);
        state2.tw(j) = rng.Random(-1.0, 1.0);
      }

      GdDerivative(state1, state2, settings, bundle);

      const Real d_gd_num =
          td.d_gd_tf1.dot(state1.tw) + td.d_gd_tf2.dot(state2.tw);

      EXPECT_NEAR(dd.d_gd, d_gd_num, kTolGrad);
    }
  }
}

TEST(GdGradientTest, FrustumHalfspace) {
  // Numerical derivative computations can be unstable with float.
  if (typeid(Real) == typeid(float)) GTEST_SKIP();

  const Real rt = 0.5;
  const Real rb = 1.0;
  const Real height = 2.0;
  Frustum set1(rt, rb, height, Real(0.1));
  Halfspace<3> set2(Real(0.1));

  Settings settings;

  Output<3> out;
  DirectionalDerivative<3> dd;
  TotalDerivative<3> td;
  OutputBundle<3> bundle{&out, &dd, &td};

  Rng rng;
  rng.SetSeed(42);

  for (const auto twist_frame : kTwistFrames) {
    settings.twist_frame = twist_frame;

    const Transform3r tf2 = Transform3r::Identity();

    for (int i = 0; i < 20; ++i) {
      const auto tf1 = rng.RandomTransform<3>(-2.0, 2.0);

      GrowthDistance(&set1, tf1, &set2, tf2, settings, out);
      ComputeKktNullspace(&set1, tf1, &set2, tf2, settings, bundle);
      if (!dd.value_differentiable) continue;

      GdGradient(tf1, tf2, settings, bundle);

      auto gd_func = [&](const Transform3r& t1, const Transform3r& t2) -> Real {
        Output<3> tmp_out;
        return GrowthDistance(&set1, t1, &set2, t2, settings, tmp_out);
      };

      auto [d_gd_tf1_num, d_gd_tf2_num] =
          ComputeGdGradient<3>(gd_func, tf1, tf2, settings.twist_frame);

      constexpr int tw_dim = SeDim<3>();
      EXPECT_PRED3(VectorNear<tw_dim>, td.d_gd_tf1, d_gd_tf1_num, kTolGrad);
      EXPECT_PRED3(VectorNear<tw_dim>, td.d_gd_tf2, d_gd_tf2_num, kTolGrad);

      KinematicState<3> state1, state2;
      state1.tf = tf1;
      state2.tf = tf2;

      for (int j = 0; j < tw_dim; ++j) {
        state1.tw(j) = rng.Random(-1.0, 1.0);
        state2.tw(j) = rng.Random(-1.0, 1.0);
      }

      GdDerivative(state1, state2, settings, bundle);

      const Real d_gd_num =
          td.d_gd_tf1.dot(state1.tw) + td.d_gd_tf2.dot(state2.tw);

      EXPECT_NEAR(dd.d_gd, d_gd_num, kTolGrad);
    }
  }
}

}  // namespace
