#include <gtest/gtest.h>

#include <cmath>
#include <functional>
#include <memory>

#include "dgd/data_types.h"
#include "dgd/dgd.h"
#include "dgd/geometry/geometry_2d.h"
#include "dgd/geometry/geometry_3d.h"
#include "dgd/graham_scan.h"
#include "dgd/mesh_loader.h"
#include "dgd/settings.h"
#include "dgd/utils/numerical_differentiation.h"
#include "dgd/utils/random.h"
#include "dgd/utils/transformations.h"

namespace {

using namespace dgd;

constexpr std::array<TwistFrame, 3> kTwistFrames = {
    TwistFrame::Spatial, TwistFrame::Hybrid, TwistFrame::Body};

// Integrates a rigid body transform along a small twist using first-order
// approximation.
template <int dim>
Transformr<dim> IntegrateTransform(const Transformr<dim>& tf,
                                   const Twistr<dim>& tw,
                                   TwistFrame twist_frame) {
  // Convert twist to hybrid frame.
  Twistr<dim> tw_h = tw;
  if (twist_frame == TwistFrame::Spatial) {
    Linear(tw_h) = VelocityAtPoint(tw, Affine(tf));
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

// Computes the numerical gradient of a function with respect to two rigid body
// transformations.
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

  // Evaluate the gradient at the zero twist.
  Vecr<2 * tw_dim> tw = Vecr<2 * tw_dim>::Zero();
  Vecr<2 * tw_dim> grad;

  NumericalDifferentiator nd;
  nd.Gradient(integrated_func, tw, grad);

  Twistr<dim> d_gd_tf1 = grad.template head<tw_dim>();
  Twistr<dim> d_gd_tf2 = grad.template tail<tw_dim>();

  return std::make_pair(d_gd_tf1, d_gd_tf2);
}

// Assertion functions
const Real kTol = kSqrtEps;
const Real kTolGrad = std::sqrt(kSqrtEps);

template <int dim>
bool AssertVectorEQ(const Vecr<dim>& v1, const Vecr<dim>& v2, Real tol) {
  return (v1 - v2).template lpNorm<Eigen::Infinity>() < tol;
}

template <int dim>
using ConvexSetPtr = std::unique_ptr<ConvexSet<dim>>;

template <int dim>
void SetConvexSets(ConvexSetPtr<dim>& set1, ConvexSetPtr<dim>& set2,
                   const Real margin1, const Real margin2);

// Ellipse-Polygon
template <>
void SetConvexSets<2>(ConvexSetPtr<2>& set1, ConvexSetPtr<2>& set2,
                      const Real margin1, const Real margin2) {
  // Set 1: ellipse.
  const Real hlx = Real(3.0), hly = Real(2.0);
  set1 = std::make_unique<Ellipse>(hlx, hly, margin1);

  // Set 2: polygon.
  Rng rng;
  rng.SetSeed(42);
  const int npts = 50;
  const Real len = Real(2.0);
  std::vector<Vec2r> pts, vert;
  Vec2r vec;
  const int pnorm = 6;
  for (int i = 0; i < npts; ++i) {
    vec << rng.Random(), rng.Random();
    vec = vec * len / (vec.lpNorm<pnorm>() + kEps);
    pts.push_back(vec);
  }
  GrahamScan(pts, vert);
  Real inradius = ComputePolygonInradius(vert, Vec2r::Zero());
  set2 = std::make_unique<Polygon>(vert, inradius, margin2);
}

// Cone-Mesh
template <>
void SetConvexSets<3>(ConvexSetPtr<3>& set1, ConvexSetPtr<3>& set2,
                      const Real margin1, const Real margin2) {
  // Set 1: cone.
  const Real ha = kPi / Real(6.0), radius = Real(1.0);
  const Real height = radius / std::tan(ha);
  set1 = std::make_unique<Cone>(radius, height, margin1);

  // Set 2: mesh.
  Rng rng;
  rng.SetSeed(42);
  const int npts = 200;
  const Real len = Real(2.0);
  MeshLoader ml{};
  std::vector<Vec3r> pts, vert;
  std::vector<int> graph;
  Vec3r vec;
  constexpr int pnorm = 4;
  for (int i = 0; i < npts; ++i) {
    vec << rng.Random(), rng.Random(), rng.Random();
    vec = vec * len / (vec.lpNorm<pnorm>() + kEps);
    pts.push_back(vec);
  }
  ml.ProcessPoints(pts);
  bool valid = ml.MakeVertexGraph(vert, graph);
  ASSERT_TRUE(valid);
  const Real inradius = ml.ComputeInradius(vec);
  for (auto& v : vert) v -= vec;
  set2 = std::make_unique<Mesh>(vert, graph, inradius, margin2);
}

// Circle-Circle
TEST(GdGradientTest, AnalyticalGradient2D) {
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
    GdGradient(&set1, tf1, &set2, tf2, settings, bundle);

    ASSERT_TRUE(dd.value_differentiable);

    auto gd_func = [&](const Transform2r& t1, const Transform2r& t2) -> Real {
      Output<2> tmp_out;
      return GrowthDistance(&set1, t1, &set2, t2, settings, tmp_out);
    };

    auto [d_gd_tf1_num, d_gd_tf2_num] =
        ComputeGdGradient<2>(gd_func, tf1, tf2, settings.twist_frame);

    constexpr int tw_dim = SeDim<2>();
    EXPECT_PRED3(AssertVectorEQ<tw_dim>, td.d_gd_tf1, d_gd_tf1_num, kTolGrad);
    EXPECT_PRED3(AssertVectorEQ<tw_dim>, td.d_gd_tf2, d_gd_tf2_num, kTolGrad);

    // Test directional derivative with random twists
    KinematicState<2> state1, state2;
    state1.tf = tf1;
    state2.tf = tf2;

    for (int i = 0; i < 10; ++i) {
      for (int j = 0; j < tw_dim; ++j) {
        state1.tw(j) = rng.Random(-1.0, 1.0);
        state2.tw(j) = rng.Random(-1.0, 1.0);
      }

      GdDerivative(&set1, state1, &set2, state2, settings, bundle);

      const Real d_gd_num =
          td.d_gd_tf1.dot(state1.tw) + td.d_gd_tf2.dot(state2.tw);

      EXPECT_NEAR(dd.d_gd, d_gd_num, kTolGrad);
    }
  }
}

TEST(GdGradientTest, AnalyticalGradient3D) {
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
    GdGradient(&set1, tf1, &set2, tf2, settings, bundle);

    ASSERT_TRUE(dd.value_differentiable);

    auto gd_func = [&](const Transform3r& t1, const Transform3r& t2) -> Real {
      Output<3> tmp_out;
      return GrowthDistance(&set1, t1, &set2, t2, settings, tmp_out);
    };

    auto [d_gd_tf1_num, d_gd_tf2_num] =
        ComputeGdGradient<3>(gd_func, tf1, tf2, settings.twist_frame);

    constexpr int tw_dim = SeDim<3>();
    EXPECT_PRED3(AssertVectorEQ<tw_dim>, td.d_gd_tf1, d_gd_tf1_num, kTolGrad);
    EXPECT_PRED3(AssertVectorEQ<tw_dim>, td.d_gd_tf2, d_gd_tf2_num, kTolGrad);

    // Test directional derivative with random twists
    KinematicState<3> state1, state2;
    state1.tf = tf1;
    state2.tf = tf2;

    for (int i = 0; i < 10; ++i) {
      for (int j = 0; j < tw_dim; ++j) {
        state1.tw(j) = rng.Random(-1.0, 1.0);
        state2.tw(j) = rng.Random(-1.0, 1.0);
      }

      GdDerivative(&set1, state1, &set2, state2, settings, bundle);

      const Real d_gd_num =
          td.d_gd_tf1.dot(state1.tw) + td.d_gd_tf2.dot(state2.tw);

      EXPECT_NEAR(dd.d_gd, d_gd_num, kTolGrad);
    }
  }
}

TEST(GdGradientTest, NumericalGradient2D) {
  // Numerical derivative computations can be unstable with float.
  if (typeid(Real) == typeid(float)) GTEST_SKIP();

  const int nsamples = 50;

  ConvexSetPtr<2> set1, set2;
  SetConvexSets<2>(set1, set2, Real(0.1), Real(0.1));

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
      GdGradient(set1.get(), tf1, set2.get(), tf2, settings, bundle);

      if (!dd.value_differentiable) continue;

      auto gd_func = [&](const Transform2r& t1, const Transform2r& t2) -> Real {
        Output<2> tmp_out;
        return GrowthDistance(set1.get(), t1, set2.get(), t2, settings,
                              tmp_out);
      };

      auto [d_gd_tf1_num, d_gd_tf2_num] =
          ComputeGdGradient<2>(gd_func, tf1, tf2, settings.twist_frame);

      constexpr int tw_dim = SeDim<2>();
      EXPECT_PRED3(AssertVectorEQ<tw_dim>, td.d_gd_tf1, d_gd_tf1_num, kTolGrad);
      EXPECT_PRED3(AssertVectorEQ<tw_dim>, td.d_gd_tf2, d_gd_tf2_num, kTolGrad);

      KinematicState<2> state1, state2;
      state1.tf = tf1;
      state2.tf = tf2;

      for (int j = 0; j < tw_dim; ++j) {
        state1.tw(j) = rng.Random(-1.0, 1.0);
        state2.tw(j) = rng.Random(-1.0, 1.0);
      }

      GdDerivative(set1.get(), state1, set2.get(), state2, settings, bundle);

      const Real d_gd_num =
          td.d_gd_tf1.dot(state1.tw) + td.d_gd_tf2.dot(state2.tw);

      EXPECT_NEAR(dd.d_gd, d_gd_num, kTolGrad);
    }
  }
}

TEST(GdGradientTest, NumericalGradient3D) {
  // Numerical derivative computations can be unstable with float.
  if (typeid(Real) == typeid(float)) GTEST_SKIP();

  const int nsamples = 50;

  ConvexSetPtr<3> set1, set2;
  SetConvexSets<3>(set1, set2, Real(0.1), Real(0.1));

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
      GdGradient(set1.get(), tf1, set2.get(), tf2, settings, bundle);

      if (!dd.value_differentiable) continue;

      auto gd_func = [&](const Transform3r& t1, const Transform3r& t2) -> Real {
        Output<3> tmp_out;
        return GrowthDistance(set1.get(), t1, set2.get(), t2, settings,
                              tmp_out);
      };

      auto [d_gd_tf1_num, d_gd_tf2_num] =
          ComputeGdGradient<3>(gd_func, tf1, tf2, settings.twist_frame);

      constexpr int tw_dim = SeDim<3>();
      EXPECT_PRED3(AssertVectorEQ<tw_dim>, td.d_gd_tf1, d_gd_tf1_num, kTolGrad);
      EXPECT_PRED3(AssertVectorEQ<tw_dim>, td.d_gd_tf2, d_gd_tf2_num, kTolGrad);

      KinematicState<3> state1, state2;
      state1.tf = tf1;
      state2.tf = tf2;

      for (int j = 0; j < tw_dim; ++j) {
        state1.tw(j) = rng.Random(-1.0, 1.0);
        state2.tw(j) = rng.Random(-1.0, 1.0);
      }

      GdDerivative(set1.get(), state1, set2.get(), state2, settings, bundle);

      const Real d_gd_num =
          td.d_gd_tf1.dot(state1.tw) + td.d_gd_tf2.dot(state2.tw);

      EXPECT_NEAR(dd.d_gd, d_gd_num, kTolGrad);
    }
  }
}

TEST(GdGradientTest, NumericalGradientHalfspace3D) {
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
      GdGradient(&set1, tf1, &set2, tf2, settings, bundle);

      if (!dd.value_differentiable) continue;

      auto gd_func = [&](const Transform3r& t1, const Transform3r& t2) -> Real {
        Output<3> tmp_out;
        return GrowthDistance(&set1, t1, &set2, t2, settings, tmp_out);
      };

      auto [d_gd_tf1_num, d_gd_tf2_num] =
          ComputeGdGradient<3>(gd_func, tf1, tf2, settings.twist_frame);

      constexpr int tw_dim = SeDim<3>();
      EXPECT_PRED3(AssertVectorEQ<tw_dim>, td.d_gd_tf1, d_gd_tf1_num, kTolGrad);
      EXPECT_PRED3(AssertVectorEQ<tw_dim>, td.d_gd_tf2, d_gd_tf2_num, kTolGrad);

      KinematicState<3> state1, state2;
      state1.tf = tf1;
      state2.tf = tf2;

      for (int j = 0; j < tw_dim; ++j) {
        state1.tw(j) = rng.Random(-1.0, 1.0);
        state2.tw(j) = rng.Random(-1.0, 1.0);
      }

      GdDerivative(&set1, state1, &set2, state2, settings, bundle);

      const Real d_gd_num =
          td.d_gd_tf1.dot(state1.tw) + td.d_gd_tf2.dot(state2.tw);

      EXPECT_NEAR(dd.d_gd, d_gd_num, kTolGrad);
    }
  }
}

}  // namespace
