#include <gtest/gtest.h>

#include <cmath>

#include "dgd/data_types.h"
// clang-format off
#include "dgd/utils.h"
// clang-format on

namespace {

using namespace dgd;

const Real kTol{kEpsSqrt};

TEST(RotationToZAxisTest, TwoDim) {
  SetDefaultSeed();
  const int nsamples{1000};

  Real ang{0.0}, err{0.0};
  Vec2f n;
  Rot2f rot;
  for (int i = 0; i < nsamples; ++i) {
    ang = Random(kPi);
    n = Vec2f(std::cos(ang), std::sin(ang));
    RotationToZAxis(n, rot);

    EXPECT_TRUE(rot.isUnitary(kTol));
    EXPECT_NEAR(rot.determinant(), 1.0, kTol);
    err = (rot * n - Vec2f::UnitY()).lpNorm<Eigen::Infinity>();
    EXPECT_NEAR(err, 0.0, kTol);
  }
}

TEST(RotationToZAxisTest, ThreeDim) {
  SetDefaultSeed();
  const int nsamples{1000};

  Real ang_y{0.0}, ang_z{0.0}, err{0.0};
  Vec3f n;
  Rot3f rot;
  for (int i = 0; i < nsamples; ++i) {
    ang_y = Random(kPi / 2.0);
    ang_z = Random(kPi);
    n = Vec3f(std::cos(ang_z) * std::cos(ang_y),
              std::sin(ang_z) * std::cos(ang_y), std::sin(ang_y));
    RotationToZAxis(n, rot);

    EXPECT_TRUE(rot.isUnitary(kTol));
    EXPECT_NEAR(rot.determinant(), 1.0, kTol);
    err = (rot * n - Vec3f::UnitZ()).lpNorm<Eigen::Infinity>();
    EXPECT_NEAR(err, 0.0, kTol);
  }

  n = -Vec3f::UnitZ();
  RotationToZAxis(n, rot);

  EXPECT_TRUE(rot.isUnitary(kTol));
  EXPECT_NEAR(rot.determinant(), 1.0, kTol);
  err = (rot * n - Vec3f::UnitZ()).lpNorm<Eigen::Infinity>();
  EXPECT_NEAR(err, 0.0, kTol);
}

}  // namespace
