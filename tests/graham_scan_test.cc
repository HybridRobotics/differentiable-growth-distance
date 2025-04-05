#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "dgd/data_types.h"
// clang-format off
#include "dgd/graham_scan.h"
// clang-format on
#include "dgd/utils.h"

namespace {

using namespace dgd;

const Real kTol{kEpsSqrt};

TEST(GrahamScanTest, ZeroDim) {
  std::vector<Vec2f> pts, vert;
  pts.push_back({1.0, 1.0});
  GrahamScan(pts, vert);
  ASSERT_EQ(vert.size(), 1);
  EXPECT_EQ(vert[0], Vec2f(1.0, 1.0));

  // Duplicate points.
  pts.push_back({1.0, 1.0});
  GrahamScan(pts, vert);
  ASSERT_EQ(vert.size(), 1);
  EXPECT_EQ(vert[0], Vec2f(1.0, 1.0));
}

TEST(GrahamScanTest, OneDim) {
  std::vector<Vec2f> pts, vert;
  pts.push_back({2.0, 3.0});
  pts.push_back({1.0, 1.0});
  GrahamScan(pts, vert);
  ASSERT_EQ(vert.size(), 2);
  EXPECT_EQ(vert[0], Vec2f(1.0, 1.0));
  EXPECT_EQ(vert[1], Vec2f(2.0, 3.0));

  // Duplicate points.
  pts.push_back({1.0, 1.0});
  GrahamScan(pts, vert);
  ASSERT_EQ(vert.size(), 2);
  EXPECT_EQ(vert[0], Vec2f(1.0, 1.0));
  EXPECT_EQ(vert[1], Vec2f(2.0, 3.0));

  // Collinear points.
  pts.push_back({3.0, 5.0});
  GrahamScan(pts, vert);
  ASSERT_EQ(vert.size(), 2);
  EXPECT_EQ(vert[0], Vec2f(1.0, 1.0));
  EXPECT_EQ(vert[1], Vec2f(3.0, 5.0));
}

TEST(GrahamScanTest, TwoDim) {
  // CW-sorted points.
  std::vector<Vec2f> pts, vert;
  pts.push_back({3.0, 2.0});
  pts.push_back({1.0, 0.5});
  pts.push_back({0.0, 0.0});
  GrahamScan(pts, vert);
  ASSERT_EQ(vert.size(), 3);
  EXPECT_EQ(vert[0], Vec2f(0.0, 0.0));
  EXPECT_EQ(vert[1], Vec2f(1.0, 0.5));
  EXPECT_EQ(vert[2], Vec2f(3.0, 2.0));

  // CCW-sorted points.
  pts[1] = Vec2f(2.0, 2.0);
  GrahamScan(pts, vert);
  ASSERT_EQ(vert.size(), 3);
  EXPECT_EQ(vert[0], Vec2f(0.0, 0.0));
  EXPECT_EQ(vert[1], Vec2f(3.0, 2.0));
  EXPECT_EQ(vert[2], Vec2f(2.0, 2.0));

  // Unsorted points with duplicates and p0-collinearity.
  pts.push_back({1.5, 1.0});
  pts.push_back({3.0, 2.0});
  GrahamScan(pts, vert);
  ASSERT_EQ(vert.size(), 3);
  EXPECT_EQ(vert[0], Vec2f(0.0, 0.0));
  EXPECT_EQ(vert[1], Vec2f(3.0, 2.0));
  EXPECT_EQ(vert[2], Vec2f(2.0, 2.0));

  // Unsorted points with duplicates and p0- and non p0-collinearity.
  // Note: For hill-climbing to work correctly, there must not be any
  // collinear points.
  pts.push_back({2.5, 2.0});
  GrahamScan(pts, vert);
  ASSERT_EQ(vert.size(), 3);
  EXPECT_EQ(vert[0], Vec2f(0.0, 0.0));
  EXPECT_EQ(vert[1], Vec2f(3.0, 2.0));
  EXPECT_EQ(vert[2], Vec2f(2.0, 2.0));

  // Unsorted points with duplicates, p0- and non p0-collinearity, and right
  // turns.
  pts.push_back({0.0, 4.0});
  GrahamScan(pts, vert);
  ASSERT_EQ(vert.size(), 3);
  EXPECT_EQ(vert[0], Vec2f(0.0, 0.0));
  EXPECT_EQ(vert[1], Vec2f(3.0, 2.0));
  EXPECT_EQ(vert[2], Vec2f(0.0, 4.0));
}

TEST(GrahamScanTest, CcwOrientation) {
  SetDefaultSeed();
  const int numruns{100};
  const int numpts{1000};
  const Real side_len{10.0};

  auto ccw = [](const Vec2f& u, const Vec2f& v, const Vec2f& w) -> Real {
    return (v - u).cross(w - u);
  };

  std::vector<Vec2f> pts(numpts), vert;
  int numvert;
  for (int i = 0; i < numruns; ++i) {
    for (int j = 0; j < numpts; ++j)
      pts[j] = Vec2f(Random(side_len), Random(side_len));
    numvert = GrahamScan(pts, vert);

    // Orientation test.
    if (numvert > 2) {
      for (int j = 0; j < numvert - 2; ++j)
        ASSERT_GT(ccw(vert[j], vert[j + 1], vert[j + 2]), 0.0);
      ASSERT_GT(ccw(vert.end()[-2], vert.end()[-1], vert[0]), 0.0);
      ASSERT_GT(ccw(vert.end()[-1], vert[0], vert[1]), 0.0);
    }
  }
}

// Support functions of a set and its convex hull are the same.
TEST(GrahamScanTest, SupportFunction) {
  SetDefaultSeed();
  const int numruns{100};
  const int numdir{100};
  const int numpts{1000};
  const Real side_len{10.0};

  auto support = [](const std::vector<Vec2f>& p, const Vec2f& n,
                    Vec2f& sp) -> bool {
    int idx{0};
    Real s{0.0}, sv{n.dot(p[0])};
    bool degenerate = false;
    for (int i = 1; i < static_cast<int>(p.size()); ++i) {
      s = n.dot(p[i]);
      if (s > sv) {
        idx = i;
        sv = s;
        degenerate = false;
      } else if (s == sv)
        degenerate = true;
    }
    sp = p[idx];
    return degenerate;
  };

  std::vector<Vec2f> pts(numpts), vert;
  Vec2f sp1, sp2, dir;
  for (int i = 0; i < numruns; ++i) {
    for (int j = 0; j < numpts; ++j)
      pts[j] = Vec2f(Random(side_len), Random(side_len));
    GrahamScan(pts, vert);

    // Support function test.
    for (int k = 0; k < numdir; ++k) {
      Real ang{2 * kPi / numdir * k};
      dir = Vec2f(std::cos(ang), std::sin(ang));
      bool degenerate{support(pts, dir, sp1)};
      if (degenerate) continue;
      support(vert, dir, sp2);
      EXPECT_NEAR(dir.dot(sp1), dir.dot(sp2), kTol);
      ASSERT_NEAR((sp1 - sp2).lpNorm<Eigen::Infinity>(), 0.0, kTol);
    }
  }
}

}  // namespace
