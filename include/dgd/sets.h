// Copyright 2025 Akshay Thirugnanam
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * @file sets.h
 * @author Akshay Thirugnanam (akshay_t@berkeley.edu)
 * @date 2025-03-01
 * @brief Header file to include all convex set implementations.
 */

#ifndef DGD_SETS_H_
#define DGD_SETS_H_

// Abstract convex set.
#include "dgd/geometry/convex_set.h"

// 2D convex sets.
#include "dgd/geometry/2d/ellipse.h"
#include "dgd/geometry/2d/rectangle.h"

// 3D convex sets.
#include "dgd/geometry/3d/cone.h"
#include "dgd/geometry/3d/cuboid.h"
#include "dgd/geometry/3d/cylinder.h"
#include "dgd/geometry/3d/ellipsoid.h"

// 2D/3D convex sets.
#include "dgd/geometry/xd/capsule.h"
#include "dgd/geometry/xd/polytope.h"
#include "dgd/geometry/xd/sphere.h"

#endif  // DGD_SETS_H_
