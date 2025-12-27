// Copyright 2021 DeepMind Technologies Limited
// Copyright 2025 Akshay Thirugnanam
//
// Copied and adapted from MuJoCo
// (https://github.com/google-deepmind/mujoco/blob/main/src/user/user_mesh.cc)
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
 * @author Akshay Thirugnanam (akshay_t@berkeley.edu)
 * @brief 3D mesh loader class implementation.
 */

#include "dgd/mesh_loader.h"

#include <Eigen/Dense>
#include <cmath>
#include <csetjmp>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#define TINYOBJLOADER_IMPLEMENTATION  // define this in only *one* .cc
// Optional: gives robust triangulation. Requires C++11.
// #define TINYOBJLOADER_USE_MAPBOX_EARCUT
#include "tinyobjloader/tiny_obj_loader.h"

extern "C" {
#include "libqhull_r/qhull_ra.h"
}

#include "dgd/data_types.h"

namespace dgd {

void MeshLoader::LoadObj(const std::string& input, bool is_file) {
  tinyobj::ObjReader objReader;

  if (is_file) {
    objReader.ParseFromFile(input);
  } else {
    objReader.ParseFromString(input, std::string());
  }

  if (!objReader.Valid()) {
    std::string err = "Could not parse .obj file";
    if (is_file) err += std::string(": ") + input;
    throw std::runtime_error(err);
  }

  const auto& attrib = objReader.GetAttrib();
  normal_ = attrib.normals;
  facenormal_.clear();

  if (!objReader.GetShapes().empty()) {
    const auto& mesh = objReader.GetShapes()[0].mesh;

    // Iterate over mesh faces.
    std::vector<tinyobj::index_t> face_indices;
    for (int face = 0, idx = 0; idx < static_cast<int>(mesh.indices.size());) {
      int nfacevert = mesh.num_face_vertices[face];
      if (nfacevert < 3 || nfacevert > 4) {
        throw std::runtime_error(
            "Only tri or quad meshes are supported for OBJ");
      }

      for (int j = 0; j < 3; ++j) face_indices.push_back(mesh.indices[idx + j]);

      if (nfacevert == 4) {
        // Second triangle of quad.
        face_indices.push_back(mesh.indices[idx]);
        face_indices.push_back(mesh.indices[idx + 2]);
        face_indices.push_back(mesh.indices[idx + 3]);
      }
      idx += nfacevert;
      ++face;
    }

    // For each vertex, store index and normal.
    for (const auto& mesh_index : face_indices) {
      face_.push_back(mesh_index.vertex_index);

      if (!normal_.empty()) {
        facenormal_.push_back(mesh_index.normal_index);
      }
    }
  }

  // Copy vertex (point) data.
  ProcessPoints(attrib.vertices);
}

bool MeshLoader::MakeVertexGraph(std::vector<Vec3r>& vert,
                                 std::vector<int>& graph) {
  vert.clear();
  graph.clear();

  int adr, ok, curlong, totlong, exitcode;
  facetT *facet, **facetp;
  vertexT *vertex, *vertex1, **vertex1p;

  std::string qhopt = "qhull Qt";
  if (maxhullvert_ > -1) {
    // qhull "TA" actually means "number of vertices added after the initial
    // simplex".
    qhopt += " TA" + std::to_string(maxhullvert_ - 4);
  }

  // Graph not needed for small meshes.
  if (npts() < 4) return false;

  qhT qh_qh;
  qhT* qh = &qh_qh;
  qh_zero(qh, stderr);

  // qhull basic init.
  qh_init_A(qh, stdin, stdout, stderr, 0, NULL);

  // Install longjmp error handler.
  exitcode = setjmp(qh->errexit);
  qh->NOerrexit = false;
  if (!exitcode) {
    // Actual init.
    qh_initflags(qh, const_cast<char*>(qhopt.c_str()));
    double* data = this->pts_.data();
    qh_init_B(qh, data, npts(), 3, False);

    // Construct convex hull.
    qh_qhull(qh);
    qh_triangulate(qh);
    qh_vertexneighbors(qh);

    // Allocate memory for the vertex graph:
    // nvert, nface, vert_edgeadr[nvert], edge_localid[nvert + 3*nface]
    const int nvert = qh->num_vertices;
    const int nface = qh->num_facets;
    const int szgraph = 2 + 2 * nvert + 3 * nface;
    vert.resize(nvert);
    graph.resize(szgraph);
    graph[0] = nvert;
    graph[1] = nface;

    std::unordered_map<int, int> vert_globalid;
    vert_globalid.reserve(nvert);

    // Indices for convenience.
    const int vert_edgeadr = 2;
    const int edge_localid = 2 + nvert;

    // Fill in graph data.
    int i = adr = 0;
    ok = 1;
    FORALLvertices {
      // Point id of this vertex, check.
      const int pid = qh_pointid(qh, vertex->point);
      if (pid < 0 || pid >= npts()) {
        ok = 0;
        break;
      }

      // Save edge address and global id of this vertex.
      graph[vert_edgeadr + i] = adr;
      vert_globalid[pid] = i;
      const double* p = &pts_[3 * pid];
      vert[i] = Vec3r(Real(p[0]), Real(p[1]), Real(p[2]));

      // Process neighboring faces and their vertices.
      const int start = adr;
      FOREACHsetelement_(facetT, vertex->neighbors, facet) {
        int cnt = 0;
        FOREACHsetelement_(vertexT, facet->vertices, vertex1) {
          ++cnt;

          // Point id of face vertex, check.
          const int pid1 = qh_pointid(qh, vertex1->point);
          if (pid1 < 0 || pid1 >= npts()) {
            ok = 0;
            break;
          }

          // If different from vertex id, try to insert.
          if (pid != pid1) {
            // Check for previous record.
            int j;
            for (j = start; j < adr; ++j)
              if (pid1 == graph[edge_localid + j]) break;

            // Not found: insert.
            if (j >= adr) {
              graph[edge_localid + adr] = pid1;
              ++adr;
            }
          }
        }

        // Make sure we have triangle: SHOULD NOT OCCUR.
        if (cnt != 3) throw std::runtime_error("Qhull did not return triangle");
      }

      // Insert separator, advance to next vertex.
      graph[edge_localid + adr] = -1;
      ++adr;
      ++i;
    }

    // Size check: SHOULD NOT OCCUR.
    if (adr != nvert + 3 * nface) {
      throw std::runtime_error("Wrong size in convex hull graph");
    }

    // Free all.
    qh_freeqhull(qh, !qh_ALL);
    qh_memfreeshort(qh, &curlong, &totlong);

    // Bad graph: delete.
    if (!ok) {
      vert.clear();
      graph.clear();
      return false;
    }

    // Replace global ids with local ids in edge data.
    for (int i = 0; i < nvert + 3 * nface; ++i) {
      int& vid = graph[edge_localid + i];
      if (vid >= 0) {
        // Make sure we found a match: SHOULD NOT OCCUR.
        const auto it = vert_globalid.find(vid);
        if (it == vert_globalid.end()) {
          throw std::runtime_error("Vertex id not found in convex hull");
        }

        vid = it->second;
      }
    }
  }

  // longjmp error handler.
  else {
    // Free all.
    qh_freeqhull(qh, !qh_ALL);
    qh_memfreeshort(qh, &curlong, &totlong);
    vert.clear();
    graph.clear();

    throw std::runtime_error("Qhull error");
  }

  return true;
}

bool MeshLoader::MakeFacetGraph(std::vector<Vec3r>& normal,
                                std::vector<Real>& offset,
                                std::vector<int>& graph,
                                Vec3r& interior_point) {
  normal.clear();
  offset.clear();
  graph.clear();

  int adr, ok, curlong, totlong, exitcode;
  facetT *facet, *facet1, **facet1p;

  std::string qhopt = "qhull Qt";
  if (maxhullvert_ > -1) {
    // qhull "TA" actually means "number of vertices added after the initial
    // simplex".
    qhopt += " TA" + std::to_string(maxhullvert_ - 4);
  }

  // Graph not needed for small meshes.
  if (npts() < 4) return false;

  // Function to evaluate facet equation at point.
  auto eqn = [](facetT* facet, double* pt) -> double {
    return (facet->normal[0] * pt[0] + facet->normal[1] * pt[1] +
            facet->normal[2] * pt[2] + facet->offset);
  };
  const double kEpsEqn = std::pow(kEps, 0.9);

  qhT qh_qh;
  qhT* qh = &qh_qh;
  qh_zero(qh, stderr);

  // qhull basic init.
  qh_init_A(qh, stdin, stdout, stderr, 0, NULL);

  // Install longjmp error handler.
  exitcode = setjmp(qh->errexit);
  qh->NOerrexit = false;
  if (!exitcode) {
    // Actual init.
    qh_initflags(qh, const_cast<char*>(qhopt.c_str()));
    double* data = this->pts_.data();
    qh_init_B(qh, data, npts(), 3, False);

    // Construct convex hull.
    qh_qhull(qh);

    // Allocate memory for the facet graph:
    // nfacet, nridge, facet_ridgeadr[nfacet], ridge_localid[nfacet + 2*nridge]
    const int nvert = qh->num_vertices;
    const int nfacet = qh->num_facets;
    const int nridge = nfacet + nvert - 2;
    const int szgraph = 2 + 2 * nfacet + 2 * nridge;
    normal.resize(nfacet);
    offset.resize(nfacet);
    graph.resize(szgraph);
    graph[0] = nfacet;
    graph[1] = nridge;
    for (int i = 0; i < 3; ++i) interior_point(i) = Real(qh->interior_point[i]);

    std::unordered_map<unsigned int, int> facet_globalid;
    facet_globalid.reserve(nfacet);

    // Indices for convenience.
    const int facet_ridgeadr = 2;
    const int ridge_localid = 2 + nfacet;

    // Fill in graph data.
    Real norm;
    int i = adr = 0;
    ok = 1;
    FORALLfacets {
      facet_globalid[facet->id] = i;

      // Check constraint at facet.
      if (eqn(facet, SETfirstt_(facet->vertices, vertexT)->point) > kEpsEqn) {
        throw std::runtime_error("Incorrect normal and offset at facet");
      }

      // Set normal and offset.
      const Real sign =
          eqn(facet, qh->interior_point) <= 0.0 ? Real(1.0) : Real(-1.0);
      for (int k = 0; k < 3; ++k) normal[i](k) = sign * Real(facet->normal[k]);
      offset[i] = sign * Real(facet->offset);
      norm = normal[i].norm();
      if (norm < kEps) throw std::runtime_error("Zero normal vector at facet");
      normal[i] = normal[i] / norm;
      offset[i] = offset[i] / norm;

      // Save ridge address of this facet.
      graph[facet_ridgeadr + i] = adr;

      // Set facet neighbours.
      int cnt = 0;
      FOREACHsetelement_(facetT, facet->neighbors, facet1) {
        // Insert facet id.
        graph[ridge_localid + adr] = facet1->id;
        ++cnt;
        ++adr;
      }
      if (cnt < 3) {
        throw std::runtime_error("Facet with less than three neighbours");
      }

      // Insert separator, advance to next facet.
      graph[ridge_localid + adr] = -1;
      ++adr;
      ++i;
    }

    // Size check: SHOULD NOT OCCUR.
    if (i != nfacet || adr != nfacet + 2 * nridge) {
      throw std::runtime_error("Wrong size in convex hull graph");
    }

    // Free all.
    qh_freeqhull(qh, !qh_ALL);
    qh_memfreeshort(qh, &curlong, &totlong);

    // Bad graph: delete.
    if (!ok) {
      normal.clear();
      offset.clear();
      graph.clear();
      return false;
    }

    // Replace global ids with local ids in facet data.
    for (int i = 0; i < nfacet + 2 * nridge; ++i) {
      int& fid = graph[ridge_localid + i];
      if (fid >= 0) {
        // Make sure we found a match: SHOULD NOT OCCUR.
        const auto it = facet_globalid.find(fid);
        if (it == facet_globalid.end()) {
          throw std::runtime_error("Facet id not found in convex hull");
        }

        fid = it->second;
      }
    }

    // Reorder facet neighbours according to CCW orientation.
    int nn;
    Vec3r n, n1, t1, t2;
    for (int i = 0; i < nfacet; ++i) {
      n = normal[i];
      adr = ridge_localid + graph[facet_ridgeadr + i];

      // Number of neighbours.
      if (i < nfacet - 1) {
        nn = graph[facet_ridgeadr + i + 1] - graph[facet_ridgeadr + i] - 1;
      } else {
        nn = nfacet + 2 * nridge - graph[facet_ridgeadr + i] - 1;
      }

      // Check for adjacent parallel facets.
      for (int j = 0; j < nn; ++j) {
        n1 = normal[graph[adr + j]];
        if (std::abs(n1.dot(n)) > Real(1.0) - kEps) {
          throw std::runtime_error("Adjacent facets are (anti)parallel");
        }
      }

      // Compute tangent vectors at facet.
      t1 = n.cross(normal[graph[adr]]).normalized();
      t2 = n.cross(t1);

      // Sort neighbours.
      auto compare = [&normal, &t1, &t2](int j, int k) -> bool {
        const Real xj = normal[j].dot(t1), yj = normal[j].dot(t2);
        const Real xk = normal[k].dot(t1), yk = normal[k].dot(t2);

        bool uj = (yj > Real(0.0)) || (yj == Real(0.0) && xj >= Real(0.0));
        bool uk = (yk > Real(0.0)) || (yk == Real(0.0) && xk >= Real(0.0));
        if (uj != uk) return uj;
        return (xj * yk > xk * yj);
      };

      std::sort(graph.begin() + adr, graph.begin() + adr + nn, compare);
    }
  }

  // longjmp error handler.
  else {
    // Free all.
    qh_freeqhull(qh, !qh_ALL);
    qh_memfreeshort(qh, &curlong, &totlong);
    normal.clear();
    offset.clear();
    graph.clear();

    throw std::runtime_error("Qhull error");
  }

  return true;
}

Real MeshLoader::ComputeInradius(const std::vector<Vec3r>& normal,
                                 const std::vector<Real>& offset,
                                 const Vec3r& interior_point) const {
  Real max = -kInf;
  Real eqn;
  for (int i = 0; i < static_cast<int>(normal.size()); ++i) {
    eqn = normal[i].dot(interior_point) + offset[i];
    if (eqn >= Real(0.0)) {
      throw std::runtime_error("Point is not in the polytope interior");
    }
    max = std::max(max, eqn);
  }

  return -max;
}

Real MeshLoader::ComputeInradius(Vec3r& interior_point, bool use_given_ip) {
  std::vector<Vec3r> normal;
  std::vector<Real> offset;
  std::vector<int> graph;
  Vec3r interior_point_qh;
  const bool valid{MakeFacetGraph(normal, offset, graph, interior_point_qh)};
  if (!valid) throw std::runtime_error("Qhull error");

  if (!use_given_ip) interior_point = interior_point_qh;
  return ComputeInradius(normal, offset, interior_point);
}

}  // namespace dgd
