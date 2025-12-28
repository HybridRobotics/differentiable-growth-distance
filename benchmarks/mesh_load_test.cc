#include <iostream>
#include <vector>

#include "dgd/data_types.h"
#include "internal_helpers/filesystem_utils.h"
#include "internal_helpers/mesh_loader.h"

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <asset_folder_path>" << std::endl;
    return EXIT_FAILURE;
  }

  std::string path = argv[1];
  if (!dgd::bench::IsValidDirectory(path)) return EXIT_FAILURE;

  const auto filenames = dgd::bench::GetObjFileNames(path);
  if (filenames.empty()) {
    std::cerr << "No .obj files found in the specified directory" << std::endl;
    return EXIT_FAILURE;
  }

  dgd::bench::MeshProperties mp;
  for (const auto& filename : filenames) {
    try {
      mp.SetVertexMeshFromObjFile(filename);
      std::cout << "[Success] " << filename << std::endl
                << "  #vertices: " << mp.nvert << std::endl;
    } catch (const std::runtime_error& e) {
      std::cerr << "[Error] Qhull error with file: '" << filename
                << "': " << e.what() << std::endl;
    }
  }

  return EXIT_SUCCESS;
}
