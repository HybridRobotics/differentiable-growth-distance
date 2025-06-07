#include <iostream>
#include <vector>

#include "dgd/data_types.h"
#include "internal_helpers/mesh_loader.h"
#include "internal_helpers/utils.h"

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <asset_folder_path>" << std::endl;
    return EXIT_FAILURE;
  }

  std::string path = argv[1];
  if (!dgd::internal::IsValidDirectory(path)) return EXIT_FAILURE;

  std::vector<std::string> filenames;
  dgd::internal::GetObjFileNames(path, filenames);
  if (filenames.empty()) {
    std::cerr << "No .obj files found in the specified directory" << std::endl;
    return EXIT_FAILURE;
  }

  dgd::internal::MeshProperties mp;
  for (const auto& filename : filenames) {
    try {
      dgd::internal::SetVertexMeshFromObjFile(filename, mp);
      std::cout << "[Success] " << filename << std::endl;
    } catch (const std::runtime_error& e) {
      std::cerr << "[Error] Qhull error with file: '" << filename
                << "': " << e.what() << std::endl;
    }
  }

  return EXIT_SUCCESS;
}
