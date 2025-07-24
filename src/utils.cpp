#include <cmath>
#include <complex>

#include <hpdmk.h>
#include <sctl.hpp>
#include <utils.hpp>

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace hpdmk {
    std::vector<std::vector<double>> read_particle_info(const std::string& filename) {
        std::vector<std::vector<double>> particles;
        std::ifstream infile(filename);
        std::string line;

        std::getline(infile, line);

        while (std::getline(infile, line)) {
            std::istringstream iss(line);
            double x, y, z, q;
            if (!(iss >> x >> y >> z >> q)) continue;
            particles.push_back({x, y, z, q});
        }
        return particles;
    }
}