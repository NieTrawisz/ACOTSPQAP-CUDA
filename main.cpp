/*
    main.cpp - Example usage of CUDA ACO implementation
    Includes TSPLIB and QAPLIB file readers
*/

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <cstring>
#include <chrono>
#include <iomanip>

// Forward declare the ACO data structure (opaque pointer)
struct ACOData;

// ACO algorithm types
enum ACOAlgorithm { AS, EAS, MMAS, RAS, ACS, BWAS };
enum ProblemType { TSP, QAP };

// External CUDA ACO functions (defined in cuda_aco_lib.cu)
extern "C" {
    ACOData* aco_init(int n_cities, int n_ants, ACOAlgorithm algo, ProblemType prob);
    void aco_load_problem(ACOData* aco, float* distance_matrix, float* flow_matrix);
    void aco_run(ACOData* aco, int max_iterations, float* best_tour, float* best_length);
    void aco_cleanup(ACOData* aco);
}

// Problem instance structure
struct ProblemInstance {
    std::string name;
    int dimension;
    std::vector<std::vector<float>> distance_matrix;
    std::vector<std::vector<float>> flow_matrix;  // For QAP
    std::vector<std::pair<float, float>> coordinates;  // For TSP
    float best_known;
};

// Command line arguments
struct Config {
    std::string problem_type = "tsp";
    std::string instance_file = "";
    std::string algorithm = "acs";
    int n_ants = 128;
    int max_iterations = 1000;
    float alpha = 1.0f;
    float beta = 2.0f;
    float rho = 0.5f;
    float q0 = 0.9f;
    bool quiet = false;
    bool use_ls = true;
    int ls_freq = 10;
    int seed = -1;
};

// Parse command line arguments
Config parse_args(int argc, char* argv[]) {
    Config config;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--problem" && i + 1 < argc) {
            config.problem_type = argv[++i];
        } else if (arg == "--instance" && i + 1 < argc) {
            config.instance_file = argv[++i];
        } else if (arg == "--algorithm" && i + 1 < argc) {
            config.algorithm = argv[++i];
        } else if (arg == "--ants" && i + 1 < argc) {
            config.n_ants = std::stoi(argv[++i]);
        } else if (arg == "--iterations" && i + 1 < argc) {
            config.max_iterations = std::stoi(argv[++i]);
        } else if (arg == "--alpha" && i + 1 < argc) {
            config.alpha = std::stof(argv[++i]);
        } else if (arg == "--beta" && i + 1 < argc) {
            config.beta = std::stof(argv[++i]);
        } else if (arg == "--rho" && i + 1 < argc) {
            config.rho = std::stof(argv[++i]);
        } else if (arg == "--q0" && i + 1 < argc) {
            config.q0 = std::stof(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            config.seed = std::stoi(argv[++i]);
        } else if (arg == "--quiet") {
            config.quiet = true;
        } else if (arg == "--no-ls") {
            config.use_ls = false;
        } else if (arg == "--help") {
            std::cout << "CUDA ACO - GPU-Accelerated Ant Colony Optimization\n";
            std::cout << "Usage: " << argv[0] << " [options]\n";
            std::cout << "Options:\n";
            std::cout << "  --problem <tsp|qap>      Problem type (default: tsp)\n";
            std::cout << "  --instance <file>        Instance file path\n";
            std::cout << "  --algorithm <type>       Algorithm: as, eas, mmas, ras, acs, bwas (default: acs)\n";
            std::cout << "  --ants <n>              Number of ants (default: 128)\n";
            std::cout << "  --iterations <n>        Max iterations (default: 1000)\n";
            std::cout << "  --alpha <val>           Pheromone influence (default: 1.0)\n";
            std::cout << "  --beta <val>            Heuristic influence (default: 2.0)\n";
            std::cout << "  --rho <val>             Evaporation rate (default: 0.5)\n";
            std::cout << "  --q0 <val>              ACS q0 parameter (default: 0.9)\n";
            std::cout << "  --seed <n>              Random seed\n";
            std::cout << "  --quiet                 Suppress output\n";
            std::cout << "  --no-ls                 Disable local search\n";
            exit(0);
        }
    }
    
    return config;
}

// Calculate Euclidean distance
float euclidean_distance(const std::pair<float, float>& p1, 
                         const std::pair<float, float>& p2) {
    float dx = p1.first - p2.first;
    float dy = p1.second - p2.second;
    return std::sqrt(dx * dx + dy * dy);
}

// Load TSP instance (TSPLIB format)
ProblemInstance load_tsp_instance(const std::string& filename) {
    ProblemInstance problem;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    std::string line;
    std::string edge_weight_type = "EUC_2D";
    
    // Parse header
    while (std::getline(file, line)) {
        if (line.find("NAME") != std::string::npos) {
            size_t pos = line.find(':');
            if (pos != std::string::npos) {
                problem.name = line.substr(pos + 1);
            }
        } else if (line.find("DIMENSION") != std::string::npos) {
            size_t pos = line.find(':');
            if (pos != std::string::npos) {
                problem.dimension = std::stoi(line.substr(pos + 1));
            }
        } else if (line.find("EDGE_WEIGHT_TYPE") != std::string::npos) {
            size_t pos = line.find(':');
            if (pos != std::string::npos) {
                edge_weight_type = line.substr(pos + 1);
            }
        } else if (line.find("NODE_COORD_SECTION") != std::string::npos) {
            break;
        }
    }
    
    // Read coordinates
    problem.coordinates.resize(problem.dimension);
    for (int i = 0; i < problem.dimension; i++) {
        int idx;
        float x, y;
        file >> idx >> x >> y;
        problem.coordinates[i] = {x, y};
    }
    
    // Calculate distance matrix
    problem.distance_matrix.resize(problem.dimension, 
                                  std::vector<float>(problem.dimension, 0.0f));
    
    for (int i = 0; i < problem.dimension; i++) {
        for (int j = 0; j < problem.dimension; j++) {
            if (i != j) {
                problem.distance_matrix[i][j] = euclidean_distance(
                    problem.coordinates[i], problem.coordinates[j]);
            }
        }
    }
    
    file.close();
    return problem;
}

// Load QAP instance (QAPLIB format)
ProblemInstance load_qap_instance(const std::string& filename) {
    ProblemInstance problem;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    // Read dimension
    file >> problem.dimension;
    
    // Read flow matrix
    problem.flow_matrix.resize(problem.dimension, 
                               std::vector<float>(problem.dimension));
    for (int i = 0; i < problem.dimension; i++) {
        for (int j = 0; j < problem.dimension; j++) {
            file >> problem.flow_matrix[i][j];
        }
    }
    
    // Read distance matrix
    problem.distance_matrix.resize(problem.dimension, 
                                  std::vector<float>(problem.dimension));
    for (int i = 0; i < problem.dimension; i++) {
        for (int j = 0; j < problem.dimension; j++) {
            file >> problem.distance_matrix[i][j];
        }
    }
    
    file.close();
    return problem;
}

// Convert algorithm string to enum
ACOAlgorithm get_algorithm_type(const std::string& algo) {
    if (algo == "as") return AS;
    if (algo == "eas") return EAS;
    if (algo == "mmas") return MMAS;
    if (algo == "ras") return RAS;
    if (algo == "acs") return ACS;
    if (algo == "bwas") return BWAS;
    return ACS;  // Default
}

// Print solution quality
void print_solution(const float* tour, int dimension, float length, 
                   float best_known, double time_ms) {
    std::cout << "\n=== Solution Summary ===" << std::endl;
    std::cout << "Tour length: " << std::fixed << std::setprecision(2) 
              << length << std::endl;
    
    if (best_known > 0) {
        float gap = ((length - best_known) / best_known) * 100.0f;
        std::cout << "Best known: " << best_known << std::endl;
        std::cout << "Gap: " << std::setprecision(2) << gap << "%" << std::endl;
    }
    
    std::cout << "Time: " << std::setprecision(2) << time_ms << " ms" << std::endl;
    std::cout << "Tour: ";
    for (int i = 0; i < std::min(10, dimension); i++) {
        std::cout << (int)tour[i] << " ";
    }
    if (dimension > 10) std::cout << "...";
    std::cout << std::endl;
}

// Main function
int main(int argc, char* argv[]) {
    try {
        // Parse arguments
        Config config = parse_args(argc, argv);
        
        if (config.instance_file.empty()) {
            std::cerr << "Error: No instance file provided. Use --instance <file>" 
                      << std::endl;
            return 1;
        }
        
        // Load problem instance
        ProblemInstance problem;
        ProblemType prob_type;
        
        if (!config.quiet) {
            std::cout << "Loading " << config.problem_type << " instance: " 
                      << config.instance_file << std::endl;
        }
        
        if (config.problem_type == "tsp") {
            problem = load_tsp_instance(config.instance_file);
            prob_type = TSP;
        } else if (config.problem_type == "qap") {
            problem = load_qap_instance(config.instance_file);
            prob_type = QAP;
        } else {
            std::cerr << "Error: Unknown problem type: " << config.problem_type 
                      << std::endl;
            return 1;
        }
        
        // Flatten matrices for CUDA
        float* distance_matrix = new float[problem.dimension * problem.dimension];
        float* flow_matrix = nullptr;
        
        for (int i = 0; i < problem.dimension; i++) {
            for (int j = 0; j < problem.dimension; j++) {
                distance_matrix[i * problem.dimension + j] = 
                    problem.distance_matrix[i][j];
            }
        }
        
        if (prob_type == QAP) {
            flow_matrix = new float[problem.dimension * problem.dimension];
            for (int i = 0; i < problem.dimension; i++) {
                for (int j = 0; j < problem.dimension; j++) {
                    flow_matrix[i * problem.dimension + j] = 
                        problem.flow_matrix[i][j];
                }
            }
        }
        
        // Initialize CUDA ACO
        if (!config.quiet) {
            std::cout << "Initializing CUDA ACO..." << std::endl;
            std::cout << "  Problem size: " << problem.dimension << std::endl;
            std::cout << "  Algorithm: " << config.algorithm << std::endl;
            std::cout << "  Ants: " << config.n_ants << std::endl;
            std::cout << "  Iterations: " << config.max_iterations << std::endl;
        }
        
        ACOAlgorithm algo = get_algorithm_type(config.algorithm);
        ACOData* aco = aco_init(problem.dimension, config.n_ants, algo, prob_type);
        
        // Load problem data
        aco_load_problem(aco, distance_matrix, flow_matrix);
        
        // Prepare for results
        float* best_tour = new float[problem.dimension];
        float best_length;
        
        // Run ACO
        if (!config.quiet) {
            std::cout << "Running optimization..." << std::endl;
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        
        aco_run(aco, config.max_iterations, best_tour, &best_length);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        // Print results
        if (!config.quiet) {
            print_solution(best_tour, problem.dimension, best_length, 
                          problem.best_known, duration.count());
        } else {
            // Quiet mode: just print essential info
            std::cout << problem.dimension << "," << config.n_ants << "," 
                      << best_length << "," << duration.count() << std::endl;
        }
        
        // Cleanup
        aco_cleanup(aco);
        delete[] distance_matrix;
        delete[] flow_matrix;
        delete[] best_tour;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}