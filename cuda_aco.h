/*
    cuda_aco.h - Header file for CUDA ACO implementation
    Shared between CUDA and C++ files
*/

#ifndef CUDA_ACO_H
#define CUDA_ACO_H

#ifdef __cplusplus
extern "C" {
#endif

// Forward declaration of opaque ACO data structure
typedef struct ACOData ACOData;

// ACO algorithm types
typedef enum {
    AS,    // Ant System
    EAS,   // Elitist Ant System
    MMAS,  // MAX-MIN Ant System
    RAS,   // Rank-based Ant System
    ACS,   // Ant Colony System
    BWAS   // Best-Worst Ant System
} ACOAlgorithm;

// Problem type
typedef enum {
    TSP,   // Traveling Salesman Problem
    QAP    // Quadratic Assignment Problem
} ProblemType;

// Public API functions
ACOData* aco_init(int n_cities, int n_ants, ACOAlgorithm algo, ProblemType prob);
void aco_load_problem(ACOData* aco, float* distance_matrix, float* flow_matrix);
void aco_run(ACOData* aco, int max_iterations, float* best_tour, float* best_length);
void aco_cleanup(ACOData* aco);

#ifdef __cplusplus
}
#endif

#endif // CUDA_ACO_H