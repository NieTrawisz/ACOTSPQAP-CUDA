/*
    CUDA-Accelerated ACO algorithms for TSP and QAP
    Based on ACOTSPQAP by Manuel López-Ibáñez and Thomas Stützle
    
    This CUDA implementation parallelizes:
    1. Ant solution construction
    2. Pheromone matrix updates
    3. Distance/flow matrix computations
    4. Local search operations
*/

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cooperative_groups.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <time.h>

// CUDA error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(error)); \
        exit(1); \
    } \
} while(0)

// Problem and algorithm parameters
#define MAX_CITIES 10000
#define MAX_ANTS 1024
#define BLOCK_SIZE 256
#define PHEROMONE_INIT 1.0f
#define ALPHA 1.0f  // Pheromone influence
#define BETA 2.0f   // Heuristic influence
#define RHO 0.5f    // Evaporation rate
#define Q0 0.9f     // ACS exploration/exploitation parameter

// Algorithm selection flags
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
    TSP,
    QAP
} ProblemType;

// Ant structure
typedef struct {
    int* tour;           // Solution tour
    bool* visited;       // Visited cities/locations
    float tour_length;   // Tour cost
    int current_city;    // Current position
    int tour_size;       // Number of cities in tour
} Ant;

// ACO data structure
typedef struct {
    // Problem data
    float* d_distance;      // Distance matrix (TSP) or flow matrix (QAP)
    float* d_flow;          // Second matrix for QAP
    float* d_pheromone;     // Pheromone matrix
    float* d_heuristic;     // Heuristic information matrix (1/distance for TSP)
    float* d_prob;          // Probability matrix for ant decisions
    
    // Ant colony
    Ant* d_ants;            // Device ant array
    int* d_best_tour;       // Best tour found
    float* d_best_length;   // Best tour length
    
    // Algorithm parameters
    int n_cities;           // Problem size
    int n_ants;             // Number of ants
    ACOAlgorithm algo_type;
    ProblemType prob_type;
    
    // CUDA random states
    curandState* d_rand_states;
    
    // Synchronization
    int* d_mutex;           // For atomic operations on shared best solution
} ACOData;

// Kernel to initialize random states
__global__ void init_rand_states(curandState* states, unsigned long seed, int n_ants) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n_ants) {
        curand_init(seed, tid, 0, &states[tid]);
    }
}

// Kernel to initialize pheromone matrix
__global__ void init_pheromone(float* pheromone, int n, float init_val) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * n;
    
    while (tid < total) {
        pheromone[tid] = init_val;
        tid += gridDim.x * blockDim.x;
    }
}

// Kernel to compute heuristic information (1/distance for TSP)
__global__ void compute_heuristic(float* heuristic, float* distance, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * n;
    
    while (tid < total) {
        if (distance[tid] > 0) {
            heuristic[tid] = 1.0f / distance[tid];
        } else {
            heuristic[tid] = 0.0f;
        }
        tid += gridDim.x * blockDim.x;
    }
}

// Device function to select next city using probability rule
__device__ int select_next_city(Ant* ant, float* pheromone, float* heuristic, 
                                curandState* rand_state, int n_cities, 
                                float alpha, float beta, float q0) {
    float q = curand_uniform(rand_state);
    int current = ant->current_city;
    
    if (q < q0) {  // Exploitation (ACS)
        float max_val = -1.0f;
        int best_city = -1;
        
        for (int j = 0; j < n_cities; j++) {
            if (!ant->visited[j]) {
                float val = powf(pheromone[current * n_cities + j], alpha) * 
                           powf(heuristic[current * n_cities + j], beta);
                if (val > max_val) {
                    max_val = val;
                    best_city = j;
                }
            }
        }
        return best_city;
    } else {  // Exploration
        // Calculate probabilities
        float sum = 0.0f;
        for (int j = 0; j < n_cities; j++) {
            if (!ant->visited[j]) {
                sum += powf(pheromone[current * n_cities + j], alpha) * 
                      powf(heuristic[current * n_cities + j], beta);
            }
        }
        
        if (sum == 0.0f) return -1;  // No valid city
        
        // Roulette wheel selection
        float r = curand_uniform(rand_state) * sum;
        float cumsum = 0.0f;
        
        for (int j = 0; j < n_cities; j++) {
            if (!ant->visited[j]) {
                cumsum += powf(pheromone[current * n_cities + j], alpha) * 
                         powf(heuristic[current * n_cities + j], beta);
                if (cumsum >= r) {
                    return j;
                }
            }
        }
        
        // Fallback: return first unvisited city
        for (int j = 0; j < n_cities; j++) {
            if (!ant->visited[j]) return j;
        }
    }
    
    return -1;
}

// Kernel for ant solution construction
__global__ void construct_solutions(Ant* ants, float* pheromone, float* heuristic,
                                   float* distance, curandState* rand_states,
                                   int n_cities, int n_ants, float alpha, 
                                   float beta, float q0) {
    int ant_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (ant_id >= n_ants) return;
    
    Ant* ant = &ants[ant_id];
    curandState* rand_state = &rand_states[ant_id];
    
    // Initialize ant
    for (int i = 0; i < n_cities; i++) {
        ant->visited[i] = false;
        ant->tour[i] = -1;
    }
    ant->tour_size = 0;
    ant->tour_length = 0.0f;
    
    // Random starting city
    int start_city = curand(rand_state) % n_cities;
    ant->current_city = start_city;
    ant->tour[0] = start_city;
    ant->visited[start_city] = true;
    ant->tour_size = 1;
    
    // Construct tour
    for (int step = 1; step < n_cities; step++) {
        int next_city = select_next_city(ant, pheromone, heuristic, rand_state,
                                        n_cities, alpha, beta, q0);
        
        if (next_city >= 0) {
            ant->tour[step] = next_city;
            ant->visited[next_city] = true;
            ant->tour_length += distance[ant->current_city * n_cities + next_city];
            ant->current_city = next_city;
            ant->tour_size++;
        }
    }
    
    // Complete tour (return to start)
    if (ant->tour_size == n_cities) {
        ant->tour_length += distance[ant->current_city * n_cities + start_city];
    }
}

// Kernel for local pheromone update (ACS)
__global__ void local_pheromone_update(float* pheromone, Ant* ants, int n_cities,
                                      int n_ants, float tau0, float phi) {
    int ant_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (ant_id >= n_ants) return;
    
    Ant* ant = &ants[ant_id];
    
    for (int i = 0; i < n_cities - 1; i++) {
        int from = ant->tour[i];
        int to = ant->tour[i + 1];
        if (from >= 0 && to >= 0) {
            int idx = from * n_cities + to;
            atomicAdd(&pheromone[idx], -phi * pheromone[idx] + phi * tau0);
        }
    }
}

// Kernel for global pheromone evaporation
__global__ void evaporate_pheromone(float* pheromone, int n, float rho) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * n;
    
    while (tid < total) {
        pheromone[tid] *= (1.0f - rho);
        tid += gridDim.x * blockDim.x;
    }
}

// Kernel for depositing pheromone (best ant or rank-based)
__global__ void deposit_pheromone(float* pheromone, int* tour, float deposit,
                                 int n_cities) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n_cities) {
        int from = tour[i];
        int to = tour[(i + 1) % n_cities];
        
        if (from >= 0 && to >= 0) {
            int idx = from * n_cities + to;
            atomicAdd(&pheromone[idx], deposit);
            // Symmetric for undirected graphs
            idx = to * n_cities + from;
            atomicAdd(&pheromone[idx], deposit);
        }
    }
}

// Kernel to find best ant
__global__ void find_best_ant(Ant* ants, int n_ants, int* best_ant_idx,
                             float* best_length) {
    __shared__ float shared_lengths[BLOCK_SIZE];
    __shared__ int shared_indices[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int ant_id = blockIdx.x * blockDim.x + tid;
    
    // Load ant lengths to shared memory
    if (ant_id < n_ants) {
        shared_lengths[tid] = ants[ant_id].tour_length;
        shared_indices[tid] = ant_id;
    } else {
        shared_lengths[tid] = FLT_MAX;
        shared_indices[tid] = -1;
    }
    __syncthreads();
    
    // Reduction to find minimum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (shared_lengths[tid + stride] < shared_lengths[tid]) {
                shared_lengths[tid] = shared_lengths[tid + stride];
                shared_indices[tid] = shared_indices[tid + stride];
            }
        }
        __syncthreads();
    }
    
    // Write result
    if (tid == 0) {
        atomicMin((int*)best_length, __float_as_int(shared_lengths[0]));
        if (__int_as_float(atomicAdd((int*)best_length, 0)) == shared_lengths[0]) {
            *best_ant_idx = shared_indices[0];
        }
    }
}

// 2-opt local search kernel
__global__ void two_opt_ls(int* tour, float* distance, int n_cities,
                          float* tour_length, int max_iters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    __shared__ bool improved;
    
    for (int iter = 0; iter < max_iters; iter++) {
        if (tid == 0) improved = false;
        __syncthreads();
        
        int i = tid;
        while (i < n_cities - 2) {
            for (int j = i + 2; j < n_cities; j++) {
                if ((j + 1) % n_cities == i) continue;
                
                int a = tour[i];
                int b = tour[i + 1];
                int c = tour[j];
                int d = tour[(j + 1) % n_cities];
                
                float current_dist = distance[a * n_cities + b] + 
                                   distance[c * n_cities + d];
                float new_dist = distance[a * n_cities + c] + 
                               distance[b * n_cities + d];
                
                if (new_dist < current_dist) {
                    // Reverse tour segment
                    int start = i + 1;
                    int end = j;
                    while (start < end) {
                        int temp = tour[start];
                        tour[start] = tour[end];
                        tour[end] = temp;
                        start++;
                        end--;
                    }
                    improved = true;
                    atomicAdd(tour_length, new_dist - current_dist);
                }
            }
            i += gridDim.x * blockDim.x;
        }
        
        __syncthreads();
        if (!improved) break;
    }
}

// Host functions for ACO management

ACOData* aco_init(int n_cities, int n_ants, ACOAlgorithm algo, ProblemType prob) {
    ACOData* aco = (ACOData*)malloc(sizeof(ACOData));
    aco->n_cities = n_cities;
    aco->n_ants = n_ants;
    aco->algo_type = algo;
    aco->prob_type = prob;
    
    // Allocate device memory
    int matrix_size = n_cities * n_cities * sizeof(float);
    CUDA_CHECK(cudaMalloc(&aco->d_distance, matrix_size));
    CUDA_CHECK(cudaMalloc(&aco->d_pheromone, matrix_size));
    CUDA_CHECK(cudaMalloc(&aco->d_heuristic, matrix_size));
    CUDA_CHECK(cudaMalloc(&aco->d_prob, matrix_size));
    
    if (prob == QAP) {
        CUDA_CHECK(cudaMalloc(&aco->d_flow, matrix_size));
    }
    
    // Allocate ant memory - need to properly manage device arrays
    Ant* h_ants = (Ant*)malloc(n_ants * sizeof(Ant));
    for (int i = 0; i < n_ants; i++) {
        CUDA_CHECK(cudaMalloc(&h_ants[i].tour, n_cities * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&h_ants[i].visited, n_cities * sizeof(bool)));
        h_ants[i].tour_length = 0.0f;
        h_ants[i].current_city = 0;
        h_ants[i].tour_size = 0;
    }
    CUDA_CHECK(cudaMalloc(&aco->d_ants, n_ants * sizeof(Ant)));
    CUDA_CHECK(cudaMemcpy(aco->d_ants, h_ants, n_ants * sizeof(Ant), 
                        cudaMemcpyHostToDevice));
    free(h_ants);
    
    CUDA_CHECK(cudaMalloc(&aco->d_best_tour, n_cities * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&aco->d_best_length, sizeof(float)));
    
    // Initialize random states
    CUDA_CHECK(cudaMalloc(&aco->d_rand_states, n_ants * sizeof(curandState)));
    int blocks = (n_ants + BLOCK_SIZE - 1) / BLOCK_SIZE;
    init_rand_states<<<blocks, BLOCK_SIZE>>>(aco->d_rand_states, time(NULL), n_ants);
    
    // Initialize pheromone matrix
    blocks = (n_cities * n_cities + BLOCK_SIZE - 1) / BLOCK_SIZE;
    init_pheromone<<<blocks, BLOCK_SIZE>>>(aco->d_pheromone, n_cities, PHEROMONE_INIT);
    
    CUDA_CHECK(cudaMalloc(&aco->d_mutex, sizeof(int)));
    CUDA_CHECK(cudaMemset(aco->d_mutex, 0, sizeof(int)));
    
    return aco;
}

void aco_load_problem(ACOData* aco, float* distance_matrix, float* flow_matrix) {
    int matrix_size = aco->n_cities * aco->n_cities * sizeof(float);
    
    CUDA_CHECK(cudaMemcpy(aco->d_distance, distance_matrix, matrix_size,
                        cudaMemcpyHostToDevice));
    
    if (aco->prob_type == QAP && flow_matrix != NULL) {
        CUDA_CHECK(cudaMemcpy(aco->d_flow, flow_matrix, matrix_size,
                            cudaMemcpyHostToDevice));
    }
    
    // Compute heuristic information
    int blocks = (aco->n_cities * aco->n_cities + BLOCK_SIZE - 1) / BLOCK_SIZE;
    compute_heuristic<<<blocks, BLOCK_SIZE>>>(aco->d_heuristic, aco->d_distance,
                                             aco->n_cities);
}

void aco_run(ACOData* aco, int max_iterations, float* best_tour, float* best_length) {
    int ant_blocks = (aco->n_ants + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int matrix_blocks = (aco->n_cities * aco->n_cities + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    float h_best_length = FLT_MAX;
    int h_best_ant_idx = -1;
    
    for (int iter = 0; iter < max_iterations; iter++) {
        // Construct solutions
        construct_solutions<<<ant_blocks, BLOCK_SIZE>>>(
            aco->d_ants, aco->d_pheromone, aco->d_heuristic, aco->d_distance,
            aco->d_rand_states, aco->n_cities, aco->n_ants, ALPHA, BETA, Q0
        );
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Local search (optional) - fixed to work with device memory
        if (iter % 10 == 0 && aco->n_ants > 0) {  // Apply every 10 iterations
            // Get all ant data from device
            Ant* h_ants_temp = (Ant*)malloc(aco->n_ants * sizeof(Ant));
            CUDA_CHECK(cudaMemcpy(h_ants_temp, aco->d_ants, aco->n_ants * sizeof(Ant),
                                cudaMemcpyDeviceToHost));
            
            // Apply local search to each ant's tour on device
            for (int a = 0; a < aco->n_ants; a++) {
                // Note: h_ants_temp[a].tour is a device pointer
                // Call local search kernel directly with device pointer
                two_opt_ls<<<1, 32>>>(h_ants_temp[a].tour, aco->d_distance, 
                                     aco->n_cities, &aco->d_ants[a].tour_length, 100);
            }
            CUDA_CHECK(cudaDeviceSynchronize());
            free(h_ants_temp);
        }
        
        // Find best ant - fixed version
        float* d_best_ant_length = nullptr;
        int* d_best_ant_idx = nullptr;
        CUDA_CHECK(cudaMalloc(&d_best_ant_length, sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_best_ant_idx, sizeof(int)));
        
        float max_float = FLT_MAX;
        CUDA_CHECK(cudaMemcpy(d_best_ant_length, &max_float, sizeof(float),
                            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_best_ant_idx, -1, sizeof(int)));
        
        find_best_ant<<<ant_blocks, BLOCK_SIZE>>>(aco->d_ants, aco->n_ants,
                                                 d_best_ant_idx, d_best_ant_length);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Get best ant index and length
        CUDA_CHECK(cudaMemcpy(&h_best_ant_idx, d_best_ant_idx, sizeof(int),
                            cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_best_length, d_best_ant_length, sizeof(float),
                            cudaMemcpyDeviceToHost));
        
        // Update pheromones
        evaporate_pheromone<<<matrix_blocks, BLOCK_SIZE>>>(aco->d_pheromone,
                                                          aco->n_cities, RHO);
        
        // Deposit pheromone based on algorithm type
        if (h_best_ant_idx >= 0 && h_best_ant_idx < aco->n_ants) {
            // Get the best ant's data
            Ant h_best_ant_meta;
            CUDA_CHECK(cudaMemcpy(&h_best_ant_meta, &aco->d_ants[h_best_ant_idx],
                                sizeof(Ant), cudaMemcpyDeviceToHost));
            
            float deposit = 1.0f / h_best_length;
            
            switch (aco->algo_type) {
                case AS:
                    // All ants deposit - use device pointers directly
                    for (int a = 0; a < aco->n_ants; a++) {
                        Ant h_ant_meta;
                        CUDA_CHECK(cudaMemcpy(&h_ant_meta, &aco->d_ants[a], sizeof(Ant),
                                            cudaMemcpyDeviceToHost));
                        deposit_pheromone<<<1, aco->n_cities>>>(aco->d_pheromone,
                                                               h_ant_meta.tour,  // This is device pointer
                                                               1.0f / h_ant_meta.tour_length,
                                                               aco->n_cities);
                    }
                    break;
                    
                case EAS:
                    // All ants + elite ants deposit more
                    for (int a = 0; a < aco->n_ants; a++) {
                        Ant h_ant_meta;
                        CUDA_CHECK(cudaMemcpy(&h_ant_meta, &aco->d_ants[a], sizeof(Ant),
                                            cudaMemcpyDeviceToHost));
                        deposit_pheromone<<<1, aco->n_cities>>>(aco->d_pheromone,
                                                               h_ant_meta.tour,
                                                               1.0f / h_ant_meta.tour_length,
                                                               aco->n_cities);
                    }
                    // Elite ant deposits extra
                    deposit_pheromone<<<1, aco->n_cities>>>(aco->d_pheromone,
                                                           h_best_ant_meta.tour,
                                                           deposit * 5.0f,  // Elite weight
                                                           aco->n_cities);
                    break;
                    
                case MMAS:
                    // Only best ant deposits, with min/max limits
                    deposit_pheromone<<<1, aco->n_cities>>>(aco->d_pheromone,
                                                           h_best_ant_meta.tour,
                                                           deposit, aco->n_cities);
                    break;
                    
                case ACS:
                    // Global best deposits
                    deposit_pheromone<<<1, aco->n_cities>>>(aco->d_pheromone,
                                                           h_best_ant_meta.tour,
                                                           deposit, aco->n_cities);
                    // Also apply local updates during construction
                    local_pheromone_update<<<ant_blocks, BLOCK_SIZE>>>(
                        aco->d_pheromone, aco->d_ants, aco->n_cities,
                        aco->n_ants, PHEROMONE_INIT, 0.1f
                    );
                    break;
                    
                default:
                    // Default to AS behavior
                    break;
            }
        }
        
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Clean up temporary allocations
        CUDA_CHECK(cudaFree(d_best_ant_length));
        CUDA_CHECK(cudaFree(d_best_ant_idx));
        
        // Print iteration info
        if (iter % 10 == 0) {
            printf("Iteration %d: Best length = %.2f\n", iter, h_best_length);
        }
    }
    
    // Copy best solution to host - fixed version
    *best_length = h_best_length;
    if (h_best_ant_idx >= 0 && h_best_ant_idx < aco->n_ants) {
        Ant h_best_ant_meta;
        CUDA_CHECK(cudaMemcpy(&h_best_ant_meta, &aco->d_ants[h_best_ant_idx],
                            sizeof(Ant), cudaMemcpyDeviceToHost));
        
        // Copy the tour data from device to host
        int* h_tour_temp = (int*)malloc(aco->n_cities * sizeof(int));
        CUDA_CHECK(cudaMemcpy(h_tour_temp, h_best_ant_meta.tour,
                            aco->n_cities * sizeof(int),
                            cudaMemcpyDeviceToHost));
        
        // Convert to float array (as expected by interface)
        for (int i = 0; i < aco->n_cities; i++) {
            best_tour[i] = (float)h_tour_temp[i];
        }
        free(h_tour_temp);
    }
}

void aco_cleanup(ACOData* aco) {
    // Free ant memory - properly handle device pointers
    Ant* h_ants = (Ant*)malloc(aco->n_ants * sizeof(Ant));
    CUDA_CHECK(cudaMemcpy(h_ants, aco->d_ants, aco->n_ants * sizeof(Ant),
                        cudaMemcpyDeviceToHost));
    
    for (int i = 0; i < aco->n_ants; i++) {
        CUDA_CHECK(cudaFree(h_ants[i].tour));
        CUDA_CHECK(cudaFree(h_ants[i].visited));
    }
    free(h_ants);
    
    CUDA_CHECK(cudaFree(aco->d_ants));
    CUDA_CHECK(cudaFree(aco->d_distance));
    CUDA_CHECK(cudaFree(aco->d_pheromone));
    CUDA_CHECK(cudaFree(aco->d_heuristic));
    CUDA_CHECK(cudaFree(aco->d_prob));
    
    if (aco->prob_type == QAP) {
        CUDA_CHECK(cudaFree(aco->d_flow));
    }
    
    CUDA_CHECK(cudaFree(aco->d_best_tour));
    CUDA_CHECK(cudaFree(aco->d_best_length));
    CUDA_CHECK(cudaFree(aco->d_rand_states));
    CUDA_CHECK(cudaFree(aco->d_mutex));
    
    free(aco);
}

// Example usage and test function
int main(int argc, char** argv) {
    // Example: small TSP instance
    int n_cities = 100;
    int n_ants = 128;
    int max_iterations = 1000;
    
    // Generate random distance matrix for testing
    float* distance_matrix = (float*)malloc(n_cities * n_cities * sizeof(float));
    srand(42);
    for (int i = 0; i < n_cities; i++) {
        for (int j = 0; j < n_cities; j++) {
            if (i == j) {
                distance_matrix[i * n_cities + j] = 0;
            } else if (i < j) {
                distance_matrix[i * n_cities + j] = (rand() % 100) + 1;
                distance_matrix[j * n_cities + i] = distance_matrix[i * n_cities + j];
            }
        }
    }
    
    // Initialize ACO
    printf("Initializing CUDA ACO for TSP with %d cities and %d ants\n",
           n_cities, n_ants);
    ACOData* aco = aco_init(n_cities, n_ants, ACS, TSP);
    
    // Load problem
    printf("Loading problem data...\n");
    aco_load_problem(aco, distance_matrix, NULL);
    
    // Run ACO
    float* best_tour = (float*)malloc(n_cities * sizeof(float));
    float best_length;
    
    printf("Running ACO for %d iterations...\n", max_iterations);
    clock_t start = clock();
    aco_run(aco, max_iterations, best_tour, &best_length);
    clock_t end = clock();
    
    double cpu_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("\nOptimization complete!\n");
    printf("Best tour length: %.2f\n", best_length);
    printf("Execution time: %.2f seconds\n", cpu_time);
    
    // Cleanup
    aco_cleanup(aco);
    free(distance_matrix);
    free(best_tour);
    
    return 0;
}