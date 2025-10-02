/*
    cuda_aco_kernels.cuh - Optimized CUDA kernels for ACO
    
    Advanced optimizations:
    - Shared memory usage for frequently accessed data
    - Warp-level primitives for reductions
    - Texture memory for distance matrix
    - Coalesced memory access patterns
    - Cooperative groups for flexible synchronization
*/

#ifndef CUDA_ACO_KERNELS_CUH
#define CUDA_ACO_KERNELS_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cooperative_groups.h>
#include <cub/cub.cuh>

namespace cg = cooperative_groups;

#define WARP_SIZE 32
#define TILE_SIZE 32
#define MAX_SHARED_CITIES 512
#define CACHE_LINE_SIZE 128

// Texture memory declarations for cached read-only access
texture<float, 2, cudaReadModeElementType> tex_distance;
texture<float, 2, cudaReadModeElementType> tex_pheromone;

// Constant memory for frequently used parameters
__device__ __constant__ float d_alpha;
__device__ __constant__ float d_beta;
__device__ __constant__ float d_rho;
__device__ __constant__ float d_q0;
__device__ __constant__ int d_n_cities;
__device__ __constant__ int d_nn_size;

// Optimized data structures
typedef struct {
    int* nn_list;       // Nearest neighbor list for each city
    int nn_size;        // Size of nearest neighbor list
    float* nn_distances; // Pre-computed distances to nearest neighbors
} NNList;

typedef struct {
    float* probs;       // Probability array
    float* cumsum;      // Cumulative sum for roulette wheel
    int* candidates;    // Candidate cities
    int n_candidates;   // Number of candidates
} AntWorkspace;

// ==================== Utility Functions ====================

// Fast power function using intrinsics
__device__ __forceinline__ float fast_pow(float base, float exp) {
    return __expf(exp * __logf(base));
}

// Warp-level reduction for minimum
__device__ __forceinline__ float warp_reduce_min(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val = fminf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Warp-level reduction for sum
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level reduction using CUB
template<int BLOCK_SIZE>
__device__ float block_reduce_min(float val) {
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    return BlockReduce(temp_storage).Reduce(val, cub::Min());
}

// ==================== Optimized Ant Construction ====================

// Kernel for initializing ant memory with coalesced access
__global__ void init_ants_coalesced(Ant* ants, int n_ants, int n_cities,
                                    curandState* rand_states) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    // Coalesced initialization of tours and visited arrays
    for (int ant_id = tid; ant_id < n_ants; ant_id += stride) {
        Ant* ant = &ants[ant_id];
        curandState* state = &rand_states[ant_id];
        
        // Random starting city
        int start = curand(state) % n_cities;
        ant->tour[0] = start;
        ant->current_city = start;
        ant->tour_length = 0.0f;
        ant->tour_size = 1;
        
        // Initialize visited array (coalesced writes)
        for (int i = 0; i < n_cities; i++) {
            ant->visited[i] = (i == start);
            if (i > 0) ant->tour[i] = -1;
        }
    }
}

// High-performance ant solution construction with shared memory
__global__ void construct_solutions_shared(
    Ant* ants, float* pheromone, float* heuristic, 
    NNList* nn_lists, curandState* rand_states,
    int n_ants, int n_cities, int nn_size) {
    
    // Shared memory allocation
    extern __shared__ char shared_mem[];
    float* s_pheromone_cache = (float*)shared_mem;
    float* s_heuristic_cache = (float*)&s_pheromone_cache[MAX_SHARED_CITIES];
    float* s_probs = (float*)&s_heuristic_cache[MAX_SHARED_CITIES];
    
    int ant_id = blockIdx.x;
    if (ant_id >= n_ants) return;
    
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    Ant* ant = &ants[ant_id];
    curandState* rand_state = &rand_states[ant_id];
    
    // Cooperative groups for flexible synchronization
    cg::thread_block block = cg::this_thread_block();
    
    for (int step = 1; step < n_cities; step++) {
        int current = ant->current_city;
        
        // Load pheromone and heuristic data to shared memory
        for (int i = tid; i < n_cities; i += block_size) {
            s_pheromone_cache[i] = pheromone[current * n_cities + i];
            s_heuristic_cache[i] = heuristic[current * n_cities + i];
        }
        block.sync();
        
        // Thread 0 performs selection
        if (tid == 0) {
            float q = curand_uniform(rand_state);
            int next = -1;
            
            if (q < d_q0) {
                // Exploitation: choose best
                float max_val = -1.0f;
                
                // Check nearest neighbors first
                int* nn_current = &nn_lists[current].nn_list[current * nn_size];
                for (int j = 0; j < nn_size; j++) {
                    int city = nn_current[j];
                    if (!ant->visited[city]) {
                        float tau = s_pheromone_cache[city];
                        float eta = s_heuristic_cache[city];
                        float val = fast_pow(tau, d_alpha) * fast_pow(eta, d_beta);
                        
                        if (val > max_val) {
                            max_val = val;
                            next = city;
                        }
                    }
                }
                
                // Fall back to all cities if needed
                if (next == -1) {
                    for (int city = 0; city < n_cities; city++) {
                        if (!ant->visited[city]) {
                            float tau = s_pheromone_cache[city];
                            float eta = s_heuristic_cache[city];
                            float val = fast_pow(tau, d_alpha) * fast_pow(eta, d_beta);
                            
                            if (val > max_val) {
                                max_val = val;
                                next = city;
                            }
                        }
                    }
                }
            } else {
                // Exploration: probabilistic selection
                float sum = 0.0f;
                
                // Calculate probabilities in parallel
                for (int city = 0; city < n_cities; city++) {
                    if (!ant->visited[city]) {
                        float tau = s_pheromone_cache[city];
                        float eta = s_heuristic_cache[city];
                        s_probs[city] = fast_pow(tau, d_alpha) * fast_pow(eta, d_beta);
                        sum += s_probs[city];
                    } else {
                        s_probs[city] = 0.0f;
                    }
                }
                
                // Roulette wheel selection
                if (sum > 0) {
                    float r = curand_uniform(rand_state) * sum;
                    float cumsum = 0.0f;
                    
                    for (int city = 0; city < n_cities; city++) {
                        cumsum += s_probs[city];
                        if (cumsum >= r) {
                            next = city;
                            break;
                        }
                    }
                }
            }
            
            // Update ant state
            if (next >= 0) {
                ant->tour[step] = next;
                ant->visited[next] = true;
                ant->tour_length += tex2D(tex_distance, next, current);
                ant->current_city = next;
                ant->tour_size++;
            }
        }
        
        block.sync();
    }
    
    // Complete tour
    if (tid == 0) {
        ant->tour_length += tex2D(tex_distance, ant->tour[0], ant->current_city);
    }
}

// ==================== Optimized Pheromone Updates ====================

// Coalesced pheromone evaporation
__global__ void evaporate_pheromone_coalesced(float* pheromone, int n, float rho) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int total = n * n;
    
    // Process multiple elements per thread for better utilization
    for (int idx = tid; idx < total; idx += stride) {
        pheromone[idx] *= (1.0f - rho);
    }
}

// Atomic-free pheromone deposit using temporary buffers
__global__ void deposit_pheromone_buffered(
    float* pheromone, int* all_tours, float* deposits,
    int n_ants, int n_cities) {
    
    extern __shared__ float s_deposits[];
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int edge_id = tid;
    int total_edges = n_cities * n_cities;
    
    if (edge_id < total_edges) {
        float total_deposit = 0.0f;
        int from_city = edge_id / n_cities;
        int to_city = edge_id % n_cities;
        
        // Sum deposits from all ants for this edge
        for (int ant = 0; ant < n_ants; ant++) {
            int* tour = &all_tours[ant * n_cities];
            float deposit = deposits[ant];
            
            // Check if this edge is in the ant's tour
            for (int i = 0; i < n_cities; i++) {
                int tour_from = tour[i];
                int tour_to = tour[(i + 1) % n_cities];
                
                if (tour_from == from_city && tour_to == to_city) {
                    total_deposit += deposit;
                }
                // For undirected graphs
                if (tour_from == to_city && tour_to == from_city) {
                    total_deposit += deposit;
                }
            }
        }
        
        // Apply deposit
        pheromone[edge_id] += total_deposit;
    }
}

// ==================== Advanced Local Search ====================

// GPU-accelerated 2-opt with early termination
__global__ void two_opt_early_termination(
    int* tours, float* tour_lengths, int n_cities, int n_ants,
    int max_no_improve) {
    
    int ant_id = blockIdx.x;
    if (ant_id >= n_ants) return;
    
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    int* tour = &tours[ant_id * n_cities];
    float* length = &tour_lengths[ant_id];
    
    __shared__ bool improved;
    __shared__ int no_improve_count;
    __shared__ float best_improvement;
    __shared__ int best_i, best_j;
    
    if (tid == 0) {
        no_improve_count = 0;
    }
    __syncthreads();
    
    while (no_improve_count < max_no_improve) {
        if (tid == 0) {
            improved = false;
            best_improvement = 0.0f;
            best_i = -1;
            best_j = -1;
        }
        __syncthreads();
        
        // Parallel search for improvements
        for (int i = tid; i < n_cities - 2; i += block_size) {
            for (int j = i + 2; j < n_cities; j++) {
                if ((j + 1) % n_cities == i) continue;
                
                float current_dist = tex2D(tex_distance, tour[i], tour[i+1]) +
                                   tex2D(tex_distance, tour[j], tour[(j+1) % n_cities]);
                float new_dist = tex2D(tex_distance, tour[i], tour[j]) +
                               tex2D(tex_distance, tour[i+1], tour[(j+1) % n_cities]);
                float improvement = current_dist - new_dist;
                
                if (improvement > 0.001f) {
                    atomicMax((int*)&best_improvement, __float_as_int(improvement));
                    if (__int_as_float(atomicAdd((int*)&best_improvement, 0)) == improvement) {
                        best_i = i;
                        best_j = j;
                        improved = true;
                    }
                }
            }
        }
        __syncthreads();
        
        // Apply best improvement
        if (tid == 0 && improved) {
            // Reverse tour segment
            int start = best_i + 1;
            int end = best_j;
            while (start < end) {
                int temp = tour[start];
                tour[start] = tour[end];
                tour[end] = temp;
                start++;
                end--;
            }
            *length -= best_improvement;
            no_improve_count = 0;
        } else if (tid == 0) {
            no_improve_count++;
        }
        __syncthreads();
    }
}

// Parallel Or-opt local search
__global__ void or_opt_ls(int* tours, float* tour_lengths, int n_cities,
                          int n_ants, int segment_size) {
    int ant_id = blockIdx.x;
    if (ant_id >= n_ants) return;
    
    int tid = threadIdx.x;
    int* tour = &tours[ant_id * n_cities];
    
    __shared__ bool improved;
    __shared__ float best_delta;
    __shared__ int best_i, best_j;
    
    do {
        if (tid == 0) {
            improved = false;
            best_delta = 0.0f;
        }
        __syncthreads();
        
        // Parallel examination of segment moves
        for (int i = tid; i < n_cities; i += blockDim.x) {
            for (int j = 0; j < n_cities - segment_size; j++) {
                if (abs(i - j) < segment_size + 1) continue;
                
                // Calculate cost of moving segment from position i to j
                float remove_cost = 0.0f;
                float insert_cost = 0.0f;
                
                // Cost of removing segment
                remove_cost += tex2D(tex_distance, tour[(i-1+n_cities)%n_cities], tour[i]);
                remove_cost += tex2D(tex_distance, tour[(i+segment_size-1)%n_cities], 
                                   tour[(i+segment_size)%n_cities]);
                remove_cost -= tex2D(tex_distance, tour[(i-1+n_cities)%n_cities], 
                                   tour[(i+segment_size)%n_cities]);
                
                // Cost of inserting segment
                insert_cost += tex2D(tex_distance, tour[j], tour[i]);
                insert_cost += tex2D(tex_distance, tour[(i+segment_size-1)%n_cities], 
                                   tour[(j+1)%n_cities]);
                insert_cost -= tex2D(tex_distance, tour[j], tour[(j+1)%n_cities]);
                
                float delta = remove_cost - insert_cost;
                
                if (delta > 0.001f) {
                    atomicMax((int*)&best_delta, __float_as_int(delta));
                    if (__int_as_float(atomicAdd((int*)&best_delta, 0)) == delta) {
                        best_i = i;
                        best_j = j;
                        improved = true;
                    }
                }
            }
        }
        __syncthreads();
        
        // Apply best move
        if (tid == 0 && improved) {
            // Move segment from position best_i to best_j
            int temp[MAX_CITIES];
            for (int k = 0; k < segment_size; k++) {
                temp[k] = tour[(best_i + k) % n_cities];
            }
            
            // Shift elements
            if (best_j < best_i) {
                for (int k = best_i - 1; k >= best_j + 1; k--) {
                    tour[(k + segment_size) % n_cities] = tour[k % n_cities];
                }
                for (int k = 0; k < segment_size; k++) {
                    tour[(best_j + 1 + k) % n_cities] = temp[k];
                }
            } else {
                for (int k = best_i + segment_size; k <= best_j; k++) {
                    tour[(k - segment_size) % n_cities] = tour[k % n_cities];
                }
                for (int k = 0; k < segment_size; k++) {
                    tour[(best_j - segment_size + 1 + k) % n_cities] = temp[k];
                }
            }
            
            tour_lengths[ant_id] -= best_delta;
        }
        __syncthreads();
        
    } while (improved);
}

// ==================== QAP-Specific Kernels ====================

// Fast QAP solution evaluation using shared memory
__global__ void evaluate_qap_shared(int* solutions, float* flow, float* distance,
                                   float* costs, int n, int n_solutions) {
    extern __shared__ float s_mem[];
    float* s_flow = s_mem;
    float* s_dist = &s_mem[TILE_SIZE * TILE_SIZE];
    
    int sol_id = blockIdx.z;
    if (sol_id >= n_solutions) return;
    
    int* solution = &solutions[sol_id * n];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x * TILE_SIZE;
    int by = blockIdx.y * TILE_SIZE;
    
    float sum = 0.0f;
    
    // Process tiles
    for (int k = 0; k < (n + TILE_SIZE - 1) / TILE_SIZE; k++) {
        // Load tiles to shared memory
        int flow_row = by + ty;
        int flow_col = k * TILE_SIZE + tx;
        if (flow_row < n && flow_col < n) {
            s_flow[ty * TILE_SIZE + tx] = flow[flow_row * n + flow_col];
        } else {
            s_flow[ty * TILE_SIZE + tx] = 0.0f;
        }
        
        int dist_row = k * TILE_SIZE + ty;
        int dist_col = bx + tx;
        if (dist_row < n && dist_col < n) {
            int loc1 = solution[dist_row];
            int loc2 = solution[dist_col];
            s_dist[ty * TILE_SIZE + tx] = distance[loc1 * n + loc2];
        } else {
            s_dist[ty * TILE_SIZE + tx] = 0.0f;
        }
        __syncthreads();
        
        // Compute partial products
        #pragma unroll
        for (int i = 0; i < TILE_SIZE; i++) {
            sum += s_flow[ty * TILE_SIZE + i] * s_dist[i * TILE_SIZE + tx];
        }
        __syncthreads();
    }
    
    // Reduce and store result
    atomicAdd(&costs[sol_id], sum);
}

// Parallel tabu search for QAP
__global__ void qap_tabu_search(int* solutions, float* flow, float* distance,
                               int* tabu_list, int n, int n_solutions,
                               int tabu_tenure, int max_iters) {
    int sol_id = blockIdx.x;
    if (sol_id >= n_solutions) return;
    
    int tid = threadIdx.x;
    int* solution = &solutions[sol_id * n];
    int* tabu = &tabu_list[sol_id * n * n];
    
    extern __shared__ int s_best_move[];
    int* s_tabu_count = &s_best_move[3];  // i, j, iteration
    
    for (int iter = 0; iter < max_iters; iter++) {
        if (tid == 0) {
            s_best_move[0] = -1;  // best_i
            s_best_move[1] = -1;  // best_j
            s_best_move[2] = INT_MAX;  // best_delta
        }
        __syncthreads();
        
        // Evaluate all possible swaps in parallel
        int total_swaps = (n * (n - 1)) / 2;
        for (int swap_id = tid; swap_id < total_swaps; swap_id += blockDim.x) {
            // Convert swap_id to (i, j) pair
            int i = 0, j = 0;
            int k = swap_id;
            for (i = 0; i < n - 1; i++) {
                if (k < n - i - 1) {
                    j = i + 1 + k;
                    break;
                }
                k -= (n - i - 1);
            }
            
            // Check tabu status
            if (tabu[i * n + j] > iter) continue;
            
            // Calculate delta for swap
            float delta = 0.0f;
            for (int k = 0; k < n; k++) {
                if (k != i && k != j) {
                    delta += (flow[i * n + k] - flow[j * n + k]) *
                            (distance[solution[j] * n + solution[k]] -
                             distance[solution[i] * n + solution[k]]);
                    delta += (flow[k * n + i] - flow[k * n + j]) *
                            (distance[solution[k] * n + solution[j]] -
                             distance[solution[k] * n + solution[i]]);
                }
            }
            
            // Update best move
            if (delta < s_best_move[2]) {
                atomicMin(&s_best_move[2], __float_as_int(delta));
                if (__int_as_float(s_best_move[2]) == delta) {
                    s_best_move[0] = i;
                    s_best_move[1] = j;
                }
            }
        }
        __syncthreads();
        
        // Apply best move
        if (tid == 0 && s_best_move[0] >= 0) {
            int i = s_best_move[0];
            int j = s_best_move[1];
            
            // Swap
            int temp = solution[i];
            solution[i] = solution[j];
            solution[j] = temp;
            
            // Update tabu list
            tabu[i * n + j] = iter + tabu_tenure;
            tabu[j * n + i] = iter + tabu_tenure;
        }
        __syncthreads();
    }
}

// ==================== Helper Functions ====================

// Initialize texture memory
void init_textures(float* d_distance, float* d_pheromone, int n_cities) {
    // Bind distance matrix to texture
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaBindTexture2D(0, tex_distance, d_distance, channelDesc,
                     n_cities, n_cities, n_cities * sizeof(float));
    
    // Bind pheromone matrix to texture
    cudaBindTexture2D(0, tex_pheromone, d_pheromone, channelDesc,
                     n_cities, n_cities, n_cities * sizeof(float));
}

// Set constant memory parameters
void set_parameters(float alpha, float beta, float rho, float q0, 
                    int n_cities, int nn_size) {
    cudaMemcpyToSymbol(d_alpha, &alpha, sizeof(float));
    cudaMemcpyToSymbol(d_beta, &beta, sizeof(float));
    cudaMemcpyToSymbol(d_rho, &rho, sizeof(float));
    cudaMemcpyToSymbol(d_q0, &q0, sizeof(float));
    cudaMemcpyToSymbol(d_n_cities, &n_cities, sizeof(int));
    cudaMemcpyToSymbol(d_nn_size, &nn_size, sizeof(int));
}

// Compute optimal block and grid dimensions
dim3 get_optimal_dims(int problem_size, int max_threads_per_block = 256) {
    int block_size = min(max_threads_per_block, problem_size);
    int grid_size = (problem_size + block_size - 1) / block_size;
    return dim3(grid_size, 1, 1);
}

#endif // CUDA_ACO_KERNELS_CUH