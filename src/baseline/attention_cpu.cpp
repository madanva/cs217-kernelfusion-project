// CPU Baseline: Scaled Dot-Product Attention
// Double-precision reference implementation for speedup comparison
// Computes: Attention(Q, K, V) = softmax(QK^T / sqrt(d)) * V
// Measures wall-clock time for N=16 and N=64, d=16

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <time.h>

static const int D = 16;  // head dimension (matches kVectorSize)

// Fill matrix with deterministic int8-range values (same pattern as FPGA testbenches)
void fill_matrix(double* M, int rows, int cols, int seed) {
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            M[i * cols + j] = (double)(((seed + i * cols + j) * 7 + 13) % 256 - 128);
}

// Unfused attention: 3 separate phases (matches FPGA unfused variant)
void attention_unfused(const double* Q, const double* K, const double* V,
                       double* O, int N) {
    double inv_sqrt_d = 1.0 / sqrt((double)D);

    // Phase 1: S = QK^T / sqrt(d)  [N x N]
    double* S = (double*)malloc(N * N * sizeof(double));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            double dot = 0.0;
            for (int k = 0; k < D; k++)
                dot += Q[i * D + k] * K[j * D + k];
            S[i * N + j] = dot * inv_sqrt_d;
        }

    // Phase 2: P = softmax(S) row-wise  [N x N]
    double* P = (double*)malloc(N * N * sizeof(double));
    for (int i = 0; i < N; i++) {
        double row_max = S[i * N + 0];
        for (int j = 1; j < N; j++)
            if (S[i * N + j] > row_max) row_max = S[i * N + j];
        double row_sum = 0.0;
        for (int j = 0; j < N; j++) {
            P[i * N + j] = exp(S[i * N + j] - row_max);
            row_sum += P[i * N + j];
        }
        for (int j = 0; j < N; j++)
            P[i * N + j] /= row_sum;
    }

    // Phase 3: O = P * V  [N x D]
    for (int i = 0; i < N; i++)
        for (int k = 0; k < D; k++) {
            double acc = 0.0;
            for (int j = 0; j < N; j++)
                acc += P[i * N + j] * V[j * D + k];
            O[i * D + k] = acc;
        }

    free(S);
    free(P);
}

// Fused attention: online softmax + V accumulation (matches FPGA fully fused variant)
void attention_fused(const double* Q, const double* K, const double* V,
                     double* O, int N) {
    double inv_sqrt_d = 1.0 / sqrt((double)D);
    int T = 16;  // tile size (matches FPGA tile)

    for (int i = 0; i < N; i++) {
        double v_accum[D];
        memset(v_accum, 0, sizeof(v_accum));
        double running_max = -1e30;
        double running_sum = 0.0;

        int num_tiles = (N + T - 1) / T;
        for (int t = 0; t < num_tiles; t++) {
            int tile_start = t * T;
            int tile_end = (tile_start + T < N) ? tile_start + T : N;
            int tile_len = tile_end - tile_start;

            // Compute tile logits
            double tile_logits[16];
            for (int j = 0; j < tile_len; j++) {
                double dot = 0.0;
                for (int k = 0; k < D; k++)
                    dot += Q[i * D + k] * K[(tile_start + j) * D + k];
                tile_logits[j] = dot * inv_sqrt_d;
            }

            // Online softmax update
            double tile_max = tile_logits[0];
            for (int j = 1; j < tile_len; j++)
                if (tile_logits[j] > tile_max) tile_max = tile_logits[j];

            double new_max = (running_max > tile_max) ? running_max : tile_max;
            double correction = exp(running_max - new_max);

            // Correct previous accumulation
            for (int k = 0; k < D; k++)
                v_accum[k] *= correction;
            running_sum *= correction;

            // Compute tile exp and accumulate V
            double tile_exp[16];
            for (int j = 0; j < tile_len; j++) {
                tile_exp[j] = exp(tile_logits[j] - new_max);
                running_sum += tile_exp[j];
            }
            for (int j = 0; j < tile_len; j++)
                for (int k = 0; k < D; k++)
                    v_accum[k] += tile_exp[j] * V[(tile_start + j) * D + k];

            running_max = new_max;
        }

        // Normalize
        for (int k = 0; k < D; k++)
            O[i * D + k] = v_accum[k] / running_sum;
    }
}

double time_diff_us(struct timespec start, struct timespec end) {
    return (end.tv_sec - start.tv_sec) * 1e6 + (end.tv_nsec - start.tv_nsec) / 1e3;
}

void benchmark(int N, int num_iters) {
    double* Q = (double*)malloc(N * D * sizeof(double));
    double* K = (double*)malloc(N * D * sizeof(double));
    double* V = (double*)malloc(N * D * sizeof(double));
    double* O = (double*)malloc(N * D * sizeof(double));

    fill_matrix(Q, N, D, 42);
    fill_matrix(K, N, D, 137);
    fill_matrix(V, N, D, 271);

    struct timespec t0, t1;

    // Warmup
    attention_unfused(Q, K, V, O, N);
    attention_fused(Q, K, V, O, N);

    // Benchmark unfused
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < num_iters; i++)
        attention_unfused(Q, K, V, O, N);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double unfused_us = time_diff_us(t0, t1) / num_iters;

    // Benchmark fused
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < num_iters; i++)
        attention_fused(Q, K, V, O, N);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double fused_us = time_diff_us(t0, t1) / num_iters;

    printf("N=%d, d=%d:\n", N, D);
    printf("  Unfused:  %.2f us/iter (%d iters)\n", unfused_us, num_iters);
    printf("  Fused:    %.2f us/iter (%d iters)\n", fused_us, num_iters);

    // Compare with FPGA at 125 MHz (8ns clock)
    // From scverify: sim_clocks measured at tb.clk=1ns
    // FPGA runs at 125 MHz = 8ns/cycle
    if (N == 16) {
        double fpga_unfused_us = 5672 * 0.008;   // 5672 cycles * 8ns
        double fpga_fused_us   = 4392 * 0.008;   // 4392 cycles * 8ns
        printf("  FPGA Unfused @125MHz: %.2f us (%.1fx speedup vs CPU unfused)\n",
               fpga_unfused_us, unfused_us / fpga_unfused_us);
        printf("  FPGA Fused @125MHz:   %.2f us (%.1fx speedup vs CPU fused)\n",
               fpga_fused_us, fused_us / fpga_fused_us);
    } else if (N == 64) {
        double fpga_unfused_us = 84296 * 0.008;  // 84296 cycles * 8ns
        double fpga_fused_us   = 66454 * 0.008;  // 66454 cycles * 8ns
        printf("  FPGA Unfused @125MHz: %.2f us (%.1fx speedup vs CPU unfused)\n",
               fpga_unfused_us, unfused_us / fpga_unfused_us);
        printf("  FPGA Fused @125MHz:   %.2f us (%.1fx speedup vs CPU fused)\n",
               fpga_fused_us, fused_us / fpga_fused_us);
    }

    // Verify correctness: unfused vs fused should match
    double* O_unfused = (double*)malloc(N * D * sizeof(double));
    double* O_fused = (double*)malloc(N * D * sizeof(double));
    attention_unfused(Q, K, V, O_unfused, N);
    attention_fused(Q, K, V, O_fused, N);
    double max_err = 0.0;
    for (int i = 0; i < N * D; i++) {
        double err = fabs(O_unfused[i] - O_fused[i]);
        if (err > max_err) max_err = err;
    }
    printf("  Max unfused-vs-fused error: %.2e\n\n", max_err);

    free(Q); free(K); free(V); free(O);
    free(O_unfused); free(O_fused);
}

int main() {
    printf("=== CPU Baseline: Scaled Dot-Product Attention ===\n");
    printf("Double-precision floating point on host CPU\n\n");

    benchmark(16, 100000);
    benchmark(64, 10000);

    return 0;
}
