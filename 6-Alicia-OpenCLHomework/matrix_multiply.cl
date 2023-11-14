__kernel void matrix_multiply(__global const float* A, __global const float* B, __global float* C,
                             int M, int N, int K) {
    int row = get_global_id(0);
    int col = get_global_id(1);

    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        float a = A[k * M + row];
        float b = B[col * K + k];
        sum += a * b;
    }

    C[col * M + row] = sum;
}

// Kernel A: matrix_multiply_A
__kernel void matrix_multiply_A(__global const float* A, __global const float* B, __global float* C,
                             int M, int N, int K) {
    int row = get_global_id(0);
    int col = get_global_id(1);

    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        float a = A[k * M + row];
        float b = B[col * K + k];
        sum += a * b;
    }

    C[col * M + row] = sum;
}

// Kernel B: matrix_multiply_B will be less optimized than Kernel A.
__kernel void matrix_multiply_B(__global const float* A, __global const float* B, __global float* C,
                             int M, int N, int K) {
    int row = get_global_id(0);
    int col = get_global_id(1);

    float sum = 0.0f;
    for (int i = 0; i < M; i++) {
        sum += A[i * M + row] * B[col * K + i];
    }

    C[col * M + row] = sum;
}