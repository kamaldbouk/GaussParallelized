%%writefile gaus.cu

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define TILE_SIZE 16

__global__ void makeBelowPivotZero(double *a, int n, int row) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > row && i < n && j < n + 1) {
        double pivotValue = a[row * (n + 1) + row];
        if (fabs(pivotValue) > 1e-8) {
            double zeroFactor = a[i * (n + 1) + row] / pivotValue;
            a[i * (n + 1) + j] -= zeroFactor * a[row * (n + 1) + j];
        }
    }
}

void backSubstitution(double *a, double *x, int n) {
    for (int i = n - 1; i >= 0; i--) {
        x[i] = a[i * (n + 1) + n];
        for (int j = i + 1; j < n; j++) {
            x[i] -= a[i * (n + 1) + j] * x[j];
        }
        x[i] /= a[i * (n + 1) + i];
    }
}

int main() {
    int n;
    printf("Enter the order of the matrix: ");
    scanf("%d", &n);
    double *a = (double *)malloc(n * (n + 1) * sizeof(double));
    double *x = (double *)malloc(n * sizeof(double));

    // Initialize the matrix with random numbers
    srand(time(NULL));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= n; j++) {
            a[i * (n + 1) + j] = (double)(rand() % 100 + 1);  // Random numbers 1-100
        }
    }

    double *d_a;
    cudaMalloc((void**)&d_a, (n + 1) * n * sizeof(double));
    cudaMemcpy(d_a, a, (n + 1) * n * sizeof(double), cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    dim3 dimGrid((n + TILE_SIZE - 1) / TILE_SIZE, (n + TILE_SIZE - 1) / TILE_SIZE);

    clock_t start = clock();
    for (int i = 0; i < n; i++) {
        makeBelowPivotZero<<<dimGrid, dimBlock>>>(d_a, n, i);
        cudaDeviceSynchronize();
    }
    clock_t stop = clock();

    cudaMemcpy(a, d_a, (n + 1) * n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_a);

    backSubstitution(a, x, n);

    double time = ((double)(stop - start)) / CLOCKS_PER_SEC;
    printf("Execution Time: %.6f seconds\n", time);


    free(a);
    free(x);
    return 0;
}