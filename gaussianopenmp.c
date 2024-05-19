#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h> 

void freeMatrix(float **A, int n) {
    for(int i = 0; i < n; i++) {
        free(A[i]);
    }
    free(A);
}

int main() {
    int i, j, k, n;
    float **A, *x, c;
    clock_t start, end;
    double cpu_time_used;

    printf("\nEnter the order of the matrix: ");
    scanf("%d", &n);

    A = (float**)malloc(n * sizeof(float*));
    for(i = 0; i < n; i++) {
        A[i] = (float*)malloc((n + 1) * sizeof(float));
    }
    x = (float*)malloc(n * sizeof(float));

    if(A == NULL || x == NULL) {
        printf("Memory allocation failed.\n");
        return -1;
    }

    srand(time(NULL));
    start = clock();

    #pragma omp parallel for private(i, j) shared(A, n)
    for(i = 0; i < n; i++) {
        for(j = 0; j <= n; j++) {
            A[i][j] = ((float)rand() / RAND_MAX) * 99.9f + 0.1f;
        }
    }


    for(j = 0; j < n; j++) {
        if(A[j][j] == 0.0f) {
            printf("Mathematical Error!");
            freeMatrix(A, n);
            free(x);
            return -1;
        }

        #pragma omp parallel for private(i, k) shared(A, n, j) schedule(dynamic, 1)
        for(i = j + 1; i < n; i++) {
            c = A[i][j] / A[j][j];
            for(k = j; k <= n; k++) {
                A[i][k] -= c * A[j][k];
            }
        }
    }

    // Sequential backward substitution
    for(i = n - 1; i >= 0; i--) {
        float sum = 0.0;
        for(j = i + 1; j < n; j++) {
            sum += A[i][j] * x[j];
        }
        x[i] = (A[i][n] - sum) / A[i][i];
    }

    end = clock();

    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("\nThe solution is:\n");
    for(i = 0; i < n; i++) {
        printf("x[%d] = %f\n", i, x[i]);
    }

    printf("Execution time: %f seconds\n", cpu_time_used);

    freeMatrix(A, n);
    free(x);

    return 0;
}
