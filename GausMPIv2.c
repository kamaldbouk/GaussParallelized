#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

void freeMatrix(float **A, int n) {
	for (int i = 0; i < n; i++) {
		free(A[i]);
	}
	free(A);
}

void printUpperTriangular(float **A, int n) {
	printf("\nUpper Triangular Matrix:\n");
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			printf("%f\t", A[i][j]);
		}
		printf("\n");
	}
}

int main() {
	int i, j, k, n;
	float **A, *x, c, sum;
	clock_t start, end;
	double cpu_time_used;
	int rank, size;

	MPI_Init(NULL, NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (rank == 0) {
		printf("\nEnter the order of the matrix: ");
		fflush(stdout); 
		scanf("%d", &n);
	}

	MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

	// Dynamically allocate memory for the augmented matrix A[n][n+1] and vector x[n]
	A = (float**)malloc(n * sizeof(float*));
	
	for (i = 0; i < n; i++) {
		A[i] = (float*)malloc((n + 1) * sizeof(float));
	}

	x = (float*)malloc(n * sizeof(float));

	// Check if memory allocation was successful
	
	if (A == NULL || x == NULL) {
		printf("Memory allocation failed.\n");
		MPI_Finalize();
		return -1;
	}

	srand(time(NULL)); // Seed the random number generator

	// Fill the augmented matrix with random floating-point numbers
	for (i = 0; i < n; i++) {
		for (j = 0; j <= n; j++) {
			A[i][j] = ((float)rand() / RAND_MAX) * 99.9f + 0.1f; // Range: [0.1, 100.0]
		}
	}

	start = clock();

	// Parallelized Gaussian elimination to form upper triangular matrix
	for (j = 0; j < n; j++) {
		MPI_Bcast(A[j], n + 1, MPI_FLOAT, 0, MPI_COMM_WORLD); // Broadcast the pivot row
		for (i = j + 1 + rank; i < n; i += size) {
			if (A[j][j] == 0.0f) {
				printf("Mathematical Error!");
				freeMatrix(A, n);
				free(x);
				MPI_Finalize();
				return -1;
			}

			c = A[i][j] / A[j][j];
			for (k = j; k <= n; k++) {
				A[i][k] -= c * A[j][k];
			}
		}
	}

	// Backward substitution    
	for(i = n - 1; i >= 0; i--) {
		sum = 0.0;
		for(j = i + 1; j < n; j++) {
			sum += A[i][j] * x[j];
		}

		x[i] = (A[i][n] - sum) / A[i][i];
	}
    
	if (rank == 0) {
		end = clock();

		cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;

		printf("\nThe solution is:\n");
		for (i = 0; i < n; i++) {
			printf("x[%d] = %f\n", i, x[i]);
		}

		//printUpperTriangular(A, n);

		printf("Execution time: %f seconds\n", cpu_time_used);
	}

	freeMatrix(A, n);
	free(x);

	MPI_Finalize();

	return 0;
}
