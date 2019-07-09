#include <chrono>
#include <cstring>
#include <thread>

#include <mpi.h>

#include "solver.hpp"
#include "../matrix/matrix.hpp"

using std::memcpy;

using std::this_thread::sleep_for;
using std::chrono::microseconds;

void solveSeq(int rows, int cols, int iterations, double td, double h, int sleep, double ** matrix) {
    double c, l, r, t, b;
    
    double h_square = h * h;

    double * linePrevBuffer = new double[cols];
    double * lineCurrBuffer = new double[cols];

    for(int k = 0; k < iterations; k++) {

        memcpy(linePrevBuffer, matrix[0], cols * sizeof(double));
        for(int i = 1; i < rows - 1; i++) {

            memcpy(lineCurrBuffer, matrix[i], cols * sizeof(double));
            for(int j = 1; j < cols - 1; j++) {
                c = lineCurrBuffer[j];
                t = linePrevBuffer[j];
                b = matrix[i + 1][j];
                l = lineCurrBuffer[j - 1];
                r = lineCurrBuffer[j + 1];


                sleep_for(microseconds(sleep));
                matrix[i][j] = c * (1.0 - 4.0 * td / h_square) + (t + b + l + r) * (td / h_square);
            }

            memcpy(linePrevBuffer, lineCurrBuffer, cols * sizeof(double));
        }
    }
}

void solvePar(int threads, int rows, int cols, int iterations, double td, double h, int sleep, double ** matrix) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double c, l, r, t, b;

    double h_square = h * h;

    double * linePrevBuffer = new double[cols];
    double * lineCurrBuffer = new double[cols];
    double input[8];
    double output[4];

  if (rank == 0) {
    for(int k = 0; k < iterations; k++) {

      memcpy(linePrevBuffer, matrix[0], cols * sizeof(double));
      for(int i = 1; i < rows - 1; i++) {

        memcpy(lineCurrBuffer, matrix[i], cols * sizeof(double));
        for(int j = 1; j < cols - 1; j++) {
          c = lineCurrBuffer[j];
          t = linePrevBuffer[j];
          b = matrix[i + 1][j];
          l = lineCurrBuffer[j - 1];
          r = lineCurrBuffer[j + 1];

          input[0] = (double) k;
          input[1] = (double) i;
          input[2] = (double) j;
          input[3] = c;
          input[4] = l;
          input[5] = r;
          input[6] = t;
          input[7] = b;

          MPI_Send(&input,8, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD)

          sleep_for(microseconds(sleep));
         // matrix[i][j] = c * (1.0 - 4.0 * td / h_square) + (t + b + l + r) * (td / h_square);
        }

        memcpy(linePrevBuffer, lineCurrBuffer, cols * sizeof(double));
      }
    }
    for (int i = 0; i < cols * rows; i++) {
      MPI_Recv(&output, 4, MPI_DOUBLE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      matrix[(int) output[1]][(int) output[2]][(int) output[0]] = output[3];
    }
  }

    if(0 != rank) {
      MPI_Recv(&input, 8, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      output[0] = input[0];
      output[1] = input[1];
      output[2] = input[2];
      sleep_for(microseconds(sleep));
      output[3] = input[3] * (1.0 - 4.0 * td / h_square) + (input[6] + input[7] + input[4] + input[5]) * (td / h_square);
      MPI_Send(&output, 4, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
      deallocateMatrix(rows, matrix);
    }

}
