#include <chrono>
#include <cstring>
#include <thread>

#include <mpi.h>

#include "solver.hpp"
#include "../matrix/matrix.hpp"

using std::memcpy;

using std::chrono::microseconds;
using std::this_thread::sleep_for;

void solveSeq(int rows, int cols, int iterations, double td, double h, int sleep, double **matrix)
{
  double c, l, r, t, b;

  double h_square = h * h;

  double *linePrevBuffer = new double[cols];
  double *lineCurrBuffer = new double[cols];

  for (int k = 0; k < iterations; k++)
  {

    memcpy(linePrevBuffer, matrix[0], cols * sizeof(double));
    for (int i = 1; i < rows - 1; i++)
    {

      memcpy(lineCurrBuffer, matrix[i], cols * sizeof(double));
      for (int j = 1; j < cols - 1; j++)
      {
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

void solvePar(int threads, int rows, int cols, int iterations, double td, double h, int sleep, double **matrix)
{
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  double c, l, r, t, b;

  double h_square = h * h;

  double *linePrevBuffer = new double[cols];
  double *lineCurrBuffer = new double[cols];
  double *toSlaves = new double[8];
  double *toMaster = new double[4];

  int thread_rank;
  if (rank == 0)
  {
    for (int k = 0; k < iterations; k++)
    {

      memcpy(linePrevBuffer, matrix[0], cols * sizeof(double));
      for (int i = 1; i < rows - 1; i++)
      {

        memcpy(lineCurrBuffer, matrix[i], cols * sizeof(double));
        for (int j = 1; j < cols - 1; j++)
        {
          c = lineCurrBuffer[j];
          t = linePrevBuffer[j];
          b = matrix[i + 1][j];
          l = lineCurrBuffer[j - 1];
          r = lineCurrBuffer[j + 1];

          toSlaves[0] = (double)k;
          toSlaves[1] = (double)i;
          toSlaves[2] = (double)j;
          toSlaves[3] = c;
          toSlaves[4] = l;
          toSlaves[5] = r;
          toSlaves[6] = t;
          toSlaves[7] = b;

          thread_rank = ++rank % threads;

          MPI_Send(&toSlaves, 8, MPI_DOUBLE, thread_rank, 1, MPI_COMM_WORLD);
        }

        memcpy(linePrevBuffer, lineCurrBuffer, cols * sizeof(double));
      }
    }
    for (int i = 0; i < cols * rows; i++)
    {
      MPI_Recv(&toMaster, 4, MPI_DOUBLE, MPI_ANY_SOURCE, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      matrix[(int)toMaster[1]][(int)toMaster[2]] = toMaster[3];
    }
  }
  //MPI_Barrier(MPI_COMM_WORLD);

  if (0 != rank)
  {
    MPI_Recv(&toSlaves, 8, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    toMaster[0] = toSlaves[0];
    toMaster[1] = toSlaves[1];
    toMaster[2] = toSlaves[2];
    sleep_for(microseconds(sleep));
    toMaster[3] = toSlaves[3] * (1.0 - 4.0 * td / h_square) + (toSlaves[6] + toSlaves[7] + toSlaves[4] + toSlaves[5]) * (td / h_square);
    MPI_Send(&toMaster, 4, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
  }

  MPI_Finalize();
}
