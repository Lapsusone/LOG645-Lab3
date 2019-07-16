#include <gtest/gtest.h>
#include "matrix.hpp"

TEST(fillMatrix, InitialValues)
{
  double **matrix = allocateMatrix(10, 10);
  for (int i = 0; i < 10; ++i)
  {
    for (int j = 0; j < 10; ++j)
    {
      matrix[i][j] = -55;
      EXPECT_EQ(matrix[i][j], -55);
    }
  }
  fillMatrix(10, 10, matrix);
  for (int i = 0; i < 10; ++i)
  {
    for (int j = 0; j < 10; ++j)
    {
      EXPECT_NE(matrix[i][j], -55);
    }
  }
  deallocateMatrix(10, matrix);
}
