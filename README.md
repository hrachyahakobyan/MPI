# MPI
Implementations of algorithms with MPI


# laplace.c

 A program to solve 2D laplace equation in a rectangle using Jacobi convergence method
 The program uses rowwise domain decomposition to achieve parallel computing
 The rectangle is not limited to a square
 The code does not require any modification to run on differnt number of processors
 The number of actual processes must match the number specified in the program.
