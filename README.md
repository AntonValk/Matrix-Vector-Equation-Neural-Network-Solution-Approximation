# Matrix-Vector-Equation-Neural-Network-Solution-Approximation

A neural network implementation that predicts what the solution to a sparse matrix equation will be. The prediction is then fed into a solver (using methods such as Gauss-Seidel, Jacobi, etc.) as the initial guess solution vector. The initial guess is close to the solution and therefore the amount of iterations that the solver will have to complete is reduced.

## Dependencies:

Python 3.6, keras, tensorflow and various standard python libraries (numpy, scipy, pandas, etc.)
