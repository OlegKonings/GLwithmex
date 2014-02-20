GLwithmex
=========

mex interface for CUDA implementation of Stephen Boyd's admm group lasso solver, with the extra feature of mulitple lambda tesing(in parallel).

http://www.stanford.edu/~boyd/papers/admm/group_lasso/group_lasso.html

This implementation is a mostly literal translation of the solver, with the added ability to test up to 31 lambdas in parallel. Can operate on any shape of matrix A, but should be at least size (32,32), otherwise the infernal MATLAB mex overheads will no make the call worthwhile.


inputs from MATLAB are (in order)
	0) Matrix A (m,n) single precision floating point numbers (32 bit) in DENSE form AND must be passed into mex in TRANSPOSE form due to row-major format(will adjust m and n internally)
	1) vector b (m,1) single precision floating point numbers
	2) vector p (Psize length) 32 bit integer of K(Psize) length (partitions)
	3) vector u (n,1) single precision floating point numbers
	4) vector z (n,1) single precision floating point numbers
	5) float (single) rho
	6) float (single) alpha
	7) integer max_iter
	8) float (single) abstol
	9) float (single) reltol
	10) lambda array
	11) 32 bit integer array which will return the number of iterations until convergence for each lambda(size of array is equal to number of lambdas)

	outputs are (in order)
	0) vector u (n,lambdas) single precision floating point numbers
	1) vector z (n,lambdas) single precision floating point numbers
	2) vector num_iters (num_lambdas,1) 32-bit integer array
	
	
	NOTE: compile with --use_fast_math and for better parallel performance set environment variable CUDA_DEVICE_MAX_CONNECTIONS to 32 if using the Tesla line GPUs. 
	
	Testing was done with default CUDA_DEVICE_MAX_CONNECTIONS=8, but if testing more lambdas increase to number of lambdas.
