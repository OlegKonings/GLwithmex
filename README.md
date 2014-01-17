GLwithmex
=========

beta mex interface for group lasso


mex interface for ADMM group lasso (single precision). Included is the generated m file with interface which must be put in MATLAB's working directory. This version is slightly different than the original ADMM group lasso in that it returns vectors u and z.

Using MATLAB's mex interface does incur overhead (MATLAB related) which is NOT present when the .cu GPU file is run as a stand alone executable. This is especially the case on the first (initialization) run.

Since this beta version of the GPU solver accepts row-major format(and MATLAB uses column-major), Matrix A must be passed in the function in the transpose state (A').

Overall the CUDA interfacec solver returns the same answer (within 1-e6) of the MATLAB version, but is only 2-3 times faster due to the MATLAB overhead of the mex interface. This small speedup is also because of the fact that MATLAB already has the data in memory(while the mex has to init-copy-transfer-copy-solve-copy-pass-pass back results.

The related .m files are included. Will work on a hybrid version which speeds up the call.
