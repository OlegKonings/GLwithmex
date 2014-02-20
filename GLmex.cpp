#include <cstring>
#include <cmath>
#include "stdafx.h"
#include "stdio.h"
#include "mex.h"
#include "matrix.h"
#include <cuda.h>//CUDA version 5.0 SDK, 64 bit
//#include <math_functions.h>
#include <cublas_v2.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cassert>

#define _DTH cudaMemcpyDeviceToHost
#define _DTD cudaMemcpyDeviceToDevice
#define _HTD cudaMemcpyHostToDevice

//make sure these match the .cu THREADS and BLOCK_SIZE
#define CPPTHREADS 64//This is 64 due to expectation that data sets will be small, if (m*n)>= 1e6 then use 256 THREADS(adjust here and in .cu file)
#define CPPBLOCK_SIZE 16
#define MEGA (1<<8)
#define MAX_LAMBDAS 32

const int blockSize0=64;//

void inline checkError(cublasStatus_t status, const char *msg){if (status != CUBLAS_STATUS_SUCCESS){printf("%s", msg);exit(EXIT_FAILURE);}}

//NOTE: all cublas calls are done in this cpp, and all CUDA kernels are in GLcuda.cu, and accessed via extern "C"
// using wraps for all GPU kernel calls

extern "C" void generateEye_wrap(float *E, const int N,const int numBlocks);

extern "C" void pad_ATA(const float *ATA, float *TempATA,  const int N,const int padd_N);

extern "C" void d_choldc_topleft_wrap(float *M, int boffset,const int N,const dim3 t_block);

extern "C" void d_choldc_strip_wrap(float *M, int boffset,const int N,const dim3 stripgrid,const dim3 t_block);

extern "C" void get_L(const float *Pad_L, float *L, const int N,const int padd_N);

extern "C" void update_vector_q(const float *Atb, const float *z, const float *u, float *q, const float rho,const int length,const int num_lambdas,
	const dim3 &grid,const int mask);

extern "C" void x_hat_update_helper(const float *x, const float *zold, float *x_hat, const int length, const float alpha, const dim3 &grid,
	const int mask);

extern "C" void z_shrinkage_wrap(float *D_z,const float *x_hat, const float *D_u,float *norm_s, float *t_obj_arr, float *z_norm_sum,
	const int adj_size,dim3 PGrid, const int *D_cuml_part,const float *lam_arr, const int rem_start, const float rho,const int num_blx,
	cudaError_t &err,const int u_length,const int p_length,const int mask);

extern "C" void gpu_lasso_u_update_wrap(float *u,const float *xh, const float *z,const int length,const dim3 &grid,const int mask);

extern "C" void get_multi_norms(const float *x, const float *z, const float *zold, const float *u,
	float *norm_arr,const float _rho, const int length,const int num_lambdas,const int mask);


inline int get_adj_size(int num_elem){
	float p=float(num_elem)/float(MEGA);
	if(p>0.8f)return 5;
	else if(p>0.6f)return 4;
	else if(p>0.4f)return 3;
	else if(p>0.2f)return 2;
	else
		return 1;
}
inline int get_dynamic_block_size(const int adj_size,const int blkSize){return (1<<(adj_size-1))*blkSize;}


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, mxArray const *prhs[]){//will take in 12 inputs and return 3 outputs
	/*inputs are (in order)
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
	*/

	//error checking, disable(comment out) if you are sure you are passing in the correct types
	/*if(nrhs!=13)mexErrMsgTxt("Wrong number of input arguments.");
	if(nlhs!=3)mexErrMsgTxt("Too many output arguments.");
	if(!mxIsSingle(prhs[0]))mexErrMsgTxt("Matrix A needs to be of type single.");
	if(!mxIsSingle(prhs[1]))mexErrMsgTxt("vector b needs to be of type single.");
	if(!mxIsInt32(prhs[2]))mexErrMsgTxt("partition vector needs to be of type int32.");
	if(!mxIsSingle(prhs[3]))mexErrMsgTxt("vector u needs to be of type single.");
	if(!mxIsSingle(prhs[4]))mexErrMsgTxt("vector z needs to be of type single.");*/
	
	//get parameters of inputs

	float *A=(float *)mxGetPr(prhs[0]);//matrix A in MATLAB column-major format
	float *b=(float *)mxGetPr(prhs[1]);//vector b
	int *p=(int *)mxGetPr(prhs[2]);
	float *u=(float *)mxGetPr(prhs[3]);//vector u
	float *z=(float *)mxGetPr(prhs[4]);//vector z
	float *lambda_array=(float *)mxGetPr(prhs[10]);
	int *num_iters=(int *)mxGetPr(prhs[11]);

	const int Arows=(int)mxGetN(prhs[0]);//since this Group Lasso solver internally uses row major, need to pass in input A as A' from Matlab, then swap (m,n)
	const int Acols=(int)mxGetM(prhs[0]);//ditto, swaping due to A' being passed in to mex interface

	const int adj_Arows=(((Arows+16-1)>>4)<<4),adj_Acols=(((Acols+16-1)>>4)<<4);//padding for Cholesky factor

	//printf("Arows=%d Acols=%d\n",Arows,Acols);
	const int Psize=(int)mxGetM(prhs[2]);
	const float _rho=(float)mxGetScalar(prhs[5]);
	const float _alpha=(float)mxGetScalar(prhs[6]);
	const int max_iter=(int)mxGetScalar(prhs[7]);
	const float abstol=(float)mxGetScalar(prhs[8]);
	const float reltol=(float)mxGetScalar(prhs[9]);
	const int num_lambdas=(int)mxGetM(prhs[10]);
	
	cublasHandle_t handle;//init cublas_v2
	cublasStatus_t cur;
	cur = cublasCreate(&handle);
	//if (cur != CUBLAS_STATUS_SUCCESS){printf("cublasCreate returned error code %d, line(%d)\n", cur, __LINE__);exit(EXIT_FAILURE);}

	int tot=0,max_block_size=1;
	int *cum_part=(int *)malloc(Psize*sizeof(int));
	for(int i=0;i<Psize;i++){
		tot+=p[i];
		cum_part[i]=tot;
		max_block_size=max(max_block_size,p[i]+1);//
	}
	if(tot!=Acols){
		printf("invalid partition! %d\n",tot);
		free(cum_part);
		return;
	}
	
	
	const float _beta=0.0f,t_alphA=1.0f;
	const unsigned int numbytesM=Arows*Acols*sizeof(float),numbytesVC=num_lambdas*Acols*sizeof(float),numbytesVR=num_lambdas*Arows*sizeof(float);

	float *D_A,*D_b,*D_xresult,*D_Atb,*D_u,*D_z,*tempvecC,*tempvecC2,*tempvecR,*x_hat,*L,*tmpM2,*tmpM3;
	float *norm_s, *t_obj_arr,*z_norm_sum,*lam_arr;
	int *D_cuml_part;
	//allocate all device memory
    cudaError_t err=cudaMalloc((void **)&D_A,numbytesM);
	//if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	err=cudaMalloc((void **)&D_b,Arows*sizeof(float));
	//if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	err=cudaMalloc((void **)&D_xresult,numbytesVC);
	//if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	err=cudaMalloc((void **)&D_Atb,Acols*sizeof(float));
	//if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	err=cudaMalloc((void **)&D_u,numbytesVC);
	//if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	err=cudaMalloc((void **)&D_z,numbytesVC);
	//if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	err=cudaMalloc((void **)&tempvecC,numbytesVC);
	//if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	err=cudaMalloc((void **)&tempvecC2,numbytesVC);
	//if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	err=cudaMalloc((void **)&tempvecR,numbytesVR);
	//if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	err=cudaMalloc((void **)&x_hat,numbytesVC);
	//if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}

	const int num_k_bytes=num_lambdas*Psize*sizeof(float);//Psize is length for kernels(z update)
	const int adj_size=get_adj_size(max_block_size);
	const int temp_blocks_sz=get_dynamic_block_size(adj_size,blockSize0);
	const int num_blx=max(1,max_block_size/temp_blocks_sz);

	const int rem_start=(max_block_size-(max_block_size-num_blx*temp_blocks_sz));//careful
	//assert(rem_start>0);
	dim3 PGrid(num_blx,Psize,num_lambdas);
	
	err=cudaMalloc((void **)&norm_s,num_k_bytes);//checkm
	//if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	err=cudaMalloc((void **)&t_obj_arr,num_k_bytes);
	//if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	err=cudaMalloc((void **)&z_norm_sum,num_lambdas*sizeof(float));
	//if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	err=cudaMalloc((void **)&lam_arr,num_lambdas*sizeof(float));
	//if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	err=cudaMalloc((void **)&D_cuml_part,Psize*sizeof(int));
	//note: if 'skinny' the the dim of L and U will be Acols x Acols, otherwise the dim of L and U will be Arows x Arows
	const bool skinny= (Arows>=Acols);
	const int N= (skinny) ? Acols:Arows;
	const int regN= (skinny) ? adj_Acols:adj_Arows;
	const int numBytesT=N*N*sizeof(float),numBlocks=((N*N + CPPTHREADS-1)/CPPTHREADS);
	
	err=cudaMalloc((void **)&L,numBytesT);
	//if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	err=cudaMalloc((void **)&tmpM2,numBytesT);
	//if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	err=cudaMalloc((void **)&tmpM3,regN*regN*sizeof(float));

	generateEye_wrap(tmpM2,N,numBlocks);

	err=cudaMemcpy(L,tmpM2,numBytesT,_DTD);//copy eye matrix for inversion use to L
	//if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}

	err=cudaMemcpy(D_A,A,numbytesM,_HTD);//D_A gets A's vals
	//if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	err=cudaMemcpy(D_b,b,Arows*sizeof(float),_HTD);//copy over vector b
	//if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	err=cudaMemcpy(D_u,u,numbytesVC,_HTD);//copy over vector u
	//if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	err=cudaMemcpy(D_z,z,numbytesVC,_HTD);//copy over vector z
	//if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	err=cudaMemcpy(lam_arr,lambda_array,num_lambdas*sizeof(float),_HTD);//copy over vector z
	//if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	err=cudaMemcpy(D_cuml_part,cum_part,Psize*sizeof(int),_HTD);
	//if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}

	err=cudaMemset(D_xresult,0,numbytesVC);//start x off with zeros
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	//now all device memory allocated, copied and set to initial values

	//Atb = A'*b;
	cur=cublasSgemv_v2(handle,CUBLAS_OP_N,Acols,Arows,&t_alphA,D_A,Acols,D_b,1,&_beta,D_Atb,1);//Atb=AT*b,since D_A starts in row-major, not need to tranpose
	/*if (cur != CUBLAS_STATUS_SUCCESS){
		printf("error code %d, line(%d)\n", cur, __LINE__);
        exit(EXIT_FAILURE);
	}*/

	//verified for rect
	if(skinny){//A'*A + rho*eye(n)
		cur=cublasSgemm_v2(handle,CUBLAS_OP_N,CUBLAS_OP_T,Acols,Acols,Arows,&t_alphA,D_A,Acols,D_A,Acols,&_rho,tmpM2,Acols);//tmpM2=AT*A+rho*eyeMatrix(tmpM2)
		/*if(cur != CUBLAS_STATUS_SUCCESS){
			printf("error code %d, line(%d)\n", cur, __LINE__);
			exit(EXIT_FAILURE);
		}*/
	}else{//speye(m) + 1/rho*(A*A')
		float _InvRho=1.0f/_rho;
		cur=cublasSgemm_v2(handle,CUBLAS_OP_T,CUBLAS_OP_N,Arows,Arows,Acols,&_InvRho,D_A,Acols,D_A,Acols,&t_alphA,tmpM2,Arows);//tmpM2=1/rho*(A*AT)+eye(Arows)
		/*if(cur != CUBLAS_STATUS_SUCCESS){
			printf("error code %d, line(%d)\n", cur, __LINE__);
			exit(EXIT_FAILURE);
		}*/
	}
	//tmpM2 will act as  A'*A + rho*eye(n) if skinny, or as  eye(m) + 1/rho*(A*A') if fat

	//Now cholesky factor of tmpM2
	pad_ATA(tmpM2,tmpM3,N,regN);
	
	//Now cholesky factor of tmpM2
	dim3 threads(CPPBLOCK_SIZE,CPPBLOCK_SIZE);
	int todo=(regN/CPPBLOCK_SIZE);
	int reps=todo,k=CPPBLOCK_SIZE;
	float al=-1.0f,ba=1.0f;
	int n,rloc,cloc,cloc2;
	dim3 stripgrid(1,1,1);
	while(reps>2){
		stripgrid.x=reps-1;
		d_choldc_topleft_wrap(tmpM3,todo-reps,regN,threads);//d_choldc_topleft<<<1,threads>>>(tmpM3,todo-reps,regN)
		d_choldc_strip_wrap(tmpM3,todo-reps,regN,stripgrid,threads);//d_choldc_strip<<<stripgrid,threads>>>(tmpM3,todo-reps,regN)
		n=CPPBLOCK_SIZE*(reps-1);
		rloc=(CPPBLOCK_SIZE*(todo-reps+1))*regN;
		cloc=CPPBLOCK_SIZE*(todo-reps);
		cloc2=CPPBLOCK_SIZE*(todo-reps+1);
		cublasSsyrk_v2(handle,CUBLAS_FILL_MODE_UPPER,CUBLAS_OP_T,n,k,&al,(float *)&tmpM3[rloc+cloc],regN,&ba,(float *)&tmpM3[rloc+cloc2],regN);
		
		reps--;
	}
	if(todo>1){
		stripgrid.x=1;
		d_choldc_topleft_wrap(tmpM3,todo-2,regN,threads);//d_choldc_topleft<<<1,threads>>>(tmpM3,todo-2,regN)
		d_choldc_strip_wrap(tmpM3,todo-2,regN,stripgrid,threads);//d_choldc_strip<<<1,threads>>>(tmpM3,todo-2,regN)
		cublasSsyrk_v2(handle,CUBLAS_FILL_MODE_UPPER,CUBLAS_OP_T,k,k,&al,(float *)&tmpM3[(k*(todo-1))*regN +(k*(todo-2))],regN,&ba,(float *)&tmpM3[(k*(todo-1))*regN +(k*(todo-1))],regN);	
	}
	d_choldc_topleft_wrap(tmpM3,todo-1,regN,threads);//d_choldc_topleft<<<1,threads>>>(tmpM3,todo-1,regN)
	get_L(tmpM3,tmpM2,N,regN);
	//now have cholesky with padding, going forward using N

	cur=cublasStrsm_v2(handle,CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_UPPER,CUBLAS_OP_N,CUBLAS_DIAG_NON_UNIT,N,N,&t_alphA,tmpM2,N,L,N);
	if(cur!=CUBLAS_STATUS_SUCCESS){printf("error code %d, line(%d)\n", cur, __LINE__);exit(EXIT_FAILURE);}

	//tmpM2 will have value (inv(U))*inv(L))
	cur=cublasSgemm_v2(handle,CUBLAS_OP_N,CUBLAS_OP_T,N,N,N,&t_alphA,L,N,L,N,&_beta,tmpM2,N);
	if(cur!=CUBLAS_STATUS_SUCCESS){printf("error code %d, line(%d)\n", cur, __LINE__);exit(EXIT_FAILURE);}


	///////////END of pre-calculations//////////////////////

	const int num_v_blocks=(Acols+CPPTHREADS-1)/CPPTHREADS,num_vr_blocks=(Arows+CPPTHREADS-1)/CPPTHREADS;
	const float t_alpha=1.0f,t_beta=0.0f,neg_alpha=(1.0f-_alpha),neg=-1.0f,c0=sqrt(float(Acols))*abstol;

	//these will be updated within loop so no init is required
	float t_c,history_r_norm[MAX_LAMBDAS],history_s_norm[MAX_LAMBDAS],history_eps_pri[MAX_LAMBDAS],history_eps_dual[MAX_LAMBDAS],xnorm[MAX_LAMBDAS],znorm[MAX_LAMBDAS];

	int multi=0;
	
	dim3 grid((Acols+CPPTHREADS-1)/CPPTHREADS,num_lambdas,1);
	cudaStream_t *streams = (cudaStream_t *)malloc(num_lambdas*sizeof(cudaStream_t));

	for(;multi<num_lambdas;multi++){
		err=cudaStreamCreate(&(streams[multi]));
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}

	}

	//MAIN LOOP
	int mask=0;//32 bit bit-mask to keep track of lambdas which have converged

	for(int i=1;i<=max_iter;i++){//use max_iter parameter passed into function, same as Tim's 'k'

	
		update_vector_q(D_Atb,D_z,D_u,tempvecC2,_rho,Acols,num_lambdas,grid,mask);//tempvecC2 will be vectors 'q',update all q for all lambdas,q = Atb + rho*(z - u);
		
		if(skinny){

			for(multi=0;multi<num_lambdas;multi++)if(!(mask&(1<<multi))){

				cur=cublasSetStream(handle,streams[multi]);
				if(cur != CUBLAS_STATUS_SUCCESS){
					printf("error code %d, line(%d)\n", cur, __LINE__);
					exit(EXIT_FAILURE);
				}
				cur=cublasSgemv_v2(handle,CUBLAS_OP_T,Acols,Acols,&t_alpha,tmpM2,Acols,tempvecC2+multi*Acols,1,&t_beta,D_xresult+multi*Acols,1);//tmpvecC=(inv(U)*inv(L))* vector q(tempvec2)
				if(cur != CUBLAS_STATUS_SUCCESS){
					printf("error code %d, line(%d)\n", cur, __LINE__);
					exit(EXIT_FAILURE);
				}
			}
			//k
		}else{//careful here.. x = q/rho - (A'*(U \ ( L \ (A*q) )))/rho^2 , will start at innermost bracket (A*q) and work way out
			for(multi=0;multi<num_lambdas;multi++)if(!(mask&(1<<multi))){

				cur=cublasSetStream(handle,streams[multi]);
				if(cur != CUBLAS_STATUS_SUCCESS){
					printf("error code %d, line(%d)\n", cur, __LINE__);
					exit(EXIT_FAILURE);
				}

				cur=cublasSgemv_v2(handle,CUBLAS_OP_T,Acols,Arows,&t_alpha,D_A,Acols,tempvecC2+multi*Acols,1,&t_beta,tempvecR+multi*Arows,1);//A*q into tempvecR
				if(cur != CUBLAS_STATUS_SUCCESS){
					printf("error code %d, line(%d)\n", cur, __LINE__);
					exit(EXIT_FAILURE);
				}
				cur=cublasSgemv_v2(handle,CUBLAS_OP_T,N,N,&t_alpha,tmpM2,N,tempvecR+multi*Arows,1,&t_beta,L+multi*Arows,1);//(inv(U)*inv(L))* tempvecR into D_xresult
				if(cur != CUBLAS_STATUS_SUCCESS){
					printf("error code %d, line(%d)\n", cur, __LINE__);
					exit(EXIT_FAILURE);
				}
				cur=cublasSgemv_v2(handle,CUBLAS_OP_N,Acols,Arows,&t_alpha,D_A,Acols,L+multi*Arows,1,&t_beta,D_xresult+multi*Acols,1);//A'*temp 'L' into d_xresults
				if(cur != CUBLAS_STATUS_SUCCESS){
					printf("error code %d, line(%d)\n", cur, __LINE__);
					exit(EXIT_FAILURE);
				}
				t_c= -(1.0f/(_rho*_rho));
				cur=cublasSscal_v2(handle,Acols,&t_c,D_xresult+multi*Acols,1);//scale D_xresult by -1/rho^2
				if(cur != CUBLAS_STATUS_SUCCESS){
					printf("error code %d, line(%d)\n", cur, __LINE__);
					exit(EXIT_FAILURE);
				}
				t_c=(1.0f/_rho);
				cur=cublasSaxpy_v2(handle,Acols,&t_c,tempvecC2+multi*Acols,1,D_xresult+multi*Acols,1);// D_xresult=q/rho+ D_xresult
				if(cur != CUBLAS_STATUS_SUCCESS){
					printf("error code %d, line(%d)\n", cur, __LINE__);
					exit(EXIT_FAILURE);
				}

			}
			
		}//k
		

		err=cudaMemcpy(tempvecC,D_z,numbytesVC,cudaMemcpyDeviceToDevice);//copy z into tempvecC,will act as 'zold'
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}

		
		x_hat_update_helper(D_xresult,tempvecC,x_hat,Acols,_alpha,grid,mask);//x_hat = alpha*x + (1-alpha)*zold;
		
		//here is the loop which differentiates the group lasso from the lasso,tempvecC2 will store the sectional results used in updating D_z
		
		err=cudaMemset(norm_s,0,num_k_bytes);
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
		err=cudaMemset(t_obj_arr,0,num_k_bytes);
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
		err=cudaMemset(z_norm_sum,0,num_lambdas*sizeof(float));
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}

		//call helper
		z_shrinkage_wrap(D_z,x_hat,D_u,norm_s,t_obj_arr,z_norm_sum,adj_size,PGrid,D_cuml_part,lam_arr,rem_start,_rho,num_blx,err,Acols,Psize,mask);//% z-update in parallel

		//updated z by section in parallel and copy back z_norm_sum to host

		err=cudaMemcpy(znorm,z_norm_sum,num_lambdas*sizeof(float),_DTH);//not needed
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
		//
		
		gpu_lasso_u_update_wrap(D_u,x_hat,D_z,Acols,grid,mask);//gpu_lasso_u_update<<<num_v_blocks,THREADS>>>(D_u,x_hat,D_z,Acols);//u = u + (x_hat - z);
		
		err=cudaMemset(tempvecC2,0,numbytesVC);
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
		get_multi_norms(D_xresult,D_z,tempvecC, D_u, tempvecC2, _rho,Acols,num_lambdas,mask);

		err=cudaMemcpy(xnorm,tempvecC2,num_lambdas*sizeof(float),_DTH);
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
		err=cudaMemcpy(znorm,&tempvecC2[num_lambdas],num_lambdas*sizeof(float),_DTH);
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
		err=cudaMemcpy(history_r_norm,&tempvecC2[num_lambdas*2],num_lambdas*sizeof(float),_DTH);
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
		err=cudaMemcpy(history_s_norm,&tempvecC2[num_lambdas*3],num_lambdas*sizeof(float),_DTH);
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
		err=cudaMemcpy(history_eps_dual,&tempvecC2[num_lambdas*4],num_lambdas*sizeof(float),_DTH);
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}


		for(multi=0;multi<num_lambdas;multi++)if(!(mask&(1<<multi))){
			history_r_norm[multi]=sqrtf(history_r_norm[multi]);
			history_s_norm[multi]=sqrtf(history_s_norm[multi]);
			xnorm[multi]=sqrtf(xnorm[multi]);
			znorm[multi]=sqrtf(znorm[multi]);
			history_eps_pri[multi]=c0+reltol*max(xnorm[multi],znorm[multi]);
			history_eps_dual[multi]=c0+reltol*sqrtf(history_eps_dual[multi]);
			//printf("\nr_norm= %f, s_norm= %f, eps_pri= %f, eps_dual= %f",history_r_norm[multi],history_s_norm[multi],
				//history_eps_pri[multi],history_eps_dual[multi]);
			if(history_r_norm[multi]<history_eps_pri[multi] && history_s_norm[multi]<history_eps_dual[multi]){
				mask|=(1<<multi);
				num_iters[multi]=i;
				//printf("number %d done at iter %d",multi,i);
			}
			//printf("\n");
		}

		if(mask==((1<<multi)-1))break;

	}
	//printf("%d %d %d",num_iters[0],num_iters[1],num_iters[2]);
	//create answer for Matlab and copy back vectors u and z
	plhs[0]=mxCreateNumericMatrix(Acols,num_lambdas,mxSINGLE_CLASS,mxREAL);
	plhs[1]=mxCreateNumericMatrix(Acols,num_lambdas,mxSINGLE_CLASS,mxREAL);
	plhs[2]=mxCreateNumericMatrix(num_lambdas,1,mxINT32_CLASS,mxREAL);

	float *u_result=(float *)mxGetPr(plhs[0]);
	float *z_result=(float *)mxGetPr(plhs[1]);
	int *num_iter_result=(int *)mxGetPr(plhs[2]);

	err=cudaMemcpy(u_result,D_u,numbytesVC,_DTH);//copy  u info back to host
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	err=cudaMemcpy(z_result,D_z,numbytesVC,_DTH);//copy  z info back to host
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	memcpy(num_iter_result,num_iters,num_lambdas*sizeof(int));

	for(multi=0;multi<num_lambdas;multi++){
		cudaStreamDestroy(streams[multi]);
	}
	free(streams);
	free(cum_part);//free host memory

	//free all device memory
	err=cudaFree(D_A);
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	err=cudaFree(D_b);
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	err=cudaFree(D_xresult);
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	err=cudaFree(D_Atb);
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	err=cudaFree(D_u);
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	err=cudaFree(D_z);
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	err=cudaFree(tempvecC);
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	err=cudaFree(tempvecC2);
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	err=cudaFree(tempvecR);
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	err=cudaFree(x_hat);
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	err=cudaFree(L);
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	err=cudaFree(tmpM2);
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	err=cudaFree(tmpM3);
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	err=cudaFree(norm_s);
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	err=cudaFree(t_obj_arr);
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	err=cudaFree(z_norm_sum);
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	err=cudaFree(lam_arr);
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	err=cudaFree(D_cuml_part);
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}

	checkError(cublasDestroy(handle), "cublasDestroy() error!\n");
}