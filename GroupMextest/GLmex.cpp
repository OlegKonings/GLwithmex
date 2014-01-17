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
#include <ctime>
#include <cassert>

#define _DTH cudaMemcpyDeviceToHost
#define _DTD cudaMemcpyDeviceToDevice
#define _HTD cudaMemcpyHostToDevice

//make sure these match the .cu THREADS and BLOCK_SIZE
#define CPPTHREADS 64//This is 64 due to expectation that data sets will be small, if (m*n)>= 1e6 then use 256 THREADS(adjust here and in .cu file)
#define CPPBLOCK_SIZE 16
#define MEGA (1<<12)

const int blockSize0=128;//

void inline checkError(cublasStatus_t status, const char *msg){if (status != CUBLAS_STATUS_SUCCESS){printf("%s", msg);exit(EXIT_FAILURE);}}

//NOTE: all cublas calls are done in this cpp, and all CUDA kernels are in GLcuda.cu, and accessed via extern "C"
// using wraps for all GPU kernel calls

extern "C" void generateEye_wrap(float *E, const int N,const int numBlocks);
extern "C" void gpu_inplace_vector_scale_wrap(float *V, const int size,const float _s,const int numBlocks);
extern "C" void gpu_vector_add_wrap(const float *a, const float *b, float *result, const int size, const bool add,const int numBlocks);
extern "C" void gpu_lasso_u_update_wrap(float *u,const float *xh, const float *z,const int size,const int numBlocks);
extern "C" void gpu_lasso_objective_helper_wrap(float *v0,const float *v1,const int size,const int numBlocks);
extern "C" void gpu_lasso_h_helper_wrap(const float *z, const float *zold, float *v_result,const float _rho,const int size,const int numBlocks);
extern "C" void gpu_group_lasso_shrinkage_wrap(const float *x, float *z_result,const float kappa,const float e_norm,const int size,const int numBlocks);
extern "C" void d_choldc_topleft_wrap(float *M, int boffset,const int N,const dim3 t_block);
extern "C" void d_choldc_strip_wrap(float *M, int boffset,const int N,const dim3 stripgrid,const dim3 t_block);
extern "C" void z_shrinkage_wrap(float *D_z,const float *x_hat, const float *D_u,float *norm_s, float *t_obj_arr, float *z_norm_sum,
	const int adj_size,dim3 &PGrid, const int *D_cuml_part, const int rem_start, const float kappa,const int num_blx,
	cudaError_t &err);

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

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, mxArray const *prhs[]){//will take in 11 inputs and return 2 outputs
	/*inputs are (in order)
	0) Matrix A (m,n) single precision floating point numbers (32 bit) in DENSE form AND must be passed into mex in TRANSPOSE form due to row-major format(will adjust m and n internally)
	1) vector b (m,1) single precision floating point numbers
	2) vector p (Psize length) 32 bit integer of K(Psize) length (partitions)
	3) vector u (n,1) single precision floating point numbers
	4) vector z (n,1) single precision floating point numbers
	5) float (single) rho
	6) float (single) alpha
	7) float (single) lambda
	8) integer max_iter
	9) float (single) abstol
	10) float (single) reltol

	outputs are (in order)
	0) vector u (n,1) single precision floating point numbers
	1) vector z (n,1) single precision floating point numbers
	*/

	//error checking, disable(comment out) if you are sure you are passing in the correct types
	if(nrhs!=11)mexErrMsgTxt("Wrong number of input arguments.");
	if(nlhs!=2)mexErrMsgTxt("Too many output arguments.");
	if(!mxIsSingle(prhs[0]))mexErrMsgTxt("Matrix A needs to be of type single.");
	if(!mxIsSingle(prhs[1]))mexErrMsgTxt("vector b needs to be of type single.");
	if(!mxIsInt32(prhs[2]))mexErrMsgTxt("partition vector needs to be of type int32.");
	if(!mxIsSingle(prhs[3]))mexErrMsgTxt("vector u needs to be of type single.");
	if(!mxIsSingle(prhs[4]))mexErrMsgTxt("vector z needs to be of type single.");
	
	//get parameters of inputs

	float *A=(float *)mxGetPr(prhs[0]);//matrix A in MATLAB column-major format
	float *b=(float *)mxGetPr(prhs[1]);//vector b
	int *p=(int *)mxGetPr(prhs[2]);
	float *u=(float *)mxGetPr(prhs[3]);//vector u
	float *z=(float *)mxGetPr(prhs[4]);//vector z

	const int Arows=mxGetN(prhs[0]);//since this Group Lasso solver internally uses row major, need to pass in input A as A' from Matlab, then swap (m,n)
	const int Acols=mxGetM(prhs[0]);//ditto, swaping due to A' being passed in to mex interface
	//printf("Arows=%d Acols=%d\n",Arows,Acols);
	const int Psize=mxGetM(prhs[2]);
	const float _rho=(float)mxGetScalar(prhs[5]);
	const float _alpha=(float)mxGetScalar(prhs[6]);
	const float _lambda=(float)mxGetScalar(prhs[7]);
	const int max_iter=(int)mxGetScalar(prhs[8]);
	const float abstol=(float)mxGetScalar(prhs[9]);
	const float reltol=(float)mxGetScalar(prhs[10]);


	cublasHandle_t handle;//init cublas_v2
	cublasStatus_t cur;
	cur = cublasCreate(&handle);
	if (cur != CUBLAS_STATUS_SUCCESS){
		printf("cublasCreate returned error code %d, line(%d)\n", cur, __LINE__);
		exit(EXIT_FAILURE);
	}

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
	const unsigned int numbytesM=Arows*Acols*sizeof(float),numbytesVC=Acols*sizeof(float),numbytesVR=Arows*sizeof(float);

	float *D_A,*D_b,*D_xresult,*D_Atb,*D_u,*D_z,*tempvecC,*tempvecC2,*tempvecR,*x_hat,*L,*tmpM2;
	float *norm_s, *t_obj_arr,*z_norm_sum;
	int *D_cuml_part;
	//allocate all device memory
    cudaError_t err=cudaMalloc((void **)&D_A,numbytesM);
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	err=cudaMalloc((void **)&D_b,numbytesVR);
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	err=cudaMalloc((void **)&D_xresult,numbytesVC);
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	err=cudaMalloc((void **)&D_Atb,numbytesVC);
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	err=cudaMalloc((void **)&D_u,numbytesVC);
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	err=cudaMalloc((void **)&D_z,numbytesVC);
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	err=cudaMalloc((void **)&tempvecC,numbytesVC);
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	err=cudaMalloc((void **)&tempvecC2,numbytesVC);
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	err=cudaMalloc((void **)&tempvecR,numbytesVR);
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	err=cudaMalloc((void **)&x_hat,numbytesVC);
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}

	const int num_k_bytes=Psize*sizeof(float);
	const int adj_size=get_adj_size(max_block_size);
	const int temp_blocks_sz=get_dynamic_block_size(adj_size,blockSize0);
	const int num_blx=max(1,max_block_size/temp_blocks_sz);

	const int rem_start=(max_block_size-(max_block_size-num_blx*temp_blocks_sz));//careful
	assert(rem_start>0);
	dim3 PGrid(num_blx,Psize,1);
	
	err=cudaMalloc((void **)&norm_s,num_k_bytes);
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	err=cudaMalloc((void **)&t_obj_arr,num_k_bytes);
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	err=cudaMalloc((void **)&z_norm_sum,sizeof(float));
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	err=cudaMalloc((void **)&D_cuml_part,Psize*sizeof(int));
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}

	//note: if 'skinny' the the dim of L and U will be Acols x Acols, otherwise the dim of L and U will be Arows x Arows
	const bool skinny= (Arows>=Acols);
	const int N= (skinny) ? Acols:Arows;
	const int numBytesT=N*N*sizeof(float),numBlocks=((N*N + CPPTHREADS-1)/CPPTHREADS);
	
	err=cudaMalloc((void **)&L,numBytesT);
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	err=cudaMalloc((void **)&tmpM2,numBytesT);
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}

	generateEye_wrap(tmpM2,N,numBlocks);

	err=cudaMemcpy(L,tmpM2,numBytesT,_DTD);//copy eye matrix for inversion use to L
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}

	err=cudaMemcpy(D_A,A,numbytesM,_HTD);//D_A gets A's vals
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	err=cudaMemcpy(D_b,b,numbytesVR,_HTD);//copy over vector b
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	err=cudaMemcpy(D_u,u,numbytesVC,_HTD);//copy over vector u
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	err=cudaMemcpy(D_z,z,numbytesVC,_HTD);//copy over vector z
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	err=cudaMemcpy(D_cuml_part,cum_part,Psize*sizeof(int),_HTD);
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}

	err=cudaMemset(D_xresult,0,numbytesVC);//start x off with zeros
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	//now all device memory allocated, copied and set to initial values

	//Atb = A'*b;
	cur=cublasSgemv_v2(handle,CUBLAS_OP_N,Acols,Arows,&t_alphA,D_A,Acols,D_b,1,&_beta,D_Atb,1);//Atb=AT*b,since D_A starts in row-major, not need to tranpose
	if (cur != CUBLAS_STATUS_SUCCESS){
		printf("error code %d, line(%d)\n", cur, __LINE__);
        exit(EXIT_FAILURE);
	}

	//verified for rect
	if(skinny){//A'*A + rho*eye(n)
		cur=cublasSgemm_v2(handle,CUBLAS_OP_N,CUBLAS_OP_T,Acols,Acols,Arows,&t_alphA,D_A,Acols,D_A,Acols,&_rho,tmpM2,Acols);//tmpM2=AT*A+rho*eyeMatrix(tmpM2)
		if(cur != CUBLAS_STATUS_SUCCESS){
			printf("error code %d, line(%d)\n", cur, __LINE__);
			exit(EXIT_FAILURE);
		}
	}else{//speye(m) + 1/rho*(A*A')
		float _InvRho=1.0f/_rho;
		cur=cublasSgemm_v2(handle,CUBLAS_OP_T,CUBLAS_OP_N,Arows,Arows,Acols,&_InvRho,D_A,Acols,D_A,Acols,&t_alphA,tmpM2,Arows);//tmpM2=1/rho*(A*AT)+eye(Arows)
		if(cur != CUBLAS_STATUS_SUCCESS){
			printf("error code %d, line(%d)\n", cur, __LINE__);
			exit(EXIT_FAILURE);
		}
	}
	//tmpM2 will act as  A'*A + rho*eye(n) if skinny, or as  eye(m) + 1/rho*(A*A') if fat

	//Now cholesky factor of tmpM2
	dim3 threads(CPPBLOCK_SIZE,CPPBLOCK_SIZE);
	int todo=(N/CPPBLOCK_SIZE);//must be multiple of 16, padded otherwise
	int reps=todo,k=CPPBLOCK_SIZE;
	float al=-1.0f,ba=1.0f;
	int n,rloc,cloc,cloc2;
	dim3 stripgrid(1,1,1);
	while(reps>2){
		stripgrid.x=reps-1;
		d_choldc_topleft_wrap(tmpM2,todo-reps,N,threads);//d_choldc_topleft<<<1,threads>>>(tmpM2,todo-reps,N)
		d_choldc_strip_wrap(tmpM2,todo-reps,N,stripgrid,threads);//d_choldc_strip<<<stripgrid,threads>>>(tmpM2,todo-reps,N)
		n=CPPBLOCK_SIZE*(reps-1);
		rloc=(CPPBLOCK_SIZE*(todo-reps+1))*N;
		cloc=CPPBLOCK_SIZE*(todo-reps);
		cloc2=CPPBLOCK_SIZE*(todo-reps+1);
		cur=cublasSsyrk_v2(handle,CUBLAS_FILL_MODE_UPPER,CUBLAS_OP_T,n,k,&al,(float *)&tmpM2[rloc+cloc],N,&ba,(float *)&tmpM2[rloc+cloc2],N);
		if(cur!=CUBLAS_STATUS_SUCCESS){
			printf("cublas returned error code %d, line(%d)\n", cur, __LINE__);
		}
		reps--;
	}
	if(todo>1){
		stripgrid.x=1;
		d_choldc_topleft_wrap(tmpM2,todo-2,N,threads);//d_choldc_topleft<<<1,threads>>>(tmpM2,todo-2,N)
		d_choldc_strip_wrap(tmpM2,todo-2,N,stripgrid,threads);//d_choldc_strip<<<1,threads>>>(tmpM2,todo-2,N)
		cur=cublasSsyrk_v2(handle,CUBLAS_FILL_MODE_UPPER,CUBLAS_OP_T,k,k,&al,(float *)&tmpM2[(k*(todo-1))*N +(k*(todo-2))],N,&ba,(float *)&tmpM2[(k*(todo-1))*N +(k*(todo-1))],N);
		if(cur!=CUBLAS_STATUS_SUCCESS){
			printf("cublas returned error code %d, line(%d)\n", cur, __LINE__);
		}
	}
	d_choldc_topleft_wrap(tmpM2,todo-1,N,threads);//d_choldc_topleft<<<1,threads>>>(tmpM2,todo-1,N)

	//now solve for inv(tmpM2) and store in L using cuBLAS
	cur=cublasStrsm_v2(handle,CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_UPPER,CUBLAS_OP_N,CUBLAS_DIAG_NON_UNIT,N,N,&t_alphA,tmpM2,N,L,N);
	if(cur!=CUBLAS_STATUS_SUCCESS){printf("error code %d, line(%d)\n", cur, __LINE__);exit(EXIT_FAILURE);}
    
	//now L is inverse of cholesky factor, will be used in loop below
	//now declare all variable which will be used in main solver loop
	const int num_v_blocks=(Acols+CPPTHREADS-1)/CPPTHREADS,num_vr_blocks=(Arows+CPPTHREADS-1)/CPPTHREADS;
	const float t_alpha=1.0f,t_beta=0.0f,neg_alpha=(1.0f-_alpha),neg=-1.0f,c0=sqrt(float(Acols))*abstol;

	//these will be updated within loop so no init is required
	float t_c,d_obj,history_obj,obj_p0,history_r_norm,history_s_norm,history_eps_pri,nx,nnegz,history_eps_dual,nru;

	//MAIN LOOP
	for(int i=1;i<=max_iter;i++){//use max_iter parameter passed into function

		err=cudaMemcpy(tempvecC,D_z,numbytesVC,cudaMemcpyDeviceToDevice);//copy z into tempvecC
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
		cur=cublasSaxpy_v2(handle,Acols,&neg,D_u,1,tempvecC,1);//tempvecC now has result of (z-u),or -1*u + z
		if(cur != CUBLAS_STATUS_SUCCESS){
				printf("error code %d, line(%d)\n", cur, __LINE__);
				exit(EXIT_FAILURE);
			}
		err=cudaMemcpy(tempvecC2,D_Atb,numbytesVC,cudaMemcpyDeviceToDevice);//copy Atb into tempvecC2
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
		cur=cublasSaxpy_v2(handle,Acols,&_rho,tempvecC,1,tempvecC2,1);//now tempvecC2 has result of Atb + rho*(z - u), ( vector q in stanford ADMM)
		if(cur != CUBLAS_STATUS_SUCCESS){
				printf("error code %d, line(%d)\n", cur, __LINE__);
				exit(EXIT_FAILURE);
			}
		//q or tempvecC2 will be Acols length
		
		if(skinny){
			cur=cublasSgemv_v2(handle,CUBLAS_OP_T,Acols,Acols,&t_alpha,L,Acols,tempvecC2,1,&t_beta,tempvecC,1);//tmpvecC=inv(L) * vector q(tempvec2)
			if(cur != CUBLAS_STATUS_SUCCESS){
				printf("error code %d, line(%d)\n", cur, __LINE__);
				exit(EXIT_FAILURE);
			}
			cur=cublasSgemv_v2(handle,CUBLAS_OP_N,Acols,Acols,&t_alpha,L,Acols,tempvecC,1,&t_beta,D_xresult,1);//x= U \ (L \ q) or x=inv(U)*(inv(L)*q
			if(cur != CUBLAS_STATUS_SUCCESS){
				printf("error code %d, line(%d)\n", cur, __LINE__);
				exit(EXIT_FAILURE);
			}
		}else{//careful here.. x = q/rho - (A'*(U \ ( L \ (A*q) )))/rho^2 , will start at innermost bracket (A*q) and work way out
			cur=cublasSgemv_v2(handle,CUBLAS_OP_T,Acols,Arows,&t_alpha,D_A,Acols,tempvecC2,1,&t_beta,tempvecR,1);//tmpvecR=A * vector q(tempvec2)
			if(cur != CUBLAS_STATUS_SUCCESS){
				printf("error code %d, line(%d)\n", cur, __LINE__);
				exit(EXIT_FAILURE);
			}
			//v
			cur=cublasSgemv_v2(handle,CUBLAS_OP_T,Arows,Arows,&t_alpha,L,Arows,tempvecR,1,&t_beta,D_xresult,1);//now D_xresult will be temp vec which=inv(L)*tmpmvec2(A*q)
			if(cur != CUBLAS_STATUS_SUCCESS){
				printf("error code %d, line(%d)\n", cur, __LINE__);
				exit(EXIT_FAILURE);
			}
			//v
			cur=cublasSgemv_v2(handle,CUBLAS_OP_N,Arows,Arows,&t_alpha,L,Arows,D_xresult,1,&t_beta,tempvecR,1);//now tempvecC will be temp vec which=inv(U)*inv(L)*tmpmvec2(A*q)
			if(cur != CUBLAS_STATUS_SUCCESS){
				printf("error code %d, line(%d)\n", cur, __LINE__);
				exit(EXIT_FAILURE);
			}
			//v
			cur=cublasSgemv_v2(handle,CUBLAS_OP_N,Acols,Arows,&t_alpha,D_A,Acols,tempvecR,1,&t_beta,D_xresult,1);//now D_xresult will be temp vec which=AT*inv(U)*inv(L)*tmpmvec2(A*q)
			if(cur != CUBLAS_STATUS_SUCCESS){
				printf("error code %d, line(%d)\n", cur, __LINE__);
				exit(EXIT_FAILURE);
			}
			//v
			t_c= -(1.0f/(_rho*_rho));
			cur=cublasSscal_v2(handle,Acols,&t_c,D_xresult,1);//scale D_xresult by -1/rho^2
			if(cur != CUBLAS_STATUS_SUCCESS){
				printf("error code %d, line(%d)\n", cur, __LINE__);
				exit(EXIT_FAILURE);
			}
			t_c=(1.0f/_rho);
			cur=cublasSaxpy_v2(handle,Acols,&t_c,tempvecC2,1,D_xresult,1);// D_xresult=q/rho+ D_xresult
			if(cur != CUBLAS_STATUS_SUCCESS){
				printf("error code %d, line(%d)\n", cur, __LINE__);
				exit(EXIT_FAILURE);
			}
			//v
		}
		//verified 
		err=cudaMemcpy(tempvecC,D_z,numbytesVC,cudaMemcpyDeviceToDevice);//copy z into tempvecC,will act as 'zold'
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}

		err=cudaMemcpy(x_hat,D_xresult,numbytesVC,cudaMemcpyDeviceToDevice);//set x_hat to cur val xresult
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}

		//for kernel calls related to the vectors

		gpu_inplace_vector_scale_wrap(x_hat,Acols,_alpha,num_v_blocks);//gpu_inplace_vector_scale<<<num_v_blocks,THREADS>>>(x_hat,Acols,_alpha);//xhat*=alpha
		
		cur=cublasSaxpy_v2(handle,Acols,&neg_alpha,tempvecC,1,x_hat,1);//x_hat+=(1-alpha)*(tempvecC);
		if(cur != CUBLAS_STATUS_SUCCESS){
				printf("error code %d, line(%d)\n", cur, __LINE__);
				exit(EXIT_FAILURE);
			}

		//verified
		//here is the loop which differentiates the group lasso from the lasso,tempvecC2 will store the sectional results used in updating D_z
		t_c=_lambda/_rho;//kappa
		err=cudaMemset(norm_s,0,num_k_bytes);
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
		err=cudaMemset(t_obj_arr,0,num_k_bytes);
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
		err=cudaMemset(z_norm_sum,0,sizeof(float));
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
		//call helper
		z_shrinkage_wrap(D_z,x_hat,D_u,norm_s,t_obj_arr,z_norm_sum,adj_size,PGrid,D_cuml_part,rem_start,t_c,num_blx,err);

		//updated z by section in parallel and copy back z_norm_sum to host

		err=cudaMemcpy(&d_obj,z_norm_sum,sizeof(float),_DTH);
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
		//
		
		gpu_lasso_u_update_wrap(D_u,x_hat,D_z,Acols,num_v_blocks);//gpu_lasso_u_update<<<num_v_blocks,THREADS>>>(D_u,x_hat,D_z,Acols);//u = u + (x_hat - z);
		//verified
		// now compute objective value in two parts, needs to be optimized. Does not directly affect output vector x
		//this calculation differs from the regular lasso as well
		obj_p0=0.0f;
		history_obj=0.0f;
		ba=0.0f;
		//same as lasso, but the z portion of objective is different due to blocks/partitioning
		cur=cublasSgemv_v2(handle,CUBLAS_OP_T,Acols,Arows,&t_alpha,D_A,Acols,D_xresult,1,&ba,tempvecR,1);//tempvecR=A*x_result, size Arows
		if(cur != CUBLAS_STATUS_SUCCESS){
			printf("error code %d, line(%d)\n", cur, __LINE__);
			exit(EXIT_FAILURE);
		}

		gpu_lasso_objective_helper_wrap(tempvecR,D_b,Arows,num_vr_blocks);//_gpu_lasso_objective_helper<<<num_vr_blocks,THREADS>>>(tempvecR,D_b,Arows);//tempvecR will have vector result of (A*x-b).^2
		
		cur=cublasSasum_v2(handle,Arows,tempvecR,1,&obj_p0);//sum up ((A*x - b).^2), can use asum because all values of tempvecC2 are positive
		if(cur != CUBLAS_STATUS_SUCCESS){
			printf("error code %d, line(%d)\n", cur, __LINE__);
			exit(EXIT_FAILURE);
		}
		history_obj=0.5f*obj_p0+_lambda*d_obj;//gpu objective=( 1/2*sum((A*x - b).^2) + lambda*obj )
		al=-1.0f;

		//history.r_norm(k)  = norm(x - z);
		history_r_norm=0.0f;
		err=cudaMemcpy(tempvecC2,D_xresult,numbytesVC,cudaMemcpyDeviceToDevice);//tempvecC2 will have a copy of d_z
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}

		cur=cublasSaxpy_v2(handle,Acols,&neg,D_z,1,tempvecC2,1);//tempvecC2 has the result of (x-z) or (-z+x)
		if(cur != CUBLAS_STATUS_SUCCESS){
				printf("error code %d, line(%d)\n", cur, __LINE__);
				exit(EXIT_FAILURE);
			}

		cur=cublasSnrm2_v2(handle,Acols,tempvecC2,1,&history_r_norm);//get gpu euclid norm
		if(cur != CUBLAS_STATUS_SUCCESS){
			printf("error code %d, line(%d)\n", cur, __LINE__);
			exit(EXIT_FAILURE);
		}
		//verified
		//history.s_norm(k)  = norm(-rho*(z - zold));
		history_s_norm=0.0f;
		
		gpu_lasso_h_helper_wrap(D_z,tempvecC,tempvecC2,_rho,Acols,num_v_blocks);//gpu_lasso_h_helper<<<num_v_blocks,THREADS>>>(D_z,tempvecC,tempvecC2,_rho,Acols);//tempvecC2 will have result of -rho*(z - zold)
		
		cur=cublasSnrm2_v2(handle,Acols,tempvecC2,1,&history_s_norm);//get gpu euclid norm
		if(cur != CUBLAS_STATUS_SUCCESS){
			printf("error code %d, line(%d)\n", cur, __LINE__);
			exit(EXIT_FAILURE);
		}
		//verified
		//history.eps_pri(k) = sqrt(n)*ABSTOL + RELTOL*max(norm(x), norm(-z));
		history_eps_pri=0.0f;
		nx=0.0f;
		nnegz=0.0f;
		cur=cublasSnrm2_v2(handle,Acols,D_xresult,1,&nx);//get gpu euclid norm
		if(cur != CUBLAS_STATUS_SUCCESS){
			printf("error code %d, line(%d)\n", cur, __LINE__);
			exit(EXIT_FAILURE);
		}
		cur=cublasSnrm2_v2(handle,Acols,D_z,1,&nnegz);//get gpu euclid norm
		if(cur != CUBLAS_STATUS_SUCCESS){
			printf("error code %d, line(%d)\n", cur, __LINE__);
			exit(EXIT_FAILURE);
		}
		history_eps_pri=c0+reltol*max(nx,nnegz);
		//verified
		//history.eps_dual(k)= sqrt(n)*ABSTOL + RELTOL*norm(rho*u)
		history_eps_dual=0.0f;
		nru=0.0f;
		err=cudaMemcpy(tempvecC2,D_u,numbytesVC,cudaMemcpyDeviceToDevice);//tempvecC2 will have value of vector u(device)
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}

		gpu_inplace_vector_scale_wrap(tempvecC2,Acols,_rho,num_v_blocks);//gpu_inplace_vector_scale<<<num_v_blocks,THREADS>>>(tempvecC2, Acols,_rho);//u*rho
		cur=cublasSnrm2_v2(handle,Acols,tempvecC2,1,&nru);//get gpu euclid norm
		if(cur != CUBLAS_STATUS_SUCCESS){
			printf("error code %d, line(%d)\n", cur, __LINE__);
			exit(EXIT_FAILURE);
		}
		history_eps_dual=c0+reltol*nru;
		if(history_r_norm<=history_eps_pri && history_s_norm<=history_eps_dual)break;//termination condition	
	}

	//create answer for Matlab and copy back vectors u and z
	plhs[0]=mxCreateNumericMatrix(Acols,1,mxSINGLE_CLASS,mxREAL);
	plhs[1]=mxCreateNumericMatrix(Acols,1,mxSINGLE_CLASS,mxREAL);

	float *u_result=(float *)mxGetPr(plhs[0]);
	float *z_result=(float *)mxGetPr(plhs[1]);

	err=cudaMemcpy(u_result,D_u,numbytesVC,_DTH);//copy  u info back to host
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	err=cudaMemcpy(z_result,D_z,numbytesVC,_DTH);//copy  z info back to host
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}

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
	err=cudaFree(norm_s);
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	err=cudaFree(t_obj_arr);
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	err=cudaFree(z_norm_sum);
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	err=cudaFree(D_cuml_part);
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}

	checkError(cublasDestroy(handle), "cublasDestroy() error!\n");
}

