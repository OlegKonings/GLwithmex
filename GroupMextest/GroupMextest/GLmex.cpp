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
#define CPPTHREADS 64
#define CPPBLOCK_SIZE 16

void inline checkError(cublasStatus_t status, const char *msg){if (status != CUBLAS_STATUS_SUCCESS){printf("%s", msg);exit(EXIT_FAILURE);}}

//NOTE: all cublas calls are done in this cpp, and all CUDA kernels are in GLcuda.cu, and accessed via extern "C"
extern "C" void generateEye_wrap(float *E, const int N,const int numBlocks);
extern "C" void gpu_inplace_vector_scale_wrap(float *V, const int size,const float _s,const int numBlocks);
extern "C" void gpu_vector_add_wrap(const float *a, const float *b, float *result, const int size, const bool add,const int numBlocks);
extern "C" void gpu_lasso_u_update_wrap(float *u,const float *xh, const float *z,const int size,const int numBlocks);
extern "C" void gpu_lasso_objective_helper_wrap(float *v0,const float *v1,const int size,const int numBlocks);
extern "C" void gpu_lasso_h_helper_wrap(const float *z, const float *zold, float *v_result,const float _rho,const int size,const int numBlocks);
extern "C" void gpu_group_lasso_shrinkage_wrap(const float *x, float *z_result,const float kappa,const float e_norm,const int size,const int numBlocks);
extern "C" void d_choldc_topleft_wrap(float *M, int boffset,const int N,const dim3 t_block);
extern "C" void d_choldc_strip_wrap(float *M, int boffset,const int N,const dim3 stripgrid,const dim3 t_block);


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, mxArray const *prhs[]){//will take in 11 inputs and return 2 outputs
	/*inputs are (in order)
	0) Matrix A (m,n) single precision floating point numbers (32 bit) in DENSE form
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

	const int Arows=mxGetM(prhs[0]);
	const int Acols=mxGetN(prhs[0]);
	printf("Arows=%d Acols=%d\n",Arows,Acols);
	const int Psize=mxGetM(prhs[2]);
	const float _rho=(float)mxGetScalar(prhs[5]);
	const float _alpha=(float)mxGetScalar(prhs[6]);
	const float _lambda=(float)mxGetScalar(prhs[7]);
	const int max_iter=(int)mxGetScalar(prhs[8]);
	const float abstol=(float)mxGetScalar(prhs[9]);
	const float reltol=(float)mxGetScalar(prhs[10]);


	cublasHandle_t handle;//init cublas_v2
	cublasStatus_t ret;
	ret = cublasCreate(&handle);
	if (ret != CUBLAS_STATUS_SUCCESS){
		printf("cublasCreate returned error code %d, line(%d)\n", ret, __LINE__);
		exit(EXIT_FAILURE);
	}

	int tot=0;
	int *cum_part=(int *)malloc(Psize*sizeof(int));
	for(int i=0;i<Psize;i++){
		tot+=p[i];
		cum_part[i]=tot;
	}
	if(tot!=Acols){
		printf("invalid partition! %d\n",tot);
		free(cum_part);
		return;
	}
	
	cublasStatus_t cur;
	const float _beta=0.0f,t_alphA=1.0f;
	unsigned int numbytesM=Arows*Acols*sizeof(float),numbytesVC=Acols*sizeof(float),numbytesVR=Arows*sizeof(float);

	float *D_A,*D_b,*D_xresult,*D_Atb,*D_u,*D_z,*tempvecC,*tempvecC2,*tempvecR,*x_hat,*L,*tmpM2;//all device memory allocations
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
	float t_c,shrink_norm,d_obj,t_obj,history_obj,obj_p0,history_r_norm,history_s_norm,history_eps_pri,nx,nnegz,history_eps_dual,nru;
	int start_idx,begin,end,num_elem;

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
		start_idx=0;
		shrink_norm=0.0f;d_obj=0.0f;t_obj=0.0f;t_c=_lambda/_rho;//d_obj will be calculated now for use in objective function
		for(int ii=0;ii<Psize;ii++){
			begin=start_idx;
			end=cum_part[ii]-1;
			num_elem=end-begin+1;
			//need to get euclid norm of the sum of the sections of vector x_hat and vector u
			gpu_vector_add_wrap((float *)&x_hat[begin],(float *)&D_u[begin],(float *)&tempvecC2[begin],num_elem,true,num_v_blocks);//gpu_vector_add<<<num_v_blocks,THREADS>>>((float *)&x_hat[begin],(float *)&D_u[begin],(float *)&tempvecC2[begin],num_elem,true);// tempvecC2(sel)=x_hat(sel) + u(sel)
			cur=cublasSnrm2_v2(handle,num_elem,(float *)&tempvecC2[begin],1,&shrink_norm);//get the euclid norm for that section of tempvecC2
			if(cur != CUBLAS_STATUS_SUCCESS){
				printf("error code %d, line(%d)\n", cur, __LINE__);
				exit(EXIT_FAILURE);
			}

			gpu_group_lasso_shrinkage_wrap((float *)&tempvecC2[begin],(float *)&D_z[begin],t_c,shrink_norm,num_elem,num_v_blocks);//gpu_group_lasso_shrinkage<<<num_v_blocks,THREADS>>>((float *)&tempvecC2[begin],(float *)&D_z[begin],t_c,shrink_norm,num_elem);//z(sel) = shrinkage(x_hat(sel) + u(sel), lambda/rho);
			cur=cublasSnrm2_v2(handle,num_elem,(float *)&D_z[begin],1,&t_obj);//get the euclid norm for that section of tempvecC2
			if(cur != CUBLAS_STATUS_SUCCESS){
				printf("error code %d, line(%d)\n", cur, __LINE__);
				exit(EXIT_FAILURE);
			}
			d_obj+=t_obj;//obj = obj + norm(z(sel)), for objective group lasso, calulating now for use in the next step
			start_idx=end+1;
		}

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

	checkError(cublasDestroy(handle), "cublasDestroy() error!\n");
}

