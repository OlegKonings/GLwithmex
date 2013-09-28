#include "stdafx.h"
#include "stdio.h"
#include <cuda.h>
#include <math_functions.h>
#include <cublas_v2.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define BLOCK_SIZE 16
#define BLOCKSIZE BLOCK_SIZE
#define AVOIDBANKCONFLICTS 0
#define USELOOPUNROLLING 1
#define TESTBLOCKS 16
#define IDC2D(i,j,ld) (((j)*(ld))+(i))

#define THREADS 64//this is 64 because for this version of ADMM group lasso, data sets will be small. For later data sets use 256
//make sure matches cpp CPPTHREADS

#define LINEAR_BLOCK_SIZE THREADS

//general use kernels
/////////////////////////////////////////////////////////////////////////////////////
__global__ void generateEye(float *E, const int size){
    int offset = blockIdx.x*blockDim.x + threadIdx.x;
	if(offset<(size*size)){
		int y = offset/size,x = offset - y*size;
		E[offset] = (x == y) ? 1.0f:0.0f;
	}
}
extern "C" void generateEye_wrap(float *E, const int N,const int numBlocks){
	generateEye<<<numBlocks,THREADS>>>(E,N);
}
//////////////////////////////////////////////////////////////////////////////////////

__global__ void gpu_inplace_vector_scale(float *V, const int size,const float _s){
	int offset = blockIdx.x*blockDim.x + threadIdx.x;
	if(offset<size){
		V[offset]*=_s;
	}
}
extern "C" void gpu_inplace_vector_scale_wrap(float *V, const int size,const float _s,const int numBlocks){
	gpu_inplace_vector_scale<<<numBlocks,THREADS>>>(V,size,_s);
}
//////////////////////////////////////////////////////////////////////////////////////

__global__ void gpu_vector_add(const float *a, const float *b, float *result, const int size, const bool add){
	int offset = blockIdx.x*blockDim.x + threadIdx.x;
	if(offset<size){
		result[offset]= add ? (a[offset]+b[offset]):(a[offset]-b[offset]);
	}
}
extern "C" void gpu_vector_add_wrap(const float *a, const float *b, float *result, const int size, const bool add,const int numBlocks){
	gpu_vector_add<<<numBlocks,THREADS>>>(a,b,result,size,add);
}
//////////////////////////////////////////////////////////////////////////////////////

__global__ void gpu_lasso_u_update(float *u,const float *xh, const float *z,const int size){
	int offset = blockIdx.x*blockDim.x + threadIdx.x;
	if(offset<size){
		u[offset]+=(xh[offset]-z[offset]);
	}
}
extern "C" void gpu_lasso_u_update_wrap(float *u,const float *xh, const float *z,const int size,const int numBlocks){
	gpu_lasso_u_update<<<numBlocks,THREADS>>>(u,xh,z,size);
}
//////////////////////////////////////////////////////////////////////////////////////

__global__ void gpu_lasso_objective_helper(float *v0,const float *v1,const int size){//(v0 - v1).^2, with A
	int offset = blockIdx.x*blockDim.x + threadIdx.x;
	if(offset<size){
		float t=v0[offset]-v1[offset];
		v0[offset]=t*t;
	}
}
extern "C" void gpu_lasso_objective_helper_wrap(float *v0,const float *v1,const int size,const int numBlocks){
	gpu_lasso_objective_helper<<<numBlocks,THREADS>>>(v0,v1,size);
}
//////////////////////////////////////////////////////////////////////////////////////

__global__ void gpu_lasso_h_helper(const float *z, const float *zold, float *v_result,const float _rho,const int size){
	int offset = blockIdx.x*blockDim.x + threadIdx.x;
	if(offset<size){
		v_result[offset]= -_rho*(z[offset]-zold[offset]);
	}
}
extern "C" void gpu_lasso_h_helper_wrap(const float *z, const float *zold, float *v_result,const float _rho,const int size,const int numBlocks){
	gpu_lasso_h_helper<<<numBlocks,THREADS>>>(z,zold,v_result,_rho,size);
}
//////////////////////////////////////////////////////////////////////////////////////

__global__ void gpu_group_lasso_shrinkage(const float *x, float *z_result,float temp,const int size){
	int offset = blockIdx.x*blockDim.x + threadIdx.x;
	if(offset<size){
		//float temp=(1.0f-kappa/e_norm);
		z_result[offset]= (temp>0.0f) ? (temp*x[offset]):0.0f;
	}
}
extern "C" void gpu_group_lasso_shrinkage_wrap(const float *x, float *z_result,const float kappa,const float e_norm,const int size,const int numBlocks){
	gpu_group_lasso_shrinkage<<<numBlocks,THREADS>>>(x,z_result,(1.0f-kappa/e_norm),size);
}
//////////////////////////////////////////////////////////////////////////////////////

__global__ void d_choldc_topleft(float *M, int boffset,const int N){
    int tx = threadIdx.x,ty = threadIdx.y;

    __shared__ float topleft[BLOCK_SIZE][BLOCK_SIZE+1];
	int idx0=ty+BLOCK_SIZE*boffset,adj=tx+BLOCK_SIZE*boffset;

    topleft[ty][tx]=M[idx0*N+adj];
    __syncthreads();

    float fac;
// in this loop tx labels column, ty row
    for(int k=0;k<BLOCK_SIZE;k++){
		__syncthreads();
		fac=rsqrtf(topleft[k][k]);
		__syncthreads();
		if((ty==k)&&(tx>=k)){
			topleft[tx][ty]=(topleft[tx][ty])*fac;
		}
		__syncthreads();
		if ((ty>=tx)&&(tx>k)){
			topleft[ty][tx]=topleft[ty][tx]-topleft[tx][k]*topleft[ty][k]; 
		}
    }
    __syncthreads();
// here, tx labels column, ty row	
    if(ty>=tx){
		M[idx0*N+adj]=topleft[ty][tx];
    }
}
extern "C" void d_choldc_topleft_wrap(float *M, int boffset,const int N,const dim3 t_block){
	d_choldc_topleft<<<1,t_block>>>(M,boffset,N);
}
//////////////////////////////////////////////////////////////////////////////////////


__global__ void d_choldc_strip(float *M,int boffset,const int N){
// +1 since blockoffset labels the "topleft" position
// and boff is the working position...
    int boffx = blockIdx.x+boffset+1; 
    int tx = threadIdx.x,ty = threadIdx.y;
	int idx0=ty+BLOCK_SIZE*boffset,adj=tx+BLOCK_SIZE*boffset;
	int idx1=ty+boffx*BLOCK_SIZE,adj1=tx+boffset*BLOCK_SIZE;

    __shared__ float topleft[BLOCK_SIZE][BLOCK_SIZE+1];
    __shared__ float workingmat[BLOCK_SIZE][BLOCK_SIZE+1];

    topleft[ty][tx]=M[idx0*N+adj];
// actually read in transposed...
    workingmat[tx][ty]=M[idx1*N+adj1];

    __syncthreads();
    // now we forward-substitute for the new strip-elements...
    // one thread per column (a bit inefficient I'm afraid)
    if(ty==0){
		float dotprod;
		for(int k=0;k<BLOCK_SIZE;k++){
			dotprod=0.0f;
			for (int m=0;m<k;m++){
				dotprod+=topleft[k][m]*workingmat[m][tx];
			}
			workingmat[k][tx]=(workingmat[k][tx]-dotprod)/topleft[k][k];
		}
    }
    __syncthreads();
// is correctly transposed...
    M[idx1*N+adj1]=workingmat[tx][ty];
}
extern "C" void d_choldc_strip_wrap(float *M, int boffset,const int N,const dim3 stripgrid,const dim3 t_block){
	d_choldc_strip<<<stripgrid,t_block>>>(M,boffset,N);
}
//////////////////////////////////////////////////////////////////////////////////////




