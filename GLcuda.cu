#include "stdafx.h"
#include "stdio.h"
#include <cuda.h>
#include <math_functions.h>
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

const int blockSizeLocal=64;

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

__global__ void gpu_vector_add(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ result,
	const int size,const bool add){
	const int offset = blockIdx.x*blockDim.x + threadIdx.x;
	const int adj= add ? 1:-1;
	if(offset<size){
		result[offset]=a[offset]+float(adj)*b[offset];
	}
	
}
extern "C" void gpu_vector_add_wrap(const float *a, const float *b, float *result, const int size, const bool add,const int numBlocks){
	gpu_vector_add<<<numBlocks,THREADS>>>(a,b,result,size,add);
}
//////////////////////////////////////////////////////////////////////////////////////

__global__ void gpu_lasso_u_update(float* __restrict__ u,const float* __restrict__ xh, const float* __restrict__ z,const int size){
	int offset = blockIdx.x*blockDim.x + threadIdx.x;
	if(offset<size){
		u[offset]+=(xh[offset]-z[offset]);
	}
}
extern "C" void gpu_lasso_u_update_wrap(float *u,const float *xh, const float *z,const int size,const int numBlocks){
	gpu_lasso_u_update<<<numBlocks,THREADS>>>(u,xh,z,size);
}
//////////////////////////////////////////////////////////////////////////////////////

__global__ void gpu_lasso_objective_helper(float* __restrict__ v0,const float* __restrict__ v1,const int size){//(v0 - v1).^2, with A
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

__global__ void gpu_lasso_h_helper(const float* __restrict__ z, const float* __restrict__ zold,
	float* __restrict__ v_result,const float _rho,const int size){
	int offset = blockIdx.x*blockDim.x + threadIdx.x;
	if(offset<size){
		v_result[offset]= -_rho*(z[offset]-zold[offset]);
	}
}
extern "C" void gpu_lasso_h_helper_wrap(const float *z, const float *zold, float *v_result,const float _rho,const int size,const int numBlocks){
	gpu_lasso_h_helper<<<numBlocks,THREADS>>>(z,zold,v_result,_rho,size);
}
//////////////////////////////////////////////////////////////////////////////////////

__global__ void gpu_group_lasso_shrinkage(const float* __restrict__ x, float* __restrict__ z_result,
	const float kappa,const float e_norm,const int size){
	int offset = blockIdx.x*blockDim.x + threadIdx.x;
	if(offset<size){
		float temp=(1.0f-kappa/e_norm);
		z_result[offset]= (temp>0.0f) ? (temp*x[offset]):0.0f;
	}
}
extern "C" void gpu_group_lasso_shrinkage_wrap(const float *x, float *z_result,const float kappa,const float e_norm,const int size,const int numBlocks){
	gpu_group_lasso_shrinkage<<<numBlocks,THREADS>>>(x,z_result,kappa,e_norm,size);
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


template<int blockWork>//this gets the section norms of (x_hat_+u)
__global__ void GPU_version(const float* __restrict__ x_hat,const float* __restrict__ u, float* __restrict__ nrms, const int* __restrict__ cuml_part){

		
		__shared__ int beg,end;
		__shared__ float tot[2];
		if(threadIdx.x==0){
			beg= (blockIdx.y==0) ? 0:cuml_part[blockIdx.y-1];
			end=cuml_part[blockIdx.y];
		}
		__syncthreads();

		const int offset= threadIdx.x+blockIdx.x*blockWork;
		const int warp_idx=threadIdx.x%32;//
		//perform reduction through block
		float val=0.0f,tmp;
		int i=0,idx;
		for(;i<(blockWork>>6);i++){
			idx=(beg+offset)+i*THREADS;
			if(idx<end){
				tmp=x_hat[idx]+u[idx];
				val+=(tmp*tmp);
			}
		}
		val += __shfl(val, warp_idx + 16);
		val += __shfl(val, warp_idx + 8);
		val += __shfl(val, warp_idx + 4);
		val += __shfl(val, warp_idx + 2);
		val += __shfl(val, warp_idx + 1);
		if(threadIdx.x==0 || threadIdx.x==32){
			tot[threadIdx.x>>5]=val;
		}
		__syncthreads();
		if(threadIdx.x==0){
			atomicAdd(&nrms[blockIdx.y],(tot[0]+tot[1]));
		}
}

__global__ void lastStep(const float* __restrict__ x_hat,const float* __restrict__ u, float* __restrict__ nrms, const int* __restrict__ cuml_part,
	const int start_adr){

		__shared__ int beg,end;
		__shared__ float tot[2];
		if(threadIdx.x==0){
			beg= (blockIdx.y==0) ? 0:cuml_part[blockIdx.y-1];
			end=cuml_part[blockIdx.y];
		}
		__syncthreads();
		const int offset=beg+threadIdx.x+start_adr;
		const int warp_idx=threadIdx.x%32;//

		int i=1,adj=0,idx;
		float val=0.0f,tmp;
		for(;(offset+adj)<end;i++){
			idx=offset+adj;
			tmp=x_hat[idx]+u[idx];
			val+=(tmp*tmp);
			adj=(i<<6);
		}

		val += __shfl(val, warp_idx + 16);
		val += __shfl(val, warp_idx + 8);
		val += __shfl(val, warp_idx + 4);
		val += __shfl(val, warp_idx + 2);
		val += __shfl(val, warp_idx + 1);

		if(threadIdx.x==0 || threadIdx.x==32){
			tot[threadIdx.x>>5]=val;
		}
		__syncthreads();

		if(threadIdx.x==0){
			tmp=sqrtf((nrms[blockIdx.y]+tot[0]+tot[1]));
			nrms[blockIdx.y]=tmp;
		}
}
//now have norm (x_hat+u) in sections, need to finish shrinkage and then fill in SUM of the norm of z
//each section z will have the max((1-kappa/(norm of (x_hat(sel)+u(sel))

// will have as many blockIdx.y as K, need each value of z to be that (x_hat[idx]+u[idx])*norm(blockIdx.y), while getting the sum of all z norms in t_obj
template<int blockWork>
__global__ void adj_z_shrink(const float* __restrict__ nrms,const float* __restrict__ x_hat, const float* __restrict__ u,
	float* __restrict__ z,const int* __restrict__ cuml_part, float* __restrict__ t_obj,
	const float kappa){

		__shared__ int beg,end;
		__shared__ float tmp_pos;
		__shared__ float tot[2];
		if(threadIdx.x==0){
			beg= (blockIdx.y==0) ? 0:cuml_part[blockIdx.y-1];
			end=cuml_part[blockIdx.y];
			tmp_pos= (nrms[blockIdx.y]>0.0f) ? max(0.0f,(1.0f-kappa/nrms[blockIdx.y])):0.0f;
		}
		__syncthreads();

		const int offset= threadIdx.x+blockIdx.x*blockWork;
		const int warp_idx=threadIdx.x%32;//
		//perform reduction through block
		float val=0.0f,tmp;
		int i=0,idx;
		for(;i<(blockWork>>6);i++){
			idx=(beg+offset)+i*THREADS;
			if(idx<end){
				tmp=(x_hat[idx]+u[idx])*tmp_pos;
				val+=(tmp*tmp);
				z[idx]=tmp;
			}
		}
		val += __shfl(val, warp_idx + 16);
		val += __shfl(val, warp_idx + 8);
		val += __shfl(val, warp_idx + 4);
		val += __shfl(val, warp_idx + 2);
		val += __shfl(val, warp_idx + 1);
		if(threadIdx.x==0 || threadIdx.x==32){
			tot[threadIdx.x>>5]=val;
		}
		__syncthreads();
		if(threadIdx.x==0){
			atomicAdd(&t_obj[blockIdx.y],(tot[0]+tot[1]));	
		}		
}

__global__ void lastStep_z(const float* __restrict__ nrms,const float* __restrict__ x_hat, const float* __restrict__ u,
	float* __restrict__ z,const int* __restrict__ cuml_part, float* __restrict__ t_obj,const float kappa,
	const int start_adr,float* __restrict__ z_norm_sum){

		__shared__ int beg,end;
		__shared__ float tmp_pos;
		__shared__ float tot[2];
		if(threadIdx.x==0){
			beg= (blockIdx.y==0) ? 0:cuml_part[blockIdx.y-1];
			end=cuml_part[blockIdx.y];
			tmp_pos= (nrms[blockIdx.y]>0.0f) ? max(0.0f,(1.0f-kappa/nrms[blockIdx.y])):0.0f;
		}
		__syncthreads();

		const int offset=beg+threadIdx.x+start_adr;
		const int warp_idx=threadIdx.x%32;//

		int i=1,adj=0,idx;
		float val=0.0f,tmp;
		//fill in last new values of z and finish up getting the norm
		for(;(offset+adj)<end;i++){
			idx=offset+adj;
			tmp=(x_hat[idx]+u[idx])*tmp_pos;
			val+=(tmp*tmp);
			z[idx]=tmp;
			adj=(i<<6);
		}

		val += __shfl(val, warp_idx + 16);
		val += __shfl(val, warp_idx + 8);
		val += __shfl(val, warp_idx + 4);
		val += __shfl(val, warp_idx + 2);
		val += __shfl(val, warp_idx + 1);

		if(threadIdx.x==0 || threadIdx.x==32){
			tot[threadIdx.x>>5]=val;
		}
		__syncthreads();

		if(threadIdx.x==0){
			tmp=sqrtf((t_obj[blockIdx.y]+tot[0]+tot[1]));//this is the z_norm
			atomicAdd(&z_norm_sum[0],tmp);	
		}
}

//Note: norm_s, t_obj_arr and z_norm_sum will be memset prior to helper function call
extern "C" void z_shrinkage_wrap(float *D_z,const float *x_hat, const float *D_u,float *norm_s, float *t_obj_arr, float *z_norm_sum,
	const int adj_size,dim3 &PGrid, const int *D_cuml_part, const int rem_start, const float kappa,const int num_blx,
	cudaError_t &err){

		if(adj_size==1){
			GPU_version<blockSizeLocal><<<PGrid,THREADS>>>(x_hat,D_u,norm_s,D_cuml_part);
		}else if(adj_size==2){
			GPU_version<blockSizeLocal*2><<<PGrid,THREADS>>>(x_hat,D_u,norm_s,D_cuml_part);
		}else if(adj_size==3){
			GPU_version<blockSizeLocal*4><<<PGrid,THREADS>>>(x_hat,D_u,norm_s,D_cuml_part);
		}else if(adj_size==4){
			GPU_version<blockSizeLocal*8><<<PGrid,THREADS>>>(x_hat,D_u,norm_s,D_cuml_part);
		}else{
			GPU_version<blockSizeLocal*16><<<PGrid,THREADS>>>(x_hat,D_u,norm_s,D_cuml_part);
		}

	

		PGrid.x=1;
		lastStep<<<PGrid,THREADS>>>(x_hat,D_u,norm_s,D_cuml_part,rem_start);
		
		//now have all block norms of (x_hat+u) in norm_s

		//now fill in z and get sum of z_norms for sections in z_norm_sum
		PGrid.x=num_blx;
		if(adj_size==1){
			adj_z_shrink<blockSizeLocal><<<PGrid,THREADS>>>(norm_s,x_hat,D_u,D_z,D_cuml_part,t_obj_arr,kappa);
		}else if(adj_size==2){
			adj_z_shrink<blockSizeLocal*2><<<PGrid,THREADS>>>(norm_s,x_hat,D_u,D_z,D_cuml_part,t_obj_arr,kappa);
		}else if(adj_size==3){
			adj_z_shrink<blockSizeLocal*4><<<PGrid,THREADS>>>(norm_s,x_hat,D_u,D_z,D_cuml_part,t_obj_arr,kappa);
		}else if(adj_size==4){
			adj_z_shrink<blockSizeLocal*8><<<PGrid,THREADS>>>(norm_s,x_hat,D_u,D_z,D_cuml_part,t_obj_arr,kappa);
		}else{
			adj_z_shrink<blockSizeLocal*16><<<PGrid,THREADS>>>(norm_s,x_hat,D_u,D_z,D_cuml_part,t_obj_arr,kappa);
		}
		

		PGrid.x=1;

		lastStep_z<<<PGrid,THREADS>>>(norm_s,x_hat,D_u,D_z,D_cuml_part,t_obj_arr,kappa,rem_start,z_norm_sum);
	
		//End partition loop, updated vector z by section and got the sum of the norms of Z into d_obj for later use by objective
}

//need norm of x, -z , (x-z), -rho*(z-zold), rho*u
__global__ void _get_norm_all(const float* __restrict__ x, const float * __restrict__ z, const float* __restrict__ zold, const float* __restrict__ u,
	float* __restrict__ xnorm, float* __restrict__ znorm, float* __restrict__ xznorm, float* __restrict__ rzznorm,float* __restrict__ runorm,
	const int length, const float _rho){

		const int offset=threadIdx.x+blockIdx.x*blockDim.x;
		const int warp_idx=threadIdx.x%32;

		__shared__ float x_sq_sum[2],
			z_sq_sum[2],
			x_minus_z_sum[2],
			neg_rho_zzold_sum[2],
			rho_u_sum[2];

		float xx=0.0f,zz=0.0f,xz=0.0f, zzold=0.0f,ru=0.0f, tmp=0.0f;

		if(offset<length){
			xx=x[offset];
			zz=z[offset];
			xz=(xx-zz)*(xx-zz);//rnorm calc
			tmp= -_rho*(zz-zold[offset]);
			zzold=tmp*tmp;//snorm calc
			xx*=xx;//norm x
			zz*=zz;//norm (-z)
			ru=_rho*u[offset];
			ru*=ru;//norm (rho*u)
		}
		for(int ii=16;ii>0;ii>>=1){
			xx += __shfl(xx, warp_idx + ii);
			zz += __shfl(zz, warp_idx + ii);
			xz += __shfl(xz, warp_idx + ii);
			zzold += __shfl(zzold, warp_idx + ii);
			ru += __shfl(ru, warp_idx + ii);
			
		}
		if(warp_idx==0){
			x_sq_sum[threadIdx.x>>5]=xx;
			z_sq_sum[threadIdx.x>>5]=zz;
			x_minus_z_sum[threadIdx.x>>5]=xz;
			neg_rho_zzold_sum[threadIdx.x>>5]=zzold;
			rho_u_sum[threadIdx.x>>5]=ru;		
		}
		__syncthreads();

		if(threadIdx.x==0){
			atomicAdd(&xnorm[0],(x_sq_sum[0]+x_sq_sum[1]));
			atomicAdd(&znorm[0],(z_sq_sum[0]+z_sq_sum[1]));
			atomicAdd(&xznorm[0],(x_minus_z_sum[0]+x_minus_z_sum[1]));
			atomicAdd(&rzznorm[0],(neg_rho_zzold_sum[0]+neg_rho_zzold_sum[1]));
			atomicAdd(&runorm[0],(rho_u_sum[0]+rho_u_sum[1]));

		}

}

extern "C" void get_multi_norms(const float *x, const float *z, const float *zold, const float *u,
	float *norm_arr,const float _rho, const int length){

		_get_norm_all<<<(length+THREADS-1)/THREADS,THREADS>>>(x,z,zold,u,&norm_arr[0], &norm_arr[1], &norm_arr[2],&norm_arr[3],&norm_arr[4], length,_rho);
}

__global__ void update_q(const float* __restrict__ Atb, const float* __restrict__  z, const float* __restrict__ u,
	float* __restrict__ q, const float rho, const int length){

		const int offset=threadIdx.x+blockIdx.x*blockDim.x;

		if(offset<length){
			q[offset]= (Atb[offset]+rho*(z[offset]-u[offset]));
		}
}
extern "C" void update_vector_q(const float *Atb, const float *z, const float *u, float *q, const float rho,const int length){
	update_q<<<(length+THREADS-1)/THREADS,THREADS>>>(Atb,z,u,q,rho,length);

}