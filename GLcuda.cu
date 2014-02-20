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

__global__ void set_ATA(const float* __restrict__ ATA, float* __restrict__ TempATA, const int N,
	const int padd_N){
		const int i=blockIdx.y;//0 to padd_N
		const int j=threadIdx.x+blockIdx.x*blockDim.x;//0 to padd_N
		if(j<padd_N){
			TempATA[i*padd_N+j]= (j<N && i<N) ? ATA[i*N+j]:(1.0f*float(int(i==j)));
		}
}
extern "C" void pad_ATA(const float *ATA, float *TempATA,  const int N,
	const int padd_N){
		dim3 grid((padd_N+THREADS-1)/THREADS,padd_N,1);
		set_ATA<<<grid,THREADS>>>(ATA,TempATA,N,padd_N);

}

//////////////////////////////////////////////////////////////////////////////////////
__global__ void d_choldc_topleft(float *M, int boffset,const int N){
    const int tx = threadIdx.x,ty = threadIdx.y;

    __shared__ float topleft[BLOCK_SIZE][BLOCK_SIZE+1];
	int idx0=ty+BLOCK_SIZE*boffset,adj=tx+BLOCK_SIZE*boffset;

    topleft[ty][tx]=M[idx0*N+adj];
    __syncthreads();

    float fac;
    for(int k=0;k<BLOCK_SIZE;k++){
		__syncthreads();
		fac=rsqrtf(topleft[k][k]);
		//__syncthreads();
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
    const int boffx = blockIdx.x+boffset+1; 
    const int tx = threadIdx.x,ty = threadIdx.y;
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
///////////////////////////////////////////////////////////////////////////

__global__ void adj_chol(const float* __restrict__ Pad_L, float* __restrict__ L,  const int N,const int padd_N){
		const int i=blockIdx.y;
		const int j=threadIdx.x+blockIdx.x*blockDim.x;
		if(j<N){
			L[i*N+j]=Pad_L[i*padd_N+j];
		}
}
extern "C" void get_L(const float *Pad_L, float *L, const int N,const int padd_N){
		dim3 grid((N+THREADS-1)/THREADS,N,1);
		adj_chol<<<grid,THREADS>>>(Pad_L,L,N,padd_N);
}


///////////////////////////////////////////////////////////////////////////
__global__ void update_q_multi(const float* __restrict__ Atb, const float* __restrict__  z, const float* __restrict__ u,
	float* __restrict__ q, const float rho, const int length,const int mask){

		if(mask&(1<<blockIdx.y))return;

		const int offset=threadIdx.x+blockIdx.x*blockDim.x;
		const int index=blockIdx.y*length+offset;
		if(offset<length){
			q[index]=(Atb[offset]+rho*(z[index]-u[index]));
		}
}
extern "C" void update_vector_q(const float *Atb, const float *z, const float *u, float *q, const float rho,const int length,const int num_lambdas,
	const dim3 &grid,const int mask){
	//dim3 grid((length+THREADS-1)/THREADS,num_lambdas,1);
	update_q_multi<<<grid,THREADS>>>(Atb,z,u,q,rho,length,mask);

}
////////////////////////////////////////////////////////////////////////////////////

__global__ void update_x_hat_multi(const float* __restrict__ x, const float* __restrict__ z, float* __restrict__ x_hat,
	const int length, const float alpha,const int mask){

		if(mask&(1<<blockIdx.y))return;

		const int offset=threadIdx.x+blockIdx.x*blockDim.x;
		const int index=blockIdx.y*length+offset;
		if(offset<length){
			x_hat[index]=alpha*x[index]+(1.0f-alpha)*z[index];
		}

}
extern "C" void x_hat_update_helper(const float *x, const float *zold, float *x_hat, const int length, const float alpha, const dim3 &grid,
	const int mask){
		update_x_hat_multi<<<grid,THREADS>>>(x,zold,x_hat, length, alpha,mask);
}
//////////////////////////////////////////////////////////////////////////////////////



template<int blockWork>//this gets the section norms of (x_hat_+u)
__global__ void GPU_version(const float* __restrict__ x_hat,const float* __restrict__ u, float* __restrict__ nrms, 
	const int* __restrict__ cuml_part,const int u_length, const int p_length,const int mask){

		if(mask&(1<<blockIdx.z))return;

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
				tmp=x_hat[blockIdx.z*u_length+idx]+u[blockIdx.z*u_length+idx];
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
			atomicAdd(&nrms[blockIdx.z*p_length+blockIdx.y],(tot[0]+tot[1]));
		}
}

__global__ void lastStep(const float* __restrict__ x_hat,const float* __restrict__ u, float* __restrict__ nrms, const int* __restrict__ cuml_part,
	const int start_adr,const int u_length, const int p_length,const int mask){

		if(mask&(1<<blockIdx.z))return;

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
			tmp=x_hat[blockIdx.z*u_length+idx]+u[blockIdx.z*u_length+idx];
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
			tmp=sqrtf((nrms[blockIdx.z*p_length+blockIdx.y]+tot[0]+tot[1]));
			nrms[blockIdx.z*p_length+blockIdx.y]=tmp;
		}
}
//now have norm (x_hat+u) in sections, need to finish shrinkage and then fill in SUM of the norm of z
//each section z will have the max((1-kappa/(norm of (x_hat(sel)+u(sel))

// will have as many blockIdx.y as K, need each value of z to be that (x_hat[idx]+u[idx])*norm(blockIdx.y), while getting the sum of all z norms in t_obj
template<int blockWork>
__global__ void adj_z_shrink(const float* __restrict__ nrms,const float* __restrict__ x_hat, const float* __restrict__ u,
	float* __restrict__ z,const int* __restrict__ cuml_part, float* __restrict__ t_obj,
	const float* __restrict__ lam_arr,const int u_length, const int p_length,
	const float rho,const int mask){

		if(mask&(1<<blockIdx.z))return;

		__shared__ int beg,end;
		__shared__ float tmp_pos;
		__shared__ float tot[2];
		if(threadIdx.x==0){
			beg= (blockIdx.y==0) ? 0:cuml_part[blockIdx.y-1];
			end=cuml_part[blockIdx.y];
			tmp_pos= (nrms[blockIdx.z*p_length+blockIdx.y]>0.0f) ? max(0.0f,(1.0f - (lam_arr[blockIdx.z]/rho)/nrms[blockIdx.z*p_length+blockIdx.y]  )):0.0f;
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
				tmp=(x_hat[blockIdx.z*u_length+idx]+u[blockIdx.z*u_length+idx])*tmp_pos;
				val+=(tmp*tmp);
				z[blockIdx.z*u_length+idx]=tmp;
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
			atomicAdd(&t_obj[blockIdx.z*p_length+blockIdx.y],(tot[0]+tot[1]));	
		}		
}

__global__ void lastStep_z(const float* __restrict__ nrms,const float* __restrict__ x_hat, const float* __restrict__ u,
	float* __restrict__ z,const int* __restrict__ cuml_part, float* __restrict__ t_obj,const float* __restrict__ lam_arr,
	const int start_adr,float* __restrict__ z_norm_sum,const int u_length, const int p_length,const float rho,
	const int mask){

		if(mask&(1<<blockIdx.z))return;

		__shared__ int beg,end;
		__shared__ float tmp_pos;
		__shared__ float tot[2];
		if(threadIdx.x==0){
			beg= (blockIdx.y==0) ? 0:cuml_part[blockIdx.y-1];
			end=cuml_part[blockIdx.y];
			tmp_pos= (nrms[blockIdx.z*p_length+blockIdx.y]>0.0f) ? max(0.0f,(1.0f- (lam_arr[blockIdx.z]/rho)/nrms[blockIdx.z*p_length+blockIdx.y]  ) ):0.0f;
		}
		__syncthreads();

		const int offset=beg+threadIdx.x+start_adr;
		const int warp_idx=threadIdx.x%32;//

		int i=1,adj=0,idx;
		float val=0.0f,tmp;
		//fill in last new values of z and finish up getting the norm
		for(;(offset+adj)<end;i++){
			idx=offset+adj;
			tmp=(x_hat[blockIdx.z*u_length+idx]+u[blockIdx.z*u_length+idx])*tmp_pos;
			val+=(tmp*tmp);
			z[blockIdx.z*u_length+idx]=tmp;
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
			tmp=sqrtf((t_obj[blockIdx.z*p_length+blockIdx.y]+tot[0]+tot[1]));//this is the z_norm
			atomicAdd(&z_norm_sum[blockIdx.z],tmp);	
		}
}
//z_shrinkage_wrap(D_z,x_hat,D_u,norm_s,t_obj_arr,z_norm_sum,adj_size,PGrid,D_cuml_part,lam_arr,rem_start,_rho,num_blx,err,Acols,Psize);
//Note: norm_s, t_obj_arr and z_norm_sum will be memset prior to helper function call
extern "C" void z_shrinkage_wrap(float *D_z,const float *x_hat, const float *D_u,float *norm_s, float *t_obj_arr, float *z_norm_sum,
	const int adj_size,dim3 PGrid, const int *D_cuml_part,const float *lam_arr, const int rem_start, const float rho,const int num_blx,
	cudaError_t &err,const int u_length,const int p_length,const int mask){

		if(adj_size==1){
			GPU_version<blockSizeLocal><<<PGrid,THREADS>>>(x_hat,D_u,norm_s,D_cuml_part,u_length,p_length,mask);
		}else if(adj_size==2){
			GPU_version<blockSizeLocal*2><<<PGrid,THREADS>>>(x_hat,D_u,norm_s,D_cuml_part,u_length,p_length,mask);
		}else if(adj_size==3){
			GPU_version<blockSizeLocal*4><<<PGrid,THREADS>>>(x_hat,D_u,norm_s,D_cuml_part,u_length,p_length,mask);
		}else if(adj_size==4){
			GPU_version<blockSizeLocal*8><<<PGrid,THREADS>>>(x_hat,D_u,norm_s,D_cuml_part,u_length,p_length,mask);
		}else{
			GPU_version<blockSizeLocal*16><<<PGrid,THREADS>>>(x_hat,D_u,norm_s,D_cuml_part,u_length,p_length,mask);
		}

	

		PGrid.x=1;
		lastStep<<<PGrid,THREADS>>>(x_hat,D_u,norm_s,D_cuml_part,rem_start,u_length,p_length,mask);
		
		//now have all block norms of (x_hat+u) in norm_s

		//now fill in z and get sum of z_norms for sections in z_norm_sum
		PGrid.x=num_blx;
		if(adj_size==1){
			adj_z_shrink<blockSizeLocal><<<PGrid,THREADS>>>(norm_s,x_hat,D_u,D_z,D_cuml_part,t_obj_arr,lam_arr,u_length,p_length,rho,mask);
		}else if(adj_size==2){
			adj_z_shrink<blockSizeLocal*2><<<PGrid,THREADS>>>(norm_s,x_hat,D_u,D_z,D_cuml_part,t_obj_arr,lam_arr,u_length,p_length,rho,mask);
		}else if(adj_size==3){
			adj_z_shrink<blockSizeLocal*4><<<PGrid,THREADS>>>(norm_s,x_hat,D_u,D_z,D_cuml_part,t_obj_arr,lam_arr,u_length,p_length,rho,mask);
		}else if(adj_size==4){
			adj_z_shrink<blockSizeLocal*8><<<PGrid,THREADS>>>(norm_s,x_hat,D_u,D_z,D_cuml_part,t_obj_arr,lam_arr,u_length,p_length,rho,mask);
		}else{
			adj_z_shrink<blockSizeLocal*16><<<PGrid,THREADS>>>(norm_s,x_hat,D_u,D_z,D_cuml_part,t_obj_arr,lam_arr,u_length,p_length,rho,mask);
		}
		

		PGrid.x=1;

		lastStep_z<<<PGrid,THREADS>>>(norm_s,x_hat,D_u,D_z,D_cuml_part,t_obj_arr,lam_arr,rem_start,z_norm_sum,u_length,p_length,rho,mask);
	
		//End partition loop, updated vector z by section and got the sum of the norms of Z into d_obj for later use by objective
}
///////////////////////////////////////////////////////////////////////////////

__global__ void gpu_lasso_u_update_multi(float* __restrict__ u,const float* __restrict__ xh, const float* __restrict__ z,const int length,
	const int mask){

		if(mask&(1<<blockIdx.y))return;

		const int offset=threadIdx.x+blockIdx.x*blockDim.x;
		const int index=blockIdx.y*length+offset;
		if(offset<length){
			u[index]+=(xh[index]-z[index]);
		}
}
extern "C" void gpu_lasso_u_update_wrap(float *u,const float *xh, const float *z,const int length,const dim3 &grid,const int mask){
	gpu_lasso_u_update_multi<<<grid,THREADS>>>(u,xh,z,length,mask);
}

/////////////////////////////////////////////////////////////////////////////////////
//need norm of x, -z , (x-z), -rho*(z-zold), rho*u
__global__ void _get_norm_all(const float* __restrict__ x, const float * __restrict__ z, const float* __restrict__ zold, const float* __restrict__ u,
	float* __restrict__ xnorm, float* __restrict__ znorm, float* __restrict__ xznorm, float* __restrict__ rzznorm,float* __restrict__ runorm,
	const int length, const float _rho,const int mask){

		if(mask&(1<<blockIdx.y))return;

		const int offset=threadIdx.x+blockIdx.x*blockDim.x;
		const int warp_idx=threadIdx.x%32;
		__shared__ float x_sq_sum[2],
			z_sq_sum[2],
			x_minus_z_sum[2],
			neg_rho_zzold_sum[2],
			rho_u_sum[2];

		float xx=0.0f,zz=0.0f,xz=0.0f, zzold=0.0f,ru=0.0f, tmp=0.0f;

		if(offset<length){
			xx=x[blockIdx.y*length+offset];
			zz=z[blockIdx.y*length+offset];
			xz=(xx-zz)*(xx-zz);//rnorm calc
			tmp= -_rho*(zz-zold[blockIdx.y*length+offset]);
			zzold=tmp*tmp;//snorm calc
			xx*=xx;//norm x
			zz*=zz;//norm (-z)
			ru=_rho*u[blockIdx.y*length+offset];
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
			atomicAdd(&xnorm[blockIdx.y],(x_sq_sum[0]+x_sq_sum[1]));
			atomicAdd(&znorm[blockIdx.y],(z_sq_sum[0]+z_sq_sum[1]));
			atomicAdd(&xznorm[blockIdx.y],(x_minus_z_sum[0]+x_minus_z_sum[1]));
			atomicAdd(&rzznorm[blockIdx.y],(neg_rho_zzold_sum[0]+neg_rho_zzold_sum[1]));
			atomicAdd(&runorm[blockIdx.y],(rho_u_sum[0]+rho_u_sum[1]));

		}

}

extern "C" void get_multi_norms(const float *x, const float *z, const float *zold, const float *u,
	float *norm_arr,const float _rho, const int length,const int num_lambdas,const int mask){

		dim3 grid((length+THREADS-1)/THREADS,num_lambdas,1);

		_get_norm_all<<<grid,THREADS>>>(x,z,zold,u,(float*)&norm_arr[0], (float*)&norm_arr[num_lambdas],
			(float*)&norm_arr[num_lambdas*2],(float*)&norm_arr[num_lambdas*3],
			(float*)&norm_arr[num_lambdas*4], length,_rho,mask);
}


