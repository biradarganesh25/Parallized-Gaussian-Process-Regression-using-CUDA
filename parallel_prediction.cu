#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <math.h>

#define debug 3
#define blk_size 256

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

float tol = 0.001;

using namespace std;

void fill_f(float *hf, float *hx, float *hy, int m){
    for(int i=0;i<m*m;i++){
        int r=i/m;
        int c=i%m;

        float di = (rand() / (RAND_MAX/0.1)) - 0.05;
        hf[i] = 1-(pow(hx[r]-0.5,2)+pow(hy[c]-0.5,2)) + di; 
    }
}

__global__  void calculate_A(float *dk, float *dx, float *dy, int m){
    int n=m*m;
    int tid=threadIdx.x;
    // blockDim.x will have total number of threads
    for(;tid<n*n;tid+=blockDim.x){
        // tid will be cur element of K that this thread will operate on

        int p=tid/n,q=tid%n;
        int x1=dx[p/m],y1=dy[p%m],x2=dx[q/m],y2=dy[q%m];

        float first_term = pow((x1-x2),2)/2;
        float second_term = pow((y1-y2),2)/2;
        float power_term = -(first_term+second_term);
        dk[tid]=exp(power_term)/sqrt(2*M_PI);
        // adding t*I to K, gives A
        if(p==q){
            dk[tid]+=0.01;
        }
    }    

    // return dk;
}

// blocked decomposition: the lu decomposition and the solve function are implemented
// in a blocked man out the logic is explained in that major project pdf.
__global__ void lu_decompose(float *da, int n){
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    // printf("num_threads %d\n",num_threads);

    for(int k=0;k<n-1;k++){
        __shared__ float pivotele;
        if(tid==0){
            pivotele=da[k*n+k];
        }
        __syncthreads();



        for(int col_blk_beg=k+1;col_blk_beg<n;col_blk_beg+=blk_size){

            __shared__ float first_col[blk_size];
            int rem_col=(n-col_blk_beg);
            // printf("rem_col %d\n", rem_col);
            // int blk_end_col = col_blk_beg+blk_size;
            for(int i=tid;i<min(blk_size,rem_col);i+=num_threads){
                // printf("i: %d\n",i);
                first_col[i]=da[(i+col_blk_beg)*n+k]/pivotele;
                // printf("inside col %f\n", da[(i+col_blk_beg)*n]);
            }
            __syncthreads();

            for(int pivot_blk_beg=k+1;pivot_blk_beg<n;pivot_blk_beg+=blk_size){
                __shared__ float pivot_blk[blk_size];
                int rem_pivot=(n-pivot_blk_beg);
                // int blk_end_pivot = pivot_blk_beg+blk_size;

                for(int i=tid;i<min(blk_size,rem_pivot);i+=num_threads){
                    pivot_blk[i]=da[k*n + (i+pivot_blk_beg)];
                }

                __syncthreads();

                int sub_matrix_rows=min(blk_size, rem_col);
                int sub_matrix_cols=min(blk_size, rem_pivot);
                // if(tid==0){
                //     printf("k %d col_blk_begin: %d pivot_blk_begin: %d rows: %d cols %d\n",k,col_blk_beg, pivot_blk_beg, sub_matrix_rows, sub_matrix_cols);
                // }

                for(int i=tid;i<sub_matrix_cols*sub_matrix_rows;i+=num_threads){
                    int row=i/sub_matrix_cols,col=i%sub_matrix_cols;
                    int actual_row=row+col_blk_beg, actual_col=col+pivot_blk_beg;
                    // if(tid==0){
                    //     printf("i %d row: %d col %d actualrow= %d actual_col %d\n", i, row, col, actual_row, actual_col);
                    // }
                    // da[actual_row*n+actual_col]-=da[k*n+actual_col]*col
                    da[actual_row*n+actual_col]-=pivot_blk[col]*first_col[row];
                }
                
                __syncthreads();
            }

            
            for(int i=tid;i<min(blk_size,rem_col);i+=num_threads){
                da[(i+col_blk_beg)*n+k]=first_col[i];
            }
        }
        
    }

}

//IMP: NUMBER of threads that work on this function 
//must be power of 2.
__global__ void lu_solve(float *dk, float *dz, float *df, int n){
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    int num_threads = blockDim.x;

    // sdata[0]=0;
    for(int i=0;i<n;i++){
        sdata[tid]=0;

        for(int k=tid;k<i;k+=num_threads)
        {
            sdata[tid]+=dk[i*n+k]*dz[k];
        }
        __syncthreads();



        for(unsigned int s=1; s < num_threads; s *= 2) {
            if (tid % (2 * s) == 0 )
            {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }

        if(tid==0){
            // float sum = 0;
            // for(int p=0;p<num_threads;p++){
            //     sum+=sdata[p];
            // }
            // printf("0th indx: %f\n",sdata[0]);
            dz[i]=df[i]-sdata[0];
            // dz[i]=df[i]-sum;
        }
    }

    for(int i=n-1;i>=0;i--){

        sdata[tid]=0;
        for(int k=tid+i+1;k<n;k+=num_threads){
            sdata[tid]+=dk[i*n+k]*dz[k];
        }
        __syncthreads();


        for(unsigned int s=1; s < num_threads; s *= 2) {
            if (tid % (2 * s) == 0 )
            {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }

        if(tid==0){
            // float sum = 0;
            // for(int p=0;p<n-i-1;p++){
            //     sum+=sdata[p];
            // }
            // dz[i]-=sum;
            dz[i]-=sdata[0];
            dz[i]/=dk[i*n+i];
        }
    }

}

void serial_lu_solver(float *a, float *x, float *b, int n){
    for (int i = 0; i < n; i++) {
        x[i] = b[i];

        for (int k = 0; k < i; k++)
            x[i] -= a[i*n+k] * x[k];
    }

    for (int i = n - 1; i >= 0; i--) {
        for (int k = i + 1; k < n; k++)
            x[i] -= a[i*n+k] * x[k];

        x[i] /= a[i*n+i];
    }
}

float *serial_lu_decompose(float *dh, int n){
    // float *decomposed = new float[n*n];
    // memcpy(decomposed, dh, n*n*sizeof(float));
    float *decomposed = dh;

    for(int k=0;k<n;k++){

        for(int i=k+1;i<n;i++){
            float temp = decomposed[i * n + k] / decomposed[k * n + k];
            for(int j=k+1;j<n;j++){
                decomposed[i*n+j]-=temp*decomposed[k*n+j];
            }
            decomposed[i * n + k] = temp;
        }
    }

    return decomposed;

}

// void compare_matrices(float *hk, float *decomposed, int n){

//     bool flag=false;
//     for(int i=0;i<n;i++){
//         for(int j=0;j<n;j++){
//             if(abs(hk[i*n+j]-decomposed[i*n+j])>tol){
//                 // printf("decomposition through serial and parallel difference more than tolerance!");
//                 printf("diff more than tol: parallel: %f serial: %f",hk[i*n+j],decomposed[i*n+j]);
//                 flag=true;
//             }
//         }
//     }
//     if(flag)
//     printf("parallel and serial lu decompositions are not same!\n");
//     else{
//         printf("parallel and serial decompositions are same!\n");
//     }
// }

void print_matrix(float *m, int n){
for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            cout << m[i * n + j] << " ";
        }
        cout << endl;
    }
    cout << endl;
}
void compare_host_device_matrix(float *ha, float *da, int n){

    float *temp_host = new float[n*n];
    cudaMemcpy(temp_host, da, n*n*sizeof(float), cudaMemcpyDeviceToHost);

    cout<<"host matrix: "<<endl;
    print_matrix(ha,n);

    cout<<"device matrix:"<<endl;
    print_matrix(temp_host,n);

    // bool flag=false;
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            if(abs(ha[i*n+j]-temp_host[i*n+j])>tol){
                // printf("decomposition through serial and parallel difference more than tolerance!");
                printf("diff more than tol: host: %f device: %f for i: %d, j: %d\n",ha[i*n+j],temp_host[i*n+j], i, j);
                // flag=true;
            }
        }
    }

    delete[] temp_host;

}

void print_vector(float *a, int n){

    for(int i=0;i<n;i++){
        cout<<a[i]<<" ";
    }
    cout<<endl;
}

void print_device_vector(float *da, int n){
    float *a = new float[n];
    cudaMemcpy(a, da, n*sizeof(float), cudaMemcpyDeviceToHost);

    print_vector(a, n);
    delete[] a;

}

void print_device_value(float *da, bool newline=true){
    float * a =  new float;
    cudaMemcpy(a, da, sizeof(float), cudaMemcpyDeviceToHost);

    cout<<*a;
    if(newline){
        cout<<endl;
    }
    delete a;
}


void compare_host_device_vector(float *hz, float *dz, int n){
    float *temp_host = new float[n];
    cudaMemcpy(temp_host, dz, n*sizeof(float), cudaMemcpyDeviceToHost);

    cout<<"host vector"<<endl;
    print_vector(hz, n);

    cout<<"device vector"<<endl;
    print_vector(temp_host,n);

    for(int i=0;i<n;i++){
        if(abs(hz[i] - temp_host[i])>tol){
            printf("hz %f temp_host %f for i= %d not equal!!\n", hz[i],temp_host[i],i);
        }
    }

    delete[] temp_host;
}

/*
functions are accurate compared to the host calculations. i consider host calculations as
this function was written to test whether the calculations done by the device
cereal and device calculations as parallel.

dk is first calculated on device, then this function is called. this copies dk to hk
and calls the decompose function on both host and device. these 2 matrices are 
compared with each other to ensure accuracy. after this lu solve is called
on both host and device and the solved array  is compared from both host and device 
for accuracy

*/
void test_lu_decompose_solve_without_custom(float *hk, float *hz, float *hf, float *dk, float *dz, float *df, int n){
    cout<<"Testing lu decompostion with in built values"<<endl;
    

    cudaMemcpy(hk,dk,n*n*sizeof(float),cudaMemcpyDeviceToHost);
    dim3 grid(1,1,1);
    dim3 block(32,1,1);

    lu_decompose<<<grid,block>>>(dk, n);
    serial_lu_decompose(hk, n);
    compare_host_device_matrix(hk, dk, n);

    serial_lu_solver(hk, hz, hf, n);

    lu_solve<<<grid, block, 2*sizeof(float)>>>(dk, dz, df, n);
    compare_host_device_vector(hz, dz, n);

}

/* this function reads hk, hf, copy them to the device memory, dk and df
then calls the decompose function both on host and device. the decomposed
matrix host and device are compared with each other. if they are sufficiently accurate
lu solve function is called on both host and device which fills hz and dz.
here hk, hf are A and B and hz is z, we solve Az=B. the solved vectors are 
also compared with each other.
*/
void test_lu_solve_decompose_with_custom_input(float *hk, float *hz, float *hf, float *dk, float *dz, float *df){
    int n;
    cin>>n;

    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            cin>>hk[i*n+j];
        }
    }

    for(int i=0;i<n;i++){
        cin>>hf[i];
    }

    dim3 grid(1,1,1);
    dim3 block(2,1,1);

    cudaMemcpy(dk, hk, n*n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(df, hf, n*sizeof(float), cudaMemcpyHostToDevice);

    serial_lu_decompose(hk, n);
    serial_lu_solver(hk, hz, hf, n);


    lu_decompose<<<grid,block>>>(dk,n);
    lu_solve<<<grid, block, 2*sizeof(float)>>>(dk, dz, df, n);

    compare_host_device_matrix(hk, dk, n);

    compare_host_device_vector(hz, dz, n);
}

__global__ void calculate_k_star(float xnew, float ynew, float *dkstar, float *dx, float *dy, int m){
    int n=m*m;


    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    for(int i=tid;i<n;i+=num_threads){
        int r=i/m;
        int c=i%m;
        float x=dx[r], y=dy[c];

        float power_term = -(pow(xnew-x,2)/2+pow(ynew-y,2)/2);
        dkstar[i]=exp(power_term)/sqrt(2*M_PI);
    }

}

//  this function should have the number of threads in power of 2
__global__ void predict(float *dkstar, float *dz, float *prediction, int n){

    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    int num_threads = blockDim.x;

    sdata[tid]=0;
    for(int i=tid;i<n;i+=num_threads){
        sdata[tid]+=dkstar[i]*dz[i];
    }
    __syncthreads();

    for(unsigned int s=1; s < num_threads; s *= 2) {
        if (tid % (2 * s) == 0 )
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if(tid==0){
        *prediction=sdata[0];
    }
}

int main(){
    int m;
    cout<<"Enter value of m"<<endl;
    cin>>m;
    int threads;
    cout<<"Enter number of threads"<<endl;
    cin>>threads;
    cout<<"m is "<<m<<" threads: "<<threads<<endl;
    int n=m*m;
    float *hx = new float[m];
    float *hy = new float[m];
    float *hf = new float[m*m];
    // float *hk = new float[n*n];
    // float *hz = new float[n];

    cudaError_t err;

    // if(debug > 0){
    //     for(int i=0;i<m;i++){
    //         hx[i]=i;
    //         hy[i]=i;
    //     }
    // }

    float mesh_width=1/(m+1);
    for(int i=0;i<m;i++){
        hx[i]=(i+1)*mesh_width;
        hy[i]=(i+1)*mesh_width;
    }

    fill_f(hf, hx, hy, m);
    // print_matrix(hf,m);

    float *dx, *dy, *df, *dk, *dz, *dkstar, *prediction;
    gpuErrchk(cudaMalloc((void **)&dx,m*sizeof(float)));
    gpuErrchk(cudaMalloc((void **)&dy,m*sizeof(float)));
    gpuErrchk(cudaMalloc((void **)&df,n*sizeof(float)));
    gpuErrchk(cudaMalloc((void **)&dk,n*n*sizeof(float)));
    gpuErrchk(cudaMalloc((void **)&dz,n*sizeof(float)));
    gpuErrchk(cudaMalloc((void **)&dkstar,n*sizeof(float)));
    gpuErrchk(cudaMalloc((void **)&prediction,sizeof(float)));



    cudaMemcpy(dx, hx, m*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dy, hy, m*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(df, hf, n*sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start_decompose, stop_decompose, start_solve, stop_solve;
    cudaEvent_t total_start, total_stop;
    cudaEventCreate(&start_decompose);
    cudaEventCreate(&stop_decompose);
    cudaEventCreate(&start_solve);
    cudaEventCreate(&stop_solve);
    cudaEventCreate(&total_start);
    cudaEventCreate(&total_stop);

    cudaFuncSetCacheConfig(calculate_k_star,cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(predict,cudaFuncCachePreferL1);

    dim3 grid(1,1,1);
    dim3 block(threads,1,1);
    cudaEventRecord(total_start);
    calculate_A<<<grid,block>>>(dk, dx, dy, m);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk(cudaDeviceSynchronize());

    cudaEventRecord(start_decompose);
    lu_decompose<<<grid,block, threads*sizeof(float)>>>(dk, n);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk(cudaDeviceSynchronize());
    cudaEventRecord(stop_decompose);

    cudaEventRecord(start_solve);
    lu_solve<<<grid,block,threads*sizeof(float)>>>(dk, dz, df, n);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk(cudaDeviceSynchronize());
    cudaEventRecord(stop_solve);

    
    // this prediction is to calculate the time taken
    // for predicting one value
    calculate_k_star<<<grid,block>>>(hx[0],hy[0], dkstar, dx, dy, m);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk(cudaDeviceSynchronize());
    predict<<<grid,block,threads*sizeof(float)>>>(dkstar, dz, prediction, n);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk(cudaDeviceSynchronize());
    cudaEventRecord(total_stop);


    // cout<<"Predicted Matrix: "<<endl;
    // this matrix prediction is to calculate the total 
    // accuracy of the parallel algorithm
    // float *predicted_matrix=new float[n];
    float *prediction_host=new float;
    float error=0;
    for(int i=0;i<m;i++){
    for(int j=0;j<m;j++){
        calculate_k_star<<<grid,block>>>(hx[i],hy[j], dkstar, dx, dy, m);
        predict<<<grid,block,threads*sizeof(float)>>>(dkstar, dz, prediction, n);
        
        // cudaMemcpy((void *)&predicted_matrix[i*m+j], prediction, sizeof(float), cudaMemcpyDeviceToHost);
        gpuErrchk(cudaMemcpy(prediction_host, prediction, sizeof(float), cudaMemcpyDeviceToHost));
        error += pow(*prediction_host-hf[i*m+j],2);
        // error += pow(predicted_matrix[i*m+j]-hf[i*m+j],2);
        // cout<<predicted_matrix[i*m+j]<<" ";
    }
    // cout<<endl;
    }
    delete prediction_host;
    cudaEventSynchronize(stop_decompose);
    cudaEventSynchronize(stop_solve);
    float accuracy = 1-sqrt(error/n);
    cout<<"Accuracy is "<<accuracy<<endl;
    float time_taken=0;
    cudaEventElapsedTime(&time_taken, start_decompose, stop_decompose);
    cout<<"time taken for decompse: "<< time_taken <<endl;
    cudaEventElapsedTime(&time_taken, start_solve, stop_solve);
    cout<<"time taken for solve: "<< time_taken <<endl;
    cudaEventElapsedTime(&time_taken, total_start, total_stop);
    cout<<"time taken for entire algorithm: "<< time_taken<<endl;



    // test_lu_decompose_solve_without_custom(hk, hz, hf, dk, dz, df, n);
    // test_lu_solve_without_custom(hk, hz, hf, dk, dz, df, n);
    // lu_decompose<<<grid,block>>>(dk, n);

    // test_lu_solve_decompose_with_custom_input(hk, hz, hf, dk, dz, df);



    delete[] hx, hy, hf;
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(df);
    cudaFree(dk);
    cudaFree(dz);
    cudaFree(dkstar);
    cudaFree(prediction);


}
