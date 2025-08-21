#include <cuda_fp16.h>
#include <cuda_bf16.h>   //dsy: for bfloat16


//&cutlass_gemm_signal<128, 256, 32, 64, 64, 32, 16, 8, 16, 3, 6, 1>     //Algo[58] func
//( M, N, K, rLDN, cseg_gpu_ptr, a_ptr, b_ptr, c_ptr, mm_ptr, ra_ptr, if_monitor, this->gemm_stream);

template <int ThreadblockM, int ThreadblockN, int ThreadblockK, int WarpM, int WarpN, int WarpK, 
int InstructionM, int InstructionN, int InstructionK, int NumStages, int SwizzleSize, int SplitK>
void cutlass_gemm_signal(int M, int N, int K, int ReLDN, int* CommThr, nv_bfloat16* A, nv_bfloat16* B, nv_bfloat16* D, int* MM, int* RA, bool Monitor, cudaStream_t stream);
//void cutlass_gemm_signal(int M, int N, int K, int ReLDN, int* CommThr, half* A, half* B, half* D, int* MM, int* RA, bool Monitor, cudaStream_t stream);
//dsy: for bfloat16
