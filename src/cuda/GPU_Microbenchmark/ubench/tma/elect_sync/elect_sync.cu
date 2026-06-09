#include <stdio.h>

__global__ void elect_sync_kernel() {
    unsigned int is_leader;
    unsigned int dynamic_mask = __activemask(); //this is current mask -- better for configurability
    
    asm volatile(
        "{\n\t"
        "  .reg .b32 rx;\n\t"             // 1. Declare a dummy 32-bit register 'rx' for threadID
        "  .reg .pred p;\n\t"             // 2. Declare a temporary predicate register 'p'
        "  elect.sync rx|p, %1;\n\t"      // 3. Syntax requires BOTH separated by a '|'
        "  selp.u32 %0, 1, 0, p;\n\t"     // 4. If 'p' is true, set output to 1. Else, 0. (this controls the is_leader variable)
        "}"
        : "=r"(is_leader)                 // Output %0
        : "r"(dynamic_mask)               // Input %1
    );

    // Print the result to satisfy the compiler and verify execution (commented out for now since GPGPUsim doesn't support printf)
    printf("Thread %02d | Leader Status: %u\n", threadIdx.x, is_leader);
}


int main() {
    elect_sync_kernel<<<1, 32>>>();
    cudaDeviceSynchronize();
    return 0;
}