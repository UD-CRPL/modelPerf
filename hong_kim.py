import numpy as np

Active_blocks_per_SM = 5
num_Comp_insts = 27 # 9 Computation instructions times 3
num_Threads_per_warp = 32
num_Issue_cycles = 4


##### Change Per GPU ######## NVIDIA FX5600
Freq = 1 # Frequency of the GPU in GHz
Mem_Bandwidth = 80 #GB/s
Mem_LD = 420 # DRAM access latency (machine configuration)
Departure_del_coal = 4 # Delay between two coalesced memory transactions
Departure_del_uncoal = 10 # Delay between two uncoalesced memory transactions

num_Threads_per_block = 128 # Number of threads per block Program configuration
num_Blocks = 80 # Number of blocks Program configuration

num_Active_SMs = 16 # Number of active SMs Calculated based on machine resources
num_Active_blocks_per_SM = 5 # Number of active blocks per SM Calculated based on machine resources
#############################

##### Change Per GPU ######## NVIDIA V100
Freq = 1.53 # Frequency of the GPU in GHz
Mem_Bandwidth = 900 #GB/s
Mem_LD = 500 # DRAM access latency (machine configuration)
Departure_del_coal = 2 # Delay between two coalesced memory transactions
Departure_del_uncoal = 8 # Delay between two uncoalesced memory transactions

num_Threads_per_block = 256 # Number of threads per block Program configuration
num_Blocks = 160 # Number of blocks Program configuration

num_Active_SMs = 80 # Number of active SMs Calculated based on machine resources
num_Active_blocks_per_SM = 5 # Number of active blocks per SM Calculated based on machine resources
#############################


num_Active_warps_per_SM =  128/32 * 5 #num_Active_blocks_per_SM / num_warps_per_block # Number of active warps per SM (Active_blocks_per_SM *  num_warps_per_block)

num_Comp_insts =  27 # Number of computation instructions in one thread from source code
num_Uncoal_Mem_insts = 6 # Number of uncoalesced memory instructions in one thread from source code
num_Coal_Mem_insts =  0 # Number of coalesced memory instructions in one thread from source code
num_Mem_insts = num_Uncoal_Mem_insts + num_Coal_Mem_insts # Number of memory instructions in one thread from source code
num_Total_insts =  num_Comp_insts + num_Mem_insts # Comp_insts + Mem_insts
num_Synch_insts = 2 * 3 # Number of synchronization instructions in one thread from source code


num_Coal_per_mw = 1 # Number of coalesced memory instructions per memory warp
num_Uncoal_per_mw = 32 # Number of uncoalesced memory instructions per memory warp
Load_bytes_per_warp = 128 #B=4B * 32 # Number of bytes per warp (Data size 4B * num_Threads_per_warp)



def gpu_performance_model(
    instruction_count,  # Total number of instructions
    issue_rate,         # Instructions per cycle
    memory_latency,     # Latency per memory access (cycles)
    mlp,               # Memory-Level Parallelism
    tlp,               # Thread-Level Parallelism
    compute_intensity   # Ratio of compute to memory operations
):
    """
    Computes the execution time based on an analytical GPU model.
    """
    N = num_Active_warps_per_SM

    Weight_uncoal = num_Uncoal_Mem_insts / (num_Uncoal_Mem_insts + num_Coal_Mem_insts)

    Weight_coal = num_Coal_Mem_insts / (num_Uncoal_Mem_insts + num_Coal_Mem_insts)

    Departure_delay = (Departure_del_uncoal * num_Uncoal_per_mw) * Weight_uncoal + Departure_del_coal * Weight_coal

    Mem_L_Unncoal = Mem_LD + (num_Uncoal_per_mw - 1) * Departure_del_uncoal
                    
    Mem_L_Coal = Mem_LD 

    Mem_L = Mem_L_Unncoal * Weight_uncoal + Mem_L_Coal * Weight_coal

    MWP_Without_BW_full = Mem_L / Departure_delay

    BW_per_warp = (Freq * Load_bytes_per_warp) / Mem_L

    MWP_peak_BW = Mem_Bandwidth / (BW_per_warp * num_Active_SMs)

    MWP_Without_BW = min(MWP_Without_BW_full, num_Active_warps_per_SM)

    MWP = min(MWP_Without_BW, MWP_peak_BW, N)
    
    print(f"MWP: {MWP}")

    Comp_cycles = num_Issue_cycles * num_Total_insts
    
    print(f"Comp cycles: {Comp_cycles}")

    Mem_cycles = Mem_L_Unncoal * num_Uncoal_Mem_insts + Mem_L_Coal * num_Coal_Mem_insts
    
    print(f"Mem cycles: {Mem_cycles}")

    CWP_full = (Mem_cycles + Comp_cycles) / Comp_cycles
    
    print(f"CWP_full: {CWP_full}")

    CWP = min(CWP_full, N)
    
    print(f"CWP: {CWP}")

    num_Rep = num_Blocks / (num_Active_blocks_per_SM * num_Active_SMs)

    print(f"Rep: {num_Rep}")

    if (MWP == num_Active_warps_per_SM) and (CWP == num_Active_warps_per_SM): 
      Exec_cycles_app = (Mem_cycles + Comp_cycles + (Comp_cycles/num_Mem_insts) * (MWP - 1)) * num_Rep

    elif (CWP >= MWP) or (Comp_cycles > Mem_cycles):
      Exec_cycles_app = (Mem_cycles * (N/MWP) + (Comp_cycles/num_Mem_insts) * (MWP - 1)) * num_Rep
      print(f"Mem_cycles: {Mem_cycles}")
      print(f"Comp_cycles: {Comp_cycles}")
      print(f"N: {N}")
      print(f"MWP: {MWP}")
      print(f"num_Mem_insts: {num_Mem_insts}")
      print(f"Execution cycles: {Exec_cycles_app}")

    elif (MWP > CWP):
      Exec_cycles_app = (Mem_L + Comp_cycles * N) * num_Rep

    Synch_cost = Departure_delay * (MWP - 1) * num_Synch_insts * num_Active_blocks_per_SM * num_Rep

    print(f"Departure Delay: {Departure_delay}")
    print(f"MWP: {MWP}")
    print(f"num_Synch_insts: {num_Synch_insts}")
    print(f"Active blocks per SM: {num_Active_blocks_per_SM}")
    print(f"num_Rep: {num_Rep}")
    print(f"Synch cost: {Synch_cost}")

    Exec_cycles_with_synch = Exec_cycles_app + Synch_cost
    
    return Exec_cycles_with_synch

# Example usage
if __name__ == "__main__":
    instruction_count = 1e9  # 1 billion instructions
    issue_rate = 8           # 8 instructions per cycle
    memory_latency = 400     # 400 cycles
    mlp = 4                  # 4 memory accesses in parallel
    tlp = 32                 # 32 threads per warp
    compute_intensity = 0.7  # 70% compute, 30% memory ops
    
    exec_time = gpu_performance_model(instruction_count, issue_rate, memory_latency, mlp, tlp, compute_intensity)
    print(f"Estimated Execution Time: {exec_time:.2f} cycles")
