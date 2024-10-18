# Cache Coherence Simulator (Single-Core CPU)

This project implements a single-core CPU simulator with an L1 data cache and DRAM, adhering to cache policies such as write-back, write-allocate, and LRU replacement. It processes trace files and generates detailed reports based on cache performance, including cache hits, misses, idle cycles, and more.

## Features:
- **Single-core CPU simulation** with L1 data cache and DRAM interaction.
- **Cache policies**: Write-back, Write-allocate, and LRU replacement.
- **Cycle counting** for cache hits, misses, and DRAM latency.
- **Supports configurable cache settings**: cache size, associativity, and block size.
- **Trace-driven execution**: processes trace files to simulate memory operations.

## Requirements:
- Python 3.10
- No additional external libraries required

## Usage:
1. Clone the repository.
2. Run the following command in the terminal:
   ```bash
   python3 simulator.py <input_file> [cache_size] [associativity] [block_size]
   ```
   - **input_file**: Name of the benchmark trace (e.g., `bodytrack`)
   - **cache_size**: (Optional) Cache size in bytes (default: 4096)
   - **associativity**: (Optional) Cache associativity (default: 2)
   - **block_size**: (Optional) Block size in bytes (default: 32)

Example:
```bash
python3 simulator.py blackscholes 4096 2 32
```

## Output:
The simulator generates a report with the following details:
- Total execution cycles
- Compute cycles
- Load/Store instructions
- Idle cycles
- Cache hits and misses
- Data traffic (in bytes)
- Write-back cycles

## Trace File:
Place your trace files in the specified folder structure (update `trace_folder` in the script if necessary)
Ensure the file is named appropriately (e.g., `blackscholes_0.data.txt`).

## Logging:
- The script uses Python’s `logging` module, set to `WARNING` by default.
- To enable more detailed logs (e.g., debugging), modify the `logging.basicConfig` level in the script.
