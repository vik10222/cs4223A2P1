# Cache Coherence Simulator (Multi-Core CPU)

This project simulates a multi-core CPU environment with L1 caches and DRAM, implementing MESI and Dragon cache coherence protocols. The simulator processes trace files to model memory operations and cache behavior in a shared memory environment, offering insights into cache coherence performance, data traffic, and cycle-based metrics.

## Features

- **Multi-core CPU simulation** with configurable cache coherence protocols (MESI, Dragon) and shared memory interactions.
- **Cache coherence protocols**: MESI and Dragon, with support for write-back, write-allocate, and LRU replacement policies.
- **Cycle-based performance metrics**: Tracks execution cycles, cache hits, misses, idle cycles, and data traffic.
- **Configurable cache settings**: Cache size, associativity, and block size are fully customizable.
- **Trace-driven simulation**: Executes instructions from trace files, supporting realistic modeling of memory accesses and coherence.

## Requirements

- **Python 3.10** or later
- No additional external libraries are required

## Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/vik10222/cs4223A2P1.git
   cd cs4223A2P1/Part2
   ```
2. **Ensure Python version**: Python 3.10 or newer is required for this script.

## Usage

Run the simulator using the following command:

```bash
python3 simulator_2.py <protocol> <input_file> [cache_size] [associativity] [block_size]
```

### Parameters

- **protocol**: Specify the cache coherence protocol (`MESI` or `Dragon`).
- **input_file**: Name of the benchmark trace file without core identifiers (e.g., `bodytrack` for `bodytrack_0.data.txt`, `bodytrack_1.data.txt`).
- **cache_size** (optional): Cache size in bytes (default: `4096`).
- **associativity** (optional): Cache associativity (default: `2`).
- **block_size** (optional): Block size in bytes (default: `32`).

### Example Command

```bash
python3 simulator_2.py MESI blackscholes 4096 2 32
```

### Trace Files

Place your trace files in the specified folder, ensuring the `trace_folder` variable in `simulator.py` points to the correct location (default is `/Users/oysteinweibell/Desktop/MulticoreArch/assignment2/cs4223A2P1/traces`). Trace files should be named in the format `<input_file>_<core_number>.data.txt`, e.g., `blackscholes_0.data.txt`, `blackscholes_1.data.txt`.

## Simulator Output

The simulator generates a report with the following details:

- **Overall Execution Cycles**: Total cycles taken for all cores to complete.
- **Total Data Traffic on Bus**: Data transferred over the bus in bytes.
- **Invalidations (MESI) / Updates (Dragon)**: Count of cache invalidations (MESI) or updates (Dragon) due to coherence.
- **Per-CPU Metrics**:
  - **Total Cycles**: Total cycles per core.
  - **Compute Cycles**: Cycles spent in computation (non-memory operations).
  - **Load/Store Instructions**: Number of load and store instructions executed.
  - **Idle Cycles**: Cycles spent idle due to memory requests.
  - **Cache Hits and Misses**: Counts and rates of cache hits and misses.
  - **Private and Shared Accesses**: Breakdown of private vs. shared cache accesses.
  - **Adoptions**: Number of adoptions of cache lines between caches that mitigated a DRAM writeback/flush.
  - **Useless adoptions**: Number of cache lines that weren't accessed before eviction after getting adopted.