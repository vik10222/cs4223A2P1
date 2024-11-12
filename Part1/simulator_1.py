import sys
import logging
from collections import deque

# As of now, we are only using logging.warning for simplicity
# If it is changed to 'DEBUG', the terminal will be spammed
# To tackle this and get more nuanced logging, just alter the loggin.{keyword} to your liking in this script
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

# Primarily a placeholder for now, will become more important when we introduce multi-core support
class DRAM:
    def __init__(self):
        self.access_latency = 100

    def fetch(self, address):
        logging.debug(f"Fetching data from DRAM for address {hex(address)}")
        return self.access_latency

    def write_back(self, address, block_size):
        logging.debug(f"Writing back dirty block to DRAM for address {hex(address)}")
        return self.access_latency

class Cache:
    def __init__(self, size, block_size, associativity, dram):
        self.size = size
        self.block_size = block_size
        self.associativity = associativity
        self.num_sets = size // (block_size * associativity)
        self.cache = {i: deque(maxlen=associativity) for i in range(self.num_sets)}  # Cache structure with LRU for each set
        self.hit_count = 0
        self.miss_count = 0
        self.data_traffic = 0  # in bytes
        self.dram = dram
        self.write_back_cycles = 0

    def load(self, address):
        set_index, tag = self.get_set_index_and_tag(address)
        for block in self.cache[set_index]:
            if block['tag'] == tag:
                logging.debug(f"Cache hit for address {hex(address)} in set {set_index}")
                self.hit_count += 1
                self.update_lru(set_index, tag)
                return True, 0  # Hit, no additional cycles
        # Cache miss
        logging.debug(f"Cache miss for address {hex(address)} in set {set_index}")
        self.miss_count += 1
        cycles_spent = self.fetch_block(address)
        return False, cycles_spent

    def store(self, address):
        set_index, tag = self.get_set_index_and_tag(address)
        for block in self.cache[set_index]:
            if block['tag'] == tag:
                logging.debug(f"Cache hit for store at address {hex(address)} in set {set_index}")
                self.hit_count += 1
                self.update_lru(set_index, tag)
                block['dirty'] = True
                return True, 0  # Hit, no additional cycles
        # Cache miss
        logging.debug(f"Cache miss for store at address {hex(address)} in set {set_index}")
        self.miss_count += 1
        cycles_spent = self.fetch_block(address, is_store=True)
        return False, cycles_spent

    def fetch_block(self, address, is_store=False):
        set_index, tag = self.get_set_index_and_tag(address)
        cycles_spent = 0

        # Eviction logic
        if len(self.cache[set_index]) == self.associativity:
            evicted_block = self.cache[set_index].popleft()
            if evicted_block['dirty']:
                # Reconstruct the evicted block's address
                evicted_block_address = (evicted_block['tag'] * self.num_sets + set_index) * self.block_size
                logging.debug(f"Evicting dirty block with address {hex(evicted_block_address)} from set {set_index}")
                # Perform write-back to DRAM
                write_back_latency = self.dram.write_back(evicted_block_address, self.block_size)
                self.write_back_cycles += write_back_latency
                self.data_traffic += self.block_size  # Writing back dirty block
                cycles_spent += write_back_latency  # Add write-back latency to cycles_spent

        # Fetch the new block from DRAM
        fetch_latency = self.dram.fetch(address)
        self.data_traffic += self.block_size  # Fetching block from memory
        cycles_spent += fetch_latency  # Add fetch latency to cycles_spent

        # Add the new block to the cache
        new_block = {'tag': tag, 'dirty': is_store}
        self.cache[set_index].append(new_block)
        logging.debug(f"Fetched block with address {hex(address)} into set {set_index}, dirty: {is_store}")

        return cycles_spent  # Return the total cycles spent during fetch

    def get_set_index_and_tag(self, address):
        # Calculate the set index and tag based on address, block size, and number of sets
        block_address = address // self.block_size
        set_index = block_address % self.num_sets
        tag = block_address // self.num_sets
        return set_index, tag

    def update_lru(self, set_index, tag):
        # Move the block to the most recently used position
        set_associativity = self.cache[set_index]
        for block in set_associativity:
            if block['tag'] == tag:
                set_associativity.remove(block)
                set_associativity.append(block)
                break

class CPU:
    def __init__(self, cache, trace_file):
        self.cache = cache
        self.cycle_count = 0
        self.compute_cycles = 0
        self.load_store_instructions = 0
        self.idle_cycles = 0
        self.trace_file = trace_file

    def run(self):
        logging.info(f"Starting CPU execution with trace: {self.trace_file}")
        try:
            with open(self.trace_file, 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    if len(parts) != 2:
                        logging.warning(f"Invalid trace line: {line.strip()}")
                        continue
                    instruction, value = parts
                    instruction = int(instruction)
                    if instruction in [0, 1]:
                        address = int(value, 16)
                        if instruction == 0:  # Load
                            self.load(address)
                        elif instruction == 1:  # Store
                            self.store(address)
                    elif instruction == 2:  # Other instructions (compute cycles)
                        cycles = int(value, 16)
                        self.compute(cycles)
        except FileNotFoundError:
            logging.error(f"Trace file {self.trace_file} not found.")
            sys.exit(1)

    def load(self, address):
        logging.debug(f"CPU Load instruction at address {hex(address)}")
        self.load_store_instructions += 1
        hit, cycles_spent = self.cache.load(address)
        self.cycle_count += 1  # Cache access cycle
        if not hit:
            self.cycle_count += cycles_spent  # Add cycles spent during cache miss
            self.idle_cycles += cycles_spent  # Core is idle during cache operation

    def store(self, address):
        logging.debug(f"CPU Store instruction at address {hex(address)}")
        self.load_store_instructions += 1
        hit, cycles_spent = self.cache.store(address)
        self.cycle_count += 1  # Cache access cycle
        if not hit:
            self.cycle_count += cycles_spent  # Add cycles spent during cache miss
            self.idle_cycles += cycles_spent  # Core is idle during cache operation

    def compute(self, cycles):
        logging.debug(f"CPU Compute instruction with {cycles} cycles")
        self.compute_cycles += cycles
        self.cycle_count += cycles

class Simulator:
    def __init__(self, trace_folder, trace_file, cache_size=4096, block_size=32, associativity=2):
        self.dram = DRAM()
        self.cache = Cache(cache_size, block_size, associativity, self.dram)
        self.cpu = CPU(self.cache, f"{trace_folder}/{trace_file}_0.data.txt")

    def run(self):
        logging.info("Starting simulation...")
        self.cpu.run()
        self.generate_report()

    def generate_report(self):
        logging.info("Generating report...")
        print("===== Simulation Report =====")
        print(f"Total Execution Cycles: {self.cpu.cycle_count}")
        print(f"Compute Cycles: {self.cpu.compute_cycles}")
        print(f"Load/Store Instructions: {self.cpu.load_store_instructions}")
        print(f"Idle Cycles: {self.cpu.idle_cycles}")
        print(f"Cache Hits: {self.cache.hit_count}")
        print(f"Cache Misses: {self.cache.miss_count}")
        print(f"Data Traffic on Bus: {self.cache.data_traffic} bytes")
        print(f"Write-Back Cycles: {self.cache.write_back_cycles}")
        print("=============================")

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Cache Coherence Simulator')
    # Just a placeholder for now, will come in play during Phase 2
    # parser.add_argument('protocol', type=str, choices=['MESI', 'Dragon'], default='MESI', help='Coherence protocol (e.g., MESI, Dragon)')
    parser.add_argument('input_file', type=str, help='Input benchmark name (e.g., bodytrack)')
    parser.add_argument('cache_size', type=int, nargs='?', default=4096, help='Cache size in bytes (default: 4096)')
    parser.add_argument('associativity', type=int, nargs='?', default=2, help='Cache associativity (default: 2)')
    parser.add_argument('block_size', type=int, nargs='?', default=32, help='Cache block size in bytes (default: 32)')
    args = parser.parse_args()

    # Validate cache size, associativity, and block size
    if args.cache_size <= 0 or args.associativity <= 0 or args.block_size <= 0:
        logging.error("Cache size, associativity, and block size must be positive integers.")
        sys.exit(1)
    if args.cache_size % (args.block_size * args.associativity) != 0:
        logging.error("Cache size must be a multiple of (block_size * associativity).")
        sys.exit(1)

    ##### CHANGE THIS PATH TO MATCH YOUR FOLDER STRUCTURE ####
    trace_folder = '/Users/oysteinweibell/Desktop/MulticoreArch/assignment2/cs4223A2P1/traces'
    ##########################################################

    simulator = Simulator(trace_folder, args.input_file, args.cache_size, args.block_size, args.associativity)
    simulator.run()

if __name__ == "__main__":
    main()
