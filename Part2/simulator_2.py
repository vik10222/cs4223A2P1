import sys
import logging
import math
from collections import deque, defaultdict
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Set

class CacheState(Enum):
    """Cache states for MESI and Dragon protocols"""
    INVALID = auto()
    # MESI states
    MESI_MODIFIED = auto()
    MESI_EXCLUSIVE = auto()
    MESI_SHARED = auto()
    # Dragon states
    DRAGON_MODIFIED = auto()
    DRAGON_EXCLUSIVE = auto()
    DRAGON_SHARED_CLEAN = auto()
    DRAGON_SHARED_MODIFIED = auto()

class Operation(Enum):
    """Types of cache operations"""
    LOAD = 'load'
    STORE = 'store'

class BusTransaction(Enum):
    """Types of bus transactions"""
    BUS_WRITEBACK = auto()
    BUS_READ = auto()
    # MESI-specific bus transactions
    BUS_READ_X = auto()
    BUS_UPGRADE = auto()
    FLUSH_OPT = auto()
    # Dragon-specific bus transactions
    BUS_UPDATE = auto()

class DRAM:
    def __init__(self):
        self.access_latency = 100

    def fetch(self, address: int, size: int) -> int:
        return self.access_latency

    def write_back(self, address: int, size: int) -> int:
        return self.access_latency

class AddressHandler:
    def __init__(self, block_size: int, num_sets: int):
        if (block_size & (block_size - 1)) != 0:
            raise ValueError("Block size must be a power of 2")
        if (num_sets & (num_sets - 1)) != 0:
            raise ValueError("Number of sets must be a power of 2")
        
        self.block_size = block_size
        self.num_sets = num_sets
        self.block_offset_bits = int(math.log2(block_size))
        self.index_bits = int(math.log2(num_sets))
        self.tag_bits = 32 - self.block_offset_bits - self.index_bits

    def parse_address(self, address: int) -> Tuple[int, int, int]:
        address &= 0xFFFFFFFF  # Ensure 32-bit
        offset = address & (self.block_size - 1)
        index = (address >> self.block_offset_bits) & (self.num_sets - 1)
        tag = address >> (self.block_offset_bits + self.index_bits)
        return tag, index, offset

    def get_block_address(self, address: int) -> int:
        return address & ~(self.block_size - 1)

class CacheStatistics:
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.private_accesses = 0
        self.shared_accesses = 0
        self.total_cycles = 0

    def record_access(self, hit: bool, block_addr: int, state: Optional[CacheState] = None):
        if hit:
            self.hits += 1
        else:
            self.misses += 1

    def get_miss_rate(self) -> float:
        total = self.hits + self.misses
        return self.misses / total if total > 0 else 0

    def get_stats(self) -> Dict:
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': 1 - self.get_miss_rate(),
            'private_ratio': self.private_accesses / (self.private_accesses + self.shared_accesses)
                           if (self.private_accesses + self.shared_accesses) > 0 else 0
        }

class Bus:
    def __init__(self, block_size: int, dram: DRAM):
        self.busy = False
        self.queue: deque = deque()
        self.current_transaction = None
        self.transfer_cycles_remaining = 0
        self.block_size = block_size
        self.word_size = 4
        self.data_traffic = 0
        self.stats = {
            'invalidations': 0,
            'updates': 0
            }
        self.caches = [] 
        self.dram = dram
        self.dram_queue: deque = deque()
        self.dram_cycles_remaining = 0
        self.current_dram_transaction = None
        self.dram_pending_ack = False

    def is_busy(self) -> bool:
        return self.busy or self.transfer_cycles_remaining > 0 or self.dram_cycles_remaining > 0

    def grant(self):
        self.current_transaction = self.queue.popleft()
        self.busy = True
        return self.current_transaction 

    def release(self):
        self.current_transaction = None
        self.busy = False

    def request(self, cpu_id: int, transaction: Dict):
        self.queue.append((cpu_id, transaction))

    def step(self) -> bool:
        if self.transfer_cycles_remaining > 0:
            self.transfer_cycles_remaining -= 1
            if self.transfer_cycles_remaining == 0:
                self.complete_transaction()
            return True

        if self.current_transaction:
            transaction = self.current_transaction[1]
            transaction_type = transaction['type']
            address = transaction['address']
            requesting_cpu_id = self.current_transaction[0]

            snoop_hit = False
            for cache in self.caches:
                if cache.cpu_id != requesting_cpu_id:
                    if cache.snoop(transaction_type, address):
                        snoop_hit = True

            requesting_cache = next((c for c in self.caches if c.cpu_id == requesting_cpu_id), None)
            if requesting_cache:
                requesting_cache.handle_bus_transaction(snoop_hit)

            self.transfer_cycles_remaining = self.calculate_transfer_cycles(transaction)
            self.update_traffic_stats(transaction)

            if self.transfer_cycles_remaining == 0:
                self.complete_transaction()
                return False 
            return True

        if self.queue and not self.busy:
            self.grant()
            return True

        return False

    def calculate_transfer_cycles(self, transaction):
        if transaction['type'] == BusTransaction.BUS_UPDATE:
            # Sending a single word update takes 2 cycles
            return 2
        elif transaction['type'] in [BusTransaction.BUS_READ, BusTransaction.BUS_READ_X]:
            # Sending an address for read or read exclusive takes 1 cycle
            return 1
        elif transaction['type'] == BusTransaction.BUS_WRITEBACK:
            # Writing back a dirty block to memory incurs DRAM access latency
            return self.dram.access_latency 
        elif transaction['type'] == BusTransaction.BUS_UPGRADE:
            # Sending an address to upgrade a block takes 1 cycle
            return 1 
        elif transaction['type'] == BusTransaction.FLUSH_OPT:
            # Sending a cache block with N words takes 2N cycles
            words_in_block = self.block_size // self.word_size
            return 2 * words_in_block
        # Default case: no cycles
        return 0

    def update_traffic_stats(self, transaction: Dict):
        if transaction['type'] in [BusTransaction.BUS_READ_X, BusTransaction.BUS_UPGRADE]:
            self.stats['invalidations'] += 1
        elif transaction['type'] == BusTransaction.BUS_WRITEBACK:
            self.stats['invalidations'] += 1
            self.data_traffic += self.block_size
        elif transaction['type'] == BusTransaction.FLUSH_OPT:
            self.data_traffic += self.block_size
        elif transaction['type'] == BusTransaction.BUS_UPDATE:
            self.data_traffic += self.word_size
            self.stats['updates'] += 1

    def complete_transaction(self):
        transaction = self.current_transaction[1]
        cpu_id = self.current_transaction[0]
        if transaction['type'] == BusTransaction.BUS_WRITEBACK:
            self.dram.write_back(transaction['address'], self.block_size)
            cache = next((c for c in self.caches if c.cpu_id == cpu_id), None)
            if cache:
                cache.notify_write_back_complete()
        self.release()

class CacheBlock:
    def __init__(self, tag, set_index):
        self.tag = tag
        self.set_index = set_index
        self.state = CacheState.INVALID
        self.dirty = False
        self.address = None 

    def __eq__(self, other):
        return isinstance(other, CacheBlock) and self.tag == other.tag and self.set_index == other.set_index

    def __hash__(self):
        return hash((self.tag, self.set_index))

class Cache:
    def __init__(self, cpu_id, size, block_size, associativity, bus, dram, protocol):
        self.cpu_id = cpu_id
        self.word_size = 4
        self.size = size
        self.block_size = block_size
        self.associativity = associativity
        self.num_sets = size // (block_size * associativity)
        self.cache = {i: deque(maxlen=associativity) for i in range(self.num_sets)}
        self.bus = bus
        self.dram = dram
        self.protocol = protocol 
        self.private_accesses = 0
        self.shared_accesses = 0
        self.stats = CacheStatistics()
        self.pending_bus_request = None  
        self.miss_resolved = False       
        self.pending_miss_info = None    
        self.bus_request_made = False
        self.waiting_for_bus_upgrade_completion = False
        self.last_bus_upgrade_address = None 
        self.address_handler = AddressHandler(block_size, self.num_sets)
        self.memory_fetch_cycles_remaining = 0 

    def notify_write_back_complete(self):
        pass
        # self.stats.write_back_cycles += self.dram.access_latency

    def handle_bus_transaction(self, snoop_hit: bool):
        if self.protocol == 'MESI':
            if self.pending_bus_request:
                transaction_type = self.pending_bus_request['type']
                if transaction_type in [BusTransaction.BUS_READ, BusTransaction.BUS_READ_X]:
                    self.resolve_miss(snoop_hit)
                elif transaction_type == BusTransaction.BUS_UPGRADE:
                    self.pending_bus_request = None
                    self.bus_request_made = False

        elif self.protocol == 'Dragon':
            if self.pending_bus_request and self.pending_bus_request['type'] == BusTransaction.BUS_READ:
                self.resolve_miss(snoop_hit)
            elif self.pending_bus_request and self.pending_bus_request['type'] == BusTransaction.BUS_UPDATE:
                self.pending_bus_request = None
                self.bus_request_made = False

    def change_block_state(self, block, new_state):
        old_state = block.state
        logging.debug(f"CPU {self.cpu_id}: State transition: {old_state} -> {new_state}")
        block.state = new_state

    def access(self, address, operation):
        set_index, tag = self.get_set_index_and_tag(address)
        block = self.find_block(set_index, tag)

        if block and block.state != CacheState.INVALID:
            self.process_cache_hit(block, operation)
            return True 
        else:
            self.process_cache_miss(set_index, tag, operation, address)
            return False 

    def step(self):
        if self.pending_bus_request:
            if not self.bus_request_made:
                self.bus.request(self.cpu_id, self.pending_bus_request)
                self.bus_request_made = True
            return 

        if self.waiting_for_bus_upgrade_completion:
            if not self.bus.is_busy():
                set_index, tag = self.get_set_index_and_tag(self.last_bus_upgrade_address)
                block = self.find_block(set_index, tag)
                if block:
                    self.change_block_state(block, CacheState.MESI_MODIFIED)
                self.waiting_for_bus_upgrade_completion = False
                self.last_bus_upgrade_address = None
            return 

        if self.memory_fetch_cycles_remaining > 0:
            self.memory_fetch_cycles_remaining -= 1
            if self.memory_fetch_cycles_remaining == 0:
                self.complete_miss()
            return 

        if self.miss_resolved:
            self.miss_resolved = False
            pass 


    def is_busy(self):
        return (self.pending_bus_request is not None or
                self.waiting_for_bus_upgrade_completion or
                self.memory_fetch_cycles_remaining > 0)

    def mesi_hit(self, block, operation: Operation):
        if block.state == CacheState.MESI_MODIFIED:
            self.private_accesses += 1
        elif block.state == CacheState.MESI_EXCLUSIVE:
            if operation == Operation.STORE:
                self.change_block_state(block, CacheState.MESI_MODIFIED)
            self.private_accesses += 1
        elif block.state == CacheState.MESI_SHARED:
            if operation == Operation.STORE:
                self.pending_bus_request = {'type': BusTransaction.BUS_UPGRADE, 'address': block.address}
                self.bus_request_made = False
                self.waiting_for_bus_upgrade_completion = True
                self.last_bus_upgrade_address = block.address
            self.shared_accesses += 1

    def dragon_hit(self, block, operation: Operation):
        if block.state == CacheState.INVALID:
            logging.error(f"CPU {self.cpu_id}: Invalid state on hit")
        elif block.state == CacheState.DRAGON_MODIFIED:
            self.private_accesses += 1
        elif block.state == CacheState.DRAGON_EXCLUSIVE:
            if operation == Operation.STORE:
                self.change_block_state(block, CacheState.DRAGON_MODIFIED)
            self.private_accesses += 1
        elif block.state == CacheState.DRAGON_SHARED_CLEAN:
            if operation == Operation.STORE:
                self.change_block_state(block, CacheState.DRAGON_SHARED_MODIFIED)
                self.bus.request(self.cpu_id, {'type': BusTransaction.BUS_UPDATE, 'address': block.address})
            self.shared_accesses += 1
        elif block.state == CacheState.DRAGON_SHARED_MODIFIED:
            if operation == Operation.STORE:
                self.bus.request(self.cpu_id, {'type': BusTransaction.BUS_UPDATE, 'address': block.address})
            self.shared_accesses += 1
        else:
            logging.error(f"CPU {self.cpu_id}: Unknown state {block.state} on hit")

    def process_cache_hit(self, block, operation: Operation):
        self.stats.record_access(True, block.address, block.state)
        self.update_lru(block)
        if self.protocol == 'MESI':
            self.mesi_hit(block, operation)
        elif self.protocol == 'Dragon':
            self.dragon_hit(block, operation)

    def process_cache_miss(self, set_index, tag, operation, address):
        block_address = self.address_handler.get_block_address(address)
        self.stats.record_access(False, block_address, None)
        self.evict_if_needed(set_index)
        if self.protocol == 'MESI':
            if operation == Operation.STORE: 
                bus_transaction_type = BusTransaction.BUS_READ_X
            else:
                bus_transaction_type = BusTransaction.BUS_READ
        elif self.protocol == 'Dragon':
            bus_transaction_type = BusTransaction.BUS_READ
        self.pending_bus_request = {'type': bus_transaction_type, 'address': address}
        self.pending_miss_info = {
            'set_index': set_index,
            'tag': tag,
            'operation': operation,
            'address': address
        }
        self.miss_resolved = False 

    def resolve_miss(self, snoop_hit: bool):
        if not snoop_hit:
            self.memory_fetch_cycles_remaining = self.dram.access_latency 
        else:
            words_in_block = self.block_size // self.word_size
            self.memory_fetch_cycles_remaining = 2 * words_in_block 
        self.pending_bus_request = None
        self.bus_request_made = False

    def complete_miss(self):
        set_index = self.pending_miss_info['set_index']
        tag = self.pending_miss_info['tag']
        operation = self.pending_miss_info['operation']
        address = self.pending_miss_info['address']
        block = CacheBlock(tag, set_index)
        block.address = self.address_handler.get_block_address(address)

        if self.protocol == 'MESI':
            if operation == Operation.STORE:
                block.state = CacheState.MESI_MODIFIED
                if self.shared_with_other_caches(block.address):
                    self.shared_accesses += 1
                else:
                    self.private_accesses += 1
            else:
                if self.shared_with_other_caches(block.address):
                    block.state = CacheState.MESI_SHARED
                    self.shared_accesses += 1
                else:
                    block.state = CacheState.MESI_EXCLUSIVE
                    self.private_accesses += 1

        elif self.protocol == 'Dragon':
            if operation == Operation.STORE:
                if self.shared_with_other_caches(block.address):
                    block.state = CacheState.DRAGON_SHARED_MODIFIED
                    self.shared_accesses += 1
                else:
                    block.state = CacheState.DRAGON_MODIFIED
                    self.private_accesses += 1
            else:
                if self.shared_with_other_caches(block.address):
                    block.state = CacheState.DRAGON_SHARED_CLEAN
                    self.shared_accesses += 1
                else:
                    block.state = CacheState.DRAGON_EXCLUSIVE
                    self.private_accesses += 1

        self.cache[set_index].append(block)
        self.update_lru(block)
        self.stats.record_access(False, block.address, block.state)
        self.miss_resolved = True
        if self.memory_fetch_cycles_remaining == self.dram.access_latency:
            self.bus.data_traffic += self.block_size
        self.pending_miss_info = None

    def shared_with_other_caches(self, address):
        for cache in self.bus.caches:
            if cache.cpu_id != self.cpu_id:
                set_index, tag = cache.get_set_index_and_tag(address)
                block = cache.find_block(set_index, tag)
                if block and block.state != CacheState.INVALID:
                    return True
        return False

    def snoop(self, transaction_type: BusTransaction, address: int) -> bool:
        set_index, tag = self.get_set_index_and_tag(address)
        block = self.find_block(set_index, tag)
        hit = False

        if block and block.state != CacheState.INVALID:
            hit = True
            if self.protocol == 'MESI':
                if transaction_type == BusTransaction.BUS_READ:
                    if block.state in [CacheState.MESI_MODIFIED, CacheState.MESI_EXCLUSIVE]:
                        self.bus.request(self.cpu_id, {'type': BusTransaction.FLUSH_OPT, 'address': block.address})
                        self.change_block_state(block, CacheState.MESI_SHARED)

                elif transaction_type == BusTransaction.BUS_READ_X:
                    if block.state in [CacheState.MESI_MODIFIED, CacheState.MESI_EXCLUSIVE]:
                        self.bus.request(self.cpu_id, {'type': BusTransaction.FLUSH_OPT, 'address': block.address})
                    self.change_block_state(block, CacheState.INVALID)

                elif transaction_type == BusTransaction.BUS_WRITEBACK:
                    self.change_block_state(block, CacheState.INVALID)
            
            elif self.protocol == 'Dragon':
                if block and block.state != CacheState.INVALID:
                    if transaction_type == BusTransaction.BUS_READ:
                        if block.state == CacheState.DRAGON_MODIFIED:
                            self.change_block_state(block, CacheState.DRAGON_SHARED_MODIFIED)
                        elif block.state == CacheState.DRAGON_EXCLUSIVE:
                            self.change_block_state(block, CacheState.DRAGON_SHARED_CLEAN)
                    elif transaction_type == BusTransaction.BUS_UPDATE:
                        if block.state == CacheState.DRAGON_SHARED_MODIFIED:
                            self.change_block_state(block, CacheState.DRAGON_SHARED_CLEAN)
        return hit

    def evict_if_needed(self, set_index):
        if len(self.cache[set_index]) == self.associativity:
            evicted_block = self.cache[set_index].popleft()
            if evicted_block.state in [CacheState.MESI_MODIFIED, CacheState.DRAGON_MODIFIED, CacheState.DRAGON_SHARED_MODIFIED]:
                self.write_back_to_memory(evicted_block.address)
            return evicted_block
        return None

    def write_back_to_memory(self, address):
        self.bus.request(self.cpu_id, {'type': BusTransaction.BUS_WRITEBACK, 'address': address})
        logging.debug(f"CPU {self.cpu_id}: Initiated write-back for address {hex(address)}")

    def get_set_index_and_tag(self, address):
        tag, set_index, _ = self.address_handler.parse_address(address)
        return set_index, tag

    def find_block(self, set_index, tag):
        for block in self.cache[set_index]:
            if block.tag == tag:
                return block
        return None

    def update_lru(self, block):
        set_index = block.set_index
        try:
            self.cache[set_index].remove(block)
        except ValueError:
            logging.warning(f"CPU {self.cpu_id}: Block {block.tag} not found in set {set_index} for LRU update.")
        self.cache[set_index].append(block)

class CPU:
    def __init__(self, cpu_id, cache, trace_file):
        self.cpu_id = cpu_id
        self.cache = cache
        self.compute_cycles = 0
        self.load_store_instructions = 0
        self.idle_cycles = 0
        self.trace_file = trace_file
        self.trace = self.load_trace()
        self.instruction_pointer = 0
        self.compute_cycles_remaining = 0
        self.waiting_for_cache = False
        self.total_cycles = 0

    def load_trace(self):
        instructions = []
        try:
            with open(self.trace_file, 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    if len(parts) != 2:
                        logging.warning(f"Invalid trace line: {line.strip()}")
                        continue
                    instruction, value = parts
                    instructions.append((int(instruction), value))
            return instructions
        except FileNotFoundError:
            logging.error(f"Trace file {self.trace_file} not found.")
            sys.exit(1)

    def step(self):
        if self.instruction_pointer < len(self.trace):
            self.total_cycles += 1
        if self.waiting_for_cache:
            if not self.cache.is_busy():
                self.instruction_pointer += 1
                self.waiting_for_cache = False
            else:
                self.idle_cycles += 1
            return

        if self.compute_cycles_remaining > 0:
            self.compute_cycles_remaining -= 1
            self.compute_cycles += 1
            return

        if self.instruction_pointer >= len(self.trace):
            return

        instruction, value = self.trace[self.instruction_pointer]

        if instruction in [0, 1]: 
            address = int(value, 16)
            operation = Operation.LOAD if instruction == 0 else Operation.STORE 
            self.load_store_instructions += 1
            hit = self.cache.access(address, operation)
            if not hit or self.cache.is_busy():
                self.waiting_for_cache = True
                self.idle_cycles += 1
            else:
                self.instruction_pointer += 1

        elif instruction == 2: 
            cycles = int(value, 16)
            self.compute_cycles_remaining = cycles - 1  
            self.compute_cycles += 1
            self.instruction_pointer += 1

class Simulator:
    def __init__(self, protocol, input_file, cache_size, associativity, block_size):
        self.protocol = protocol
        self.input_file = input_file
        self.cache_size = cache_size
        self.associativity = associativity
        self.block_size = block_size
        self.num_cores = 4
        self.dram = DRAM()
        self.bus = Bus(self.block_size, self.dram)
        self.cpus = []
        self.initialize_cores()
        self.global_cycle = 0

    def initialize_cores(self):
        self.bus.caches = []
        for i in range(self.num_cores):
            trace_file = f"{self.input_file}_{i}.data.txt"
            cache = Cache(i, self.cache_size, self.block_size, self.associativity, self.bus, self.dram, self.protocol)
            self.bus.caches.append(cache)
            cpu = CPU(i, cache, trace_file)
            self.cpus.append(cpu)

    def run(self):
        logging.info("Starting simulation...")
        while not self.all_programs_finished():
            self.global_cycle += 1
            self.bus.step() 
            for cache in self.bus.caches:
                cache.step()
            for cpu in self.cpus:
                cpu.step()
        self.generate_report()

    def all_programs_finished(self):
        return all(cpu.instruction_pointer >= len(cpu.trace) and
                   cpu.compute_cycles_remaining == 0 and
                   not cpu.waiting_for_cache for cpu in self.cpus)

    def generate_report(self):
        logging.info("Generating report...")
        print("===== Simulation Report =====")
        max_cycle = self.global_cycle
        print(f"Overall Execution Cycles: {max_cycle}")
        print(f"Total Data Traffic on Bus: {self.bus.data_traffic} bytes")
        
        if self.protocol == "MESI":
            print(f"Number of Invalidations: {self.bus.stats['invalidations']}")
        elif self.protocol == "Dragon":
            print(f"Number of Updates: {self.bus.stats['updates']}")
        
        for cpu in self.cpus:
            cache_stats = cpu.cache.stats.get_stats()
            print(f"--- CPU {cpu.cpu_id} ---")
            print(f"Total Cycles: {cpu.total_cycles}")
            print(f"Compute Cycles: {cpu.compute_cycles}")
            print(f"Load/Store Instructions: {cpu.load_store_instructions}")
            print(f"Idle Cycles: {cpu.idle_cycles}")
            print(f"Cache Hits: {cache_stats['hits']}")
            print(f"Cache Misses: {cache_stats['misses']}")
            print(f"Hit Rate: {cache_stats['hit_rate']:.2%}")
            print(f"Miss Rate: {(1 - cache_stats['hit_rate']):.2%}")
            print(f"Private Accesses: {cpu.cache.private_accesses}")
            print(f"Shared Accesses: {cpu.cache.shared_accesses}")
        print("=============================")

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Cache Coherence Simulator')
    parser.add_argument('protocol', type=str, choices=['MESI', 'Dragon'], help='Coherence protocol (e.g., MESI, Dragon)')
    parser.add_argument('input_file', type=str, help='Input benchmark name (e.g., bodytrack)')
    parser.add_argument('cache_size', type=int, nargs='?', default=4096, help='Cache size in bytes (default: 4096)')
    parser.add_argument('associativity', type=int, nargs='?', default=2, help='Cache associativity (default: 2)')
    parser.add_argument('block_size', type=int, nargs='?', default=32, help='Cache block size in bytes (default: 32)')
    args = parser.parse_args()

    if args.cache_size <= 0 or args.associativity <= 0 or args.block_size <= 0:
        logging.error("Cache size, associativity, and block size must be positive integers.")
        sys.exit(1)
    if args.cache_size % (args.block_size * args.associativity) != 0:
        logging.error("Cache size must be a multiple of (block_size * associativity).")
        sys.exit(1)

    trace_folder = '/Users/oysteinweibell/Desktop/MulticoreArch/assignment2/cs4223A2P1/traces'

    simulator = Simulator(args.protocol, f"{trace_folder}/{args.input_file}", args.cache_size, args.associativity, args.block_size)
    simulator.run()

if __name__ == "__main__":
    main()