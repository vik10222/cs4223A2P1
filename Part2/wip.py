import sys
import logging
import math
from collections import deque, defaultdict
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Set

class CacheState(Enum):
    """Cache states for MESI and Dragon protocols"""
    # MESI states
    MESI_MODIFIED = auto()
    MESI_EXCLUSIVE = auto()
    MESI_SHARED = auto()
    INVALID = auto()
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
    # Generic bus transactions to/from main memory
    BUS_WRITEBACK = auto()
    BUS_WRITEBACK_ACK = auto()
    BUS_READ_ADDRESS = auto()
    BUS_READ_DATA = auto()
    # MESI-specific bus transactions between caches
    BUS_READ_X = auto()
    BUS_UPGRADE = auto()
    FLUSH_OPT = auto()
    # Dragon-specific bus transactions between caches
    BUS_UPDATE = auto()

class DRAM:
    def __init__(self, total_access_latency, bus):
        self.bus = bus
        self.total_access_latency = total_access_latency
        self.request_queue = []  # Array to store requests, each with a cycle counter
        self.total_transfer_latency = 8 + 1 # 8 cycles for data and 1 cycle for address or writeback_ack

    def enqueue_request(self, transaction):
        transaction['cycles_remaining'] = self.total_access_latency - self.total_transfer_latency
        self.request_queue.append(transaction)

    def request_bus(self, transaction):
        self.bus.request(transaction)

    def decrement_cycle_counters(self):
        for request in self.request_queue:
            request['cycles_remaining'] -= 1

    def step(self):
        self.decrement_cycle_counters()
        if not any(request['cycles_remaining'] == 0 for request in self.request_queue):
            return
        for request in self.request_queue:
            if request['cycles_remaining'] == 0:
                completed_request['from_dram'] = True
                self.request_bus(completed_request)
                self.request_queue = [req for req in self.request_queue if req['cycles_remaining'] > 0]  # Remove completed request

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
        self.total_cycles = 0

    def record_access(self, hit: bool):
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
            'hit_rate': 1 - self.get_miss_rate()
        }

class Bus:
    def __init__(self, block_size: int):
        self.dram = None # Will be set later by Simulator class
        self.caches = None # Will be set later by Simulator class
        self.queue: deque = deque()
        self.dram_response_queue: deque = deque()

        self.word_size = 4
        self.block_size = block_size
        
        self.is_busy = False
        self.current_transaction = None
        self.transfer_cycles_remaining = 0
        self.data_traffic = 0
        self.stats = {
            'invalidations': 0,
            'updates': 0
            }

    def grant(self, transaction):
        if self.is_busy:
            raise ValueError("Bus is already busy when trying to grant")
        self.is_busy = True
        self.current_transaction = transaction

    def release(self):
        if not self.is_busy:
            raise ValueError("Bus is not busy when released")
        self.current_transaction = None
        self.is_busy = False

    def request(self, transaction):
        if transaction['from_dram']:
            self.dram_response_queue.append(transaction)
        else:
            self.queue.append(transaction)

    def step(self):

        if self.transfer_cycles_remaining > 0:
            self.transfer_cycles_remaining -= 1
            if self.transfer_cycles_remaining == 0:
                self.complete_transaction()
            return

        # Give priority to DRAM responses
        if self.dram_response_queue and not self.is_busy:
            self.grant(self.dram_response_queue.popleft())
            return
        
        if self.queue and not self.is_busy:
            self.grant(self.queue.popleft())
            return

    def calculate_transfer_cycles(self, transaction):
        transaction_type = transaction['type']
        if transaction_type == BusTransaction.BUS_WRITEBACK:
            words_in_block = self.block_size // self.word_size
            return 2 * words_in_block
        elif transaction_type == BusTransaction.BUS_WRITEBACK_ACK:
            words_in_block = self.block_size // self.word_size
            return 1
        elif transaction_type == BusTransaction.BUS_READ_ADDRESS:
            return 1
        elif transaction_type == BusTransaction.BUS_READ_DATA:
            words_in_block = self.block_size // self.word_size
            return 2 * words_in_block
        elif transaction_type == BusTransaction.BUS_READ_X:
            return 1
        elif transaction_type == BusTransaction.BUS_UPGRADE:
            return 1 
        elif transaction_type == BusTransaction.FLUSH_OPT:
            words_in_block = self.block_size // self.word_size
            return 2 * words_in_block
        elif transaction_type == BusTransaction.BUS_UPDATE:
            words_in_block = self.block_size // self.word_size
            return 2 * words_in_block
        raise ValueError(f"Unknown transaction type: {transaction_type}")

    def update_traffic_stats(self, transaction: Dict):
        transaction_type = transaction['type']
        if transaction_type == BusTransaction.BUS_WRITEBACK:
            self.data_traffic += self.block_size
        elif transaction_type == BusTransaction.BUS_WRITEBACK_ACK:
            pass
        elif transaction_type == BusTransaction.BUS_READ_ADDRESS:
            pass
        elif transaction_type == BusTransaction.BUS_READ_DATA:
            self.data_traffic += self.block_size
        elif transaction_type == BusTransaction.BUS_READ_X:
            self.stats['invalidations'] += 1
        elif transaction_type == BusTransaction.BUS_UPGRADE:
            self.stats['invalidations'] += 1
        elif transaction_type == BusTransaction.FLUSH_OPT:
            self.data_traffic += self.block_size
        elif transaction_type == BusTransaction.BUS_UPDATE:
            self.data_traffic += self.block_size
            self.stats['updates'] += 1
        else:
            raise ValueError(f"Unknown transaction type: {transaction_type}")

    def complete_transaction(self):
        transaction = self.current_transaction
        if transaction['type'] == BusTransaction.BUS_WRITEBACK:
            self.dram.enqueue_request(Operation.STORE, transaction)
            cache = next((c for c in self.caches if c.cpu_id == cpu_id), None)
            if cache:
                cache.notify_write_back_complete()
        self.release()

class CacheBlock:
    def __init__(self, tag, set_index):
        self.tag = tag
        self.set_index = set_index
        self.state = CacheState.INVALID
        self.address = None

    def __eq__(self, other):
        return isinstance(other, CacheBlock) and self.tag == other.tag and self.set_index == other.set_index

    def __hash__(self):
        return hash((self.tag, self.set_index))

class Cache:
    def __init__(self, cpu_id, size, block_size, associativity, bus, protocol):
        self.cpu_id = cpu_id
        self.bus = bus
        self.word_size = 4
        self.size = size
        self.block_size = block_size
        self.associativity = associativity
        self.num_sets = size // (block_size * associativity)
        self.cache = {i: deque(maxlen=associativity) for i in range(self.num_sets)}
        self.protocol = protocol 
        self.private_accesses = 0
        self.shared_accesses = 0
        self.stats = CacheStatistics()
        self.address_handler = AddressHandler(block_size, self.num_sets)
        self.pending_writeback_ack = False
        self.pending_miss_info = None
        self.is_blocking = False

    def handle_bus_transaction(self, snoop_hit: bool):
        if not self.pending_miss_info:
            raise Exception("No pending miss info, when expected")
        set_index = self.pending_miss_info['set_index']
        tag = self.pending_miss_info['tag']
        operation = self.pending_miss_info['operation']
        address = self.pending_miss_info['address']
        block = CacheBlock(tag, set_index)
        block.address = self.address_handler.get_block_address(address)
        self.cache[set_index].append(block)
        self.update_lru(block)

        if self.protocol == 'MESI':
            transaction_type = self.pending_miss_info['operation']
            if transaction_type == Operation.STORE:
                block.state = CacheState.MESI_MODIFIED
            elif transaction_type == Operation.LOAD:
                if snoop_hit:
                    block.state = CacheState.MESI_SHARED
                else:
                    block.state = CacheState.MESI_EXCLUSIVE

        elif self.protocol == 'Dragon':
            if transaction_type == Operation.STORE:
                block.state = CacheState.DRAGON_MODIFIED
            elif transaction_type == Operation.LOAD:
                if snoop_hit:
                    block.state = CacheState.DRAGON_SHARED_MODIFIED
                else:
                    block.state = CacheState.DRAGON_EXCLUSIVE
        self.pending_miss_info = None
        self.is_blocking = False

    # Successfull writeback due to eviction on cache miss
    # Now we need to perform the memory operation that caused the cache miss
    def handle_dram_ack(self):
        if not self.pending_writeback_ack or not self.pending_miss_info:
            raise ValueError("No pending writeback ack expected")
        self.pending_writeback_ack = False
        pending_operation = self.pending_miss_info['operation']
        if self.protocol == 'MESI':
            if pending_operation == Operation.STORE:
                transaction_type = BusTransaction.BUS_READ_X
            elif pending_operation == Operation.LOAD:
                transaction_type = BusTransaction.BUS_READ_ADDRESS
        elif self.protocol == 'Dragon':
            if pending_operation == Operation.STORE:
                transaction_type = BusTransaction.BUS_UPDATE
            elif pending_operation == Operation.LOAD:
                transaction_type = BusTransaction.BUS_READ_ADDRESS
        self.request_bus({'type': transaction_type, 'address': self.pending_miss_info['address']})

    def change_block_state(self, block, new_state):
        block.state = new_state

    def access(self, address, operation):
        set_index, tag = self.get_set_index_and_tag(address)
        block = self.find_block(set_index, tag)
        if block and block.state != CacheState.INVALID:
            self.process_cache_hit(block, operation, address)
        else:
            self.shared_accesses += 1
            self.process_cache_miss(set_index, tag, operation, address)

    def process_cache_hit(self, block, operation, address):
        self.stats.record_access(True)
        self.update_lru(block)
        if self.protocol == 'MESI':
            self.mesi_hit(block, operation, address)
        elif self.protocol == 'Dragon':
            self.dragon_hit(block, operation, address)

    def mesi_hit(self, block, operation, address):
        if block.state == CacheState.MESI_MODIFIED:
            self.private_accesses += 1
        elif block.state == CacheState.MESI_EXCLUSIVE:
            if operation == Operation.STORE:
                self.change_block_state(block, CacheState.MESI_MODIFIED)
            self.private_accesses += 1
        elif block.state == CacheState.MESI_SHARED:
            if operation == Operation.STORE:
                self.request_bus({'type': BusTransaction.BUS_UPGRADE, 'address': address})
            self.shared_accesses += 1
        else:
            raise ValueError(f"Invalid block state: {block.state}")

    def dragon_hit(self, block, operation, address):
        if block.state == CacheState.DRAGON_MODIFIED:
            self.private_accesses += 1
        elif block.state == CacheState.DRAGON_EXCLUSIVE:
            if operation == Operation.STORE:
                self.change_block_state(block, CacheState.DRAGON_MODIFIED)
            self.private_accesses += 1
        elif block.state == CacheState.DRAGON_SHARED_CLEAN:
            if operation == Operation.STORE:
                self.change_block_state(block, CacheState.DRAGON_SHARED_MODIFIED)
                self.request_bus({'type': BusTransaction.BUS_UPDATE, 'address': address})
            self.shared_accesses += 1
        elif block.state == CacheState.DRAGON_SHARED_MODIFIED:
            if operation == Operation.STORE:
                self.request_bus({'type': BusTransaction.BUS_UPDATE, 'address': address})
            self.shared_accesses += 1
        else:
            raise ValueError(f"Invalid block state: {block.state}")

    def process_cache_miss(self, set_index, tag, operation, address):
        self.stats.record_access(False)
        if self.evict_if_needed(set_index, address):
            self.pending_writeback_ack = True
            self.pending_miss_info = {
                'set_index': set_index,
                'tag': tag,
                'operation': operation,
                'address': address
            }
            return

        # No eviction
        if self.protocol == 'MESI':
            if operation == Operation.STORE:
                transaction_type = BusTransaction.BUS_READ_X
            elif operation == Operation.LOAD:
                transaction_type = BusTransaction.BUS_READ_ADDRESS
        elif self.protocol == 'Dragon':
            transaction_type = BusTransaction.BUS_READ_ADDRESS

        self.request_bus({'type': transaction_type, 'address': address})

    def evict_if_needed(self, set_index, address):
        if len(self.cache[set_index]) == self.associativity:
            evicted_block = self.cache[set_index].popleft()
            if evicted_block.state in [CacheState.MESI_MODIFIED, CacheState.DRAGON_MODIFIED, CacheState.DRAGON_SHARED_MODIFIED]:
                self.request_bus({'type': BusTransaction.BUS_WRITEBACK, 'address': address})
                return True
        return False

    def step(self):
        if self.bus.new_transaction:
            self.snoop(self.bus.newest_transaction)

    def shared_with_other_caches(self, address):
        for cache in self.bus.caches:
            if cache.cpu_id != self.cpu_id:
                set_index, tag = cache.get_set_index_and_tag(address)
                block = cache.find_block(set_index, tag)
                if block and block.state != CacheState.INVALID:
                    return True
        return False
    def snoop_self(self, transaction):
        set_index, tag = self.get_set_index_and_tag(transaction['address'])
        block = self.find_block(set_index, tag)
        transaction_type = transaction['type']

    def snoop_other(self, transaction):
        set_index, tag = self.get_set_index_and_tag(transaction['address'])
        address = transaction['address']
        block = self.find_block(set_index, tag)
        transaction_type = transaction['type']

        if self.protocol == 'MESI':
            if block and not block.state == CacheState.INVALID:
                if transaction_type == BusTransaction.BUS_READ_ADDRESS:
                    if block.state in [CacheState.MESI_MODIFIED, CacheState.MESI_EXCLUSIVE]:
                        self.request_bus_snooping({'type': BusTransaction.FLUSH_OPT, 'address': address})
                elif transaction_type == BusTransaction.BUS_READ_X:
                    if block.state in [CacheState.MESI_MODIFIED, CacheState.MESI_EXCLUSIVE]:
                        self.request_bus_snooping({'type': BusTransaction.FLUSH_OPT, 'address': address})
                    self.change_block_state(block, CacheState.INVALID)
                elif transaction_type == BusTransaction.BUS_UPGRADE:
                    self.change_block_state(block, CacheState.INVALID)
                elif transaction_type == BusTransaction.BUS_WRITEBACK:
                    pass
            if self.pending_miss_info['address'] == transaction['address']:
                if transaction_type == BusTransaction.BUS_WRITEBACK_ACK:
                    self.handle_dram_ack()
        
        elif self.protocol == 'Dragon':
            if transaction_type == BusTransaction.BUS_READ_ADDRESS:
                if block.state == CacheState.DRAGON_MODIFIED:
                    self.change_block_state(block, CacheState.DRAGON_SHARED_MODIFIED)
                elif block.state == CacheState.DRAGON_EXCLUSIVE:
                    self.change_block_state(block, CacheState.DRAGON_SHARED_CLEAN)
            elif transaction_type == BusTransaction.BUS_UPDATE:
                if block.state == CacheState.DRAGON_SHARED_MODIFIED:
                    self.change_block_state(block, CacheState.DRAGON_SHARED_CLEAN)

    def snoop(self, transaction):
        if transaction['cpu_id'] == self.cpu_id and not transaction['from_dram']:
            self.snoop_self(transaction)
        else:
            self.snoop_other(transaction)

    def request_bus_snooping(self, transaction):
        transaction['cpu_id'] = self.cpu_id
        transaction['from_dram'] = False
        self.bus.request(transaction)

    def request_bus(self, transaction):
        if self.is_blocking: 
            raise RuntimeError("CPU is already waiting for the bus, cannot make another request originating from the CPU")
        self.is_blocking = True # Should only block when servicing bus requests originating from the CPU
        transaction['cpu_id'] = self.cpu_id
        transaction['from_dram'] = False
        self.blocking_transaction = transaction
        self.bus.request(transaction)

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
        if self.cache[set_index][-1] != block: # Update only if the block is not already the most recently used
            self.cache[set_index].remove(block)
            self.cache[set_index].append(block)

class CPU:
    def __init__(self, cpu_id, cache, trace_file):
        self.cpu_id = cpu_id
        self.cache = cache
        self.trace = self.load_trace(trace_file)

        self.total_cycles = 0
        self.compute_cycles = 0
        self.idle_cycles = 0
        self.load_store_instructions = 0
        
        self.instruction_pointer = 0
        self.compute_cycles_remaining = 0

    def load_trace(self, trace_file):
        instructions = []
        try:
            with open(trace_file, 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    if len(parts) != 2:
                        logging.warning(f"Invalid trace line: {line.strip()}")
                        continue
                    instruction, value = parts
                    instructions.append((int(instruction), value))
            return instructions[:100000]
        except FileNotFoundError:
            logging.error(f"Trace file {self.trace_file} not found.")
            sys.exit(1)

    def step(self):
        if self.instruction_pointer < len(self.trace):
            self.total_cycles += 1
        else:
            return

        if self.compute_cycles_remaining > 0:
            self.compute_cycles_remaining -= 1
            self.compute_cycles += 1
            return

        if self.cache.is_blocking:
            self.idle_cycles += 1
            return

        # We only arrive here if there are no compute cycles remaining and no outstanding memory requests
        instruction, value = self.trace[self.instruction_pointer]
        if instruction in [0, 1]: 
            address = int(value, 16)
            operation = Operation.LOAD if instruction == 0 else Operation.STORE 
            self.load_store_instructions += 1
            self.cache.access(address, operation)
            self.idle_cycles += 1 # Always 1 (cache hit) or more (cache miss) idle cycles during load/store
        elif instruction == 2: 
            cycles = int(value, 16)
            self.compute_cycles_remaining = cycles - 1 # Subtract 1 for the current cycle
            self.compute_cycles += 1

        self.instruction_pointer += 1 # After instruction fetch we increment the instruction pointer

class Simulator:
    def __init__(self, protocol, input_file, cache_size, associativity, block_size):
        self.protocol = protocol
        self.input_file = input_file
        self.cache_size = cache_size
        self.associativity = associativity
        self.block_size = block_size
        self.num_cores = 4
        self.global_cycle = 0

        # Initialize core components
        self.bus = Bus(self.block_size)
        self.dram = DRAM(100, self.bus)
        self.caches = []
        self.cpus = []

        for i in range(self.num_cores):
            trace_file = f"{self.input_file}_{i}.data.txt"
            cache = Cache(i, self.cache_size, self.block_size, self.associativity, self.bus, self.protocol)
            cpu = CPU(i, cache, trace_file)
            self.caches.append(cache)
            self.cpus.append(cpu)
        
        # Initialize cross-references
        self.initialize_connections()

    def initialize_connections(self):
        self.bus.dram = self.dram
        self.bus.caches = self.caches

    def run(self):
        logging.info("Starting simulation...")
        while not self.all_programs_finished():
            self.global_cycle += 1
            for cpu in self.cpus:
                cpu.step()
            for cache in self.bus.caches:
                cache.step()
            self.bus.step() 
            self.dram.step()
        self.generate_report()

    def all_programs_finished(self):
        return all(cpu.instruction_pointer >= len(cpu.trace) and
                   cpu.compute_cycles_remaining == 0 and
                   not cpu.cache.is_blocking for cpu in self.cpus)

    def generate_report(self):
        logging.info("Generating report...")
        print("===== Simulation Report =====")
        print(f"Overall Execution Cycles: {self.global_cycle}")
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