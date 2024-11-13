import sys
import logging
import math
import copy
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
    # Bus transaction between caches
    BUS_READ = auto()
    # MESI-specific bus transactions between caches
    BUS_READ_X = auto()
    BUS_UPGRADE = auto()
    FLUSH_OPT = auto()
    # Dragon-specific bus transactions between caches
    BUS_UPDATE = auto()
    FLUSH = auto()

class DRAM:
    def __init__(self, total_access_latency, bus):
        self.bus = bus
        self.total_access_latency = total_access_latency
        self.request_queue = []  # Array to store requests, each with a cycle counter
        self.total_transfer_latency = 8 + 1 # 8 cycles for data and 1 cycle for address or writeback_ack

    def request_bus(self, transaction):
        if transaction['type'] == BusTransaction.BUS_READ_ADDRESS:
            transaction['type'] = BusTransaction.BUS_READ_DATA
        elif transaction['type'] == BusTransaction.BUS_WRITEBACK:
            transaction['type'] = BusTransaction.BUS_WRITEBACK_ACK
        self.bus.request(transaction)

    def decrement_cycle_counters(self):
        for request in self.request_queue:
            request['cycles_remaining'] -= 1

    def snoop(self, transaction):
        if transaction['type'] in [BusTransaction.BUS_READ_ADDRESS, BusTransaction.BUS_WRITEBACK]:
            request = copy.deepcopy(transaction)
            request['cycles_remaining'] = self.total_access_latency - self.total_transfer_latency
            self.request_queue.append(request)

    def step(self):
        if self.bus.new_transaction:
            self.snoop(self.bus.current_transaction)
        self.decrement_cycle_counters()
        if not any(request['cycles_remaining'] == 0 for request in self.request_queue):
            return
        for request in self.request_queue:
            if request['cycles_remaining'] == 0:
                request['from_dram'] = True
                self.request_bus(request)
                self.request_queue = [req for req in self.request_queue if req['cycles_remaining'] > 0]  # Remove completed request

class AddressHandler:
    def __init__(self, block_size, num_sets):
        if (block_size & (block_size - 1)) != 0:
            raise ValueError("Block size must be a power of 2")
        if (num_sets & (num_sets - 1)) != 0:
            raise ValueError("Number of sets must be a power of 2")
        
        self.num_sets = num_sets
        self.block_offset_bits = int(math.log2(block_size))
        self.index_bits = int(math.log2(num_sets))

    def get_set_index_and_tag(self, block_address):
        index = (block_address >> self.block_offset_bits) & (self.num_sets - 1)
        tag = block_address >> (self.index_bits + self.block_offset_bits)
        return index, tag

    def get_block_address(self, address):
        return address >> self.block_offset_bits

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
        
        self.new_transaction = False
        self.current_transaction = None
        self.transfer_cycles_remaining = 0
        self.data_traffic = 0
        self.stats = {
            'invalidations': 0,
            'updates': 0
            }

    def grant(self, transaction):
        self.new_transaction = True # Only True for first cycle
        self.current_transaction = transaction
        self.update_traffic_stats(transaction)
        self.transfer_cycles_remaining = self.calculate_transfer_cycles(transaction) - 1 # -1 to account for current cycle
        for cache in self.caches:
            cache.update_shared_line(transaction)

    def release(self):
        if self.transfer_cycles_remaining != 0:
            raise ValueError("Bus is still busy when trying to release")
        self.current_transaction = None

    def request(self, transaction):
        transaction_copy = copy.deepcopy(transaction)
        # We prioritize DRAM requests and responses
        if transaction_copy['from_dram'] or transaction_copy['type'] in [BusTransaction.BUS_READ_ADDRESS, BusTransaction.BUS_WRITEBACK]:
            self.dram_response_queue.append(transaction_copy)
        else:
            self.queue.append(transaction)

    def step(self):
        self.new_transaction = False

        if self.transfer_cycles_remaining > 0:
            self.transfer_cycles_remaining -= 1
            return

        if self.current_transaction:
            self.release()

        # Give priority to DRAM responses
        if self.dram_response_queue:
            self.grant(self.dram_response_queue.popleft())
            return
        
        if self.queue:
            self.grant(self.queue.popleft())
            return

    def calculate_transfer_cycles(self, transaction):
        words_in_block = self.block_size // self.word_size
        transaction_type = transaction['type']
        if transaction_type == BusTransaction.BUS_WRITEBACK:
            return 2 * words_in_block
        elif transaction_type == BusTransaction.BUS_WRITEBACK_ACK:
            return 1
        elif transaction_type == BusTransaction.BUS_READ_ADDRESS:
            return 1
        elif transaction_type == BusTransaction.BUS_READ_DATA:
            return 2 * words_in_block
        elif transaction_type == BusTransaction.BUS_READ:
            return 1
        elif transaction_type == BusTransaction.BUS_READ_X:
            return 1
        elif transaction_type == BusTransaction.BUS_UPGRADE:
            return 1 
        elif transaction_type == BusTransaction.FLUSH_OPT:
            return 2 * words_in_block
        elif transaction_type == BusTransaction.BUS_UPDATE:
            return 2 * words_in_block
        elif transaction_type == BusTransaction.FLUSH:
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
        elif transaction_type == BusTransaction.BUS_READ:
            pass
        elif transaction_type == BusTransaction.BUS_READ_X:
            self.stats['invalidations'] += 1
        elif transaction_type == BusTransaction.BUS_UPGRADE:
            self.stats['invalidations'] += 1
        elif transaction_type == BusTransaction.FLUSH_OPT:
            self.data_traffic += self.block_size
        elif transaction_type == BusTransaction.BUS_UPDATE:
            self.data_traffic += self.block_size
            self.stats['updates'] += 1
        elif transaction_type == BusTransaction.FLUSH:
            self.data_traffic += self.block_size
        else:
            raise ValueError(f"Unknown transaction type: {transaction_type}")

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

        self.shared_line = False
        self.pending_writeback_ack_address = None
        self.is_blocking = False
        self.is_blocking_info = None

    def update_shared_line(self, transaction):
        if any(cache.shared_line for cache in self.bus.caches if cache != self):
            return # Shared line already asserted by another cache
        if transaction['type'] not in [BusTransaction.BUS_READ, BusTransaction.BUS_READ_X, BusTransaction.BUS_UPDATE]:
            return # Transaction type does not concern the shared line
        block_address = transaction['block_address']
        set_index, tag = self.address_handler.get_set_index_and_tag(block_address)
        block = self.find_block(set_index, tag)
        if block and block.state != CacheState.INVALID:
            self.shared_line = True # This cache is now guaranteed to be the sole asserter of the shared line

    def shared_line_asserted(self):
        return any(cache.shared_line for cache in self.bus.caches if cache != self)

    def change_block_state(self, block, new_state):
        block.state = new_state

    def access(self, address, operation):
        block_address = self.address_handler.get_block_address(address)
        set_index, tag = self.address_handler.get_set_index_and_tag(address)
        block = self.find_block(set_index, tag)
        if block and block.state != CacheState.INVALID:
            self.process_cache_hit(block, operation, block_address)
        else:
            self.shared_accesses += 1
            self.process_cache_miss(set_index, tag, operation, block_address)

    def process_cache_hit(self, block, operation, block_address):
        self.stats.record_access(True)
        self.update_lru(block)
        if self.protocol == 'MESI':
            self.mesi_hit(block, operation, block_address)
        elif self.protocol == 'Dragon':
            self.dragon_hit(block, operation, block_address)

    def mesi_hit(self, block, operation, block_address):
        if block.state == CacheState.MESI_MODIFIED:
            self.private_accesses += 1
        elif block.state == CacheState.MESI_EXCLUSIVE:
            if operation == Operation.STORE:
                self.change_block_state(block, CacheState.MESI_MODIFIED)
            self.private_accesses += 1
        elif block.state == CacheState.MESI_SHARED:
            if operation == Operation.STORE:
                self.request_bus({'type': BusTransaction.BUS_UPGRADE, 'block_address': block_address})
            self.shared_accesses += 1
        else:
            raise ValueError(f"Invalid block state: {block.state}")

    def dragon_hit(self, block, operation, block_address):
        if block.state == CacheState.DRAGON_MODIFIED:
            self.private_accesses += 1
        elif block.state == CacheState.DRAGON_EXCLUSIVE:
            if operation == Operation.STORE:
                self.change_block_state(block, CacheState.DRAGON_MODIFIED)
            self.private_accesses += 1
        elif block.state == CacheState.DRAGON_SHARED_CLEAN:
            if operation == Operation.STORE:
                self.change_block_state(block, CacheState.DRAGON_SHARED_MODIFIED)
                self.request_bus({'type': BusTransaction.BUS_UPDATE, 'block_address': block_address})
            self.shared_accesses += 1
        elif block.state == CacheState.DRAGON_SHARED_MODIFIED:
            if operation == Operation.STORE:
                self.request_bus({'type': BusTransaction.BUS_UPDATE, 'block_address': block_address})
            self.shared_accesses += 1
        else:
            raise ValueError(f"Invalid block state: {block.state}")

    def evict_and_writeback_needed(self, set_index, block_address):
        if len(self.cache[set_index]) == self.associativity:
            evicted_block = copy.deepcopy(self.cache[set_index].popleft())
            # TODO: In DRAGON_SHARED_MODIFIED, another cache might want to adopt the evicted block by asserting the shared line
            # If so, we avoid the long way to main memory
            # This logic is somewhat tricky, will implement if I have the time, as this is merely an optimization
            if evicted_block.state in [CacheState.MESI_MODIFIED, CacheState.DRAGON_MODIFIED, CacheState.DRAGON_SHARED_MODIFIED]:
                return evicted_block
        return None

    def process_cache_miss(self, set_index, tag, operation, block_address):
        self.stats.record_access(False)
        self.is_blocking_info = {
            'set_index': set_index,
            'tag': tag,
            'operation': operation,
            'block_address': block_address
        }
        evicted_block = self.evict_and_writeback_needed(set_index, block_address)
        if evicted_block:
            self.request_bus({'type': BusTransaction.BUS_WRITEBACK, 'block_address': evicted_block.address})
            self.pending_writeback_ack_address = evicted_block.address
            return # Will request block after successful writeback

        # No eviction
        self.request_block_on_miss()

    # Successfull writeback due to eviction on cache miss
    # Now we need to perform the memory operation that caused the cache miss
    def handle_dram_ack(self):
        if not (self.pending_writeback_ack_address and self.is_blocking_info):
            raise ValueError("No pending writeback ack expected")
        self.pending_writeback_ack_address = None
        self.is_blocking = False

        self.request_block_on_miss()

    def request_block_on_miss(self):
        operation = self.is_blocking_info['operation']
        block_address = self.is_blocking_info['block_address']
        if self.protocol == 'MESI':
            if operation == Operation.STORE:
                transaction_type = BusTransaction.BUS_READ_X
            elif operation == Operation.LOAD:
                transaction_type = BusTransaction.BUS_READ
        elif self.protocol == 'Dragon':
            transaction_type = BusTransaction.BUS_READ
        self.request_bus({'type': transaction_type, 'block_address': block_address})

    def step(self):
        if self.bus.new_transaction:
            self.snoop(self.bus.current_transaction)

    def snoop_mesi_self(self, transaction):
        set_index, tag = self.address_handler.get_set_index_and_tag(transaction['block_address'])
        block_address = transaction['block_address']
        block = self.find_block(set_index, tag)
        if not (block or self.is_blocking):
            return # Nothing to contribute or gain
        transaction_type = transaction['type']

        if transaction_type in [BusTransaction.BUS_READ, BusTransaction.BUS_READ_X] and not self.shared_line_asserted():
            self.is_blocking = False
            self.request_bus({'type': BusTransaction.BUS_READ_ADDRESS, 'block_address': block_address})
        elif transaction_type == BusTransaction.BUS_UPGRADE and not self.shared_line_asserted():
            self.is_blocking = False
            block.state = CacheState.MESI_MODIFIED

    def resolve_mesi_miss(self, transaction, new_block):
        operation = self.is_blocking_info['operation']
        transaction_type = transaction['type']
        if operation == Operation.STORE:
            if transaction_type in [BusTransaction.FLUSH_OPT, BusTransaction.BUS_READ_DATA]:
                self.change_block_state(new_block, CacheState.MESI_MODIFIED)
                return True
        elif operation == Operation.LOAD:
            if transaction_type == BusTransaction.FLUSH_OPT:
                self.change_block_state(new_block, CacheState.MESI_SHARED)
                return True
            elif transaction_type == BusTransaction.BUS_READ_DATA:
                self.change_block_state(new_block, CacheState.MESI_EXCLUSIVE)
                return True
        return False

    def resolve_dragon_miss(self, transaction, new_block):
        operation = self.is_blocking_info['operation']
        transaction_type = transaction['type']
        if operation == Operation.STORE:
            if transaction_type in [BusTransaction.FLUSH, BusTransaction.BUS_UPDATE]:
                self.change_block_state(new_block, CacheState.DRAGON_SHARED_CLEAN)
                return True
            elif transaction_type == BusTransaction.BUS_READ_DATA:
                self.change_block_state(new_block, CacheState.DRAGON_MODIFIED)
                return True
        elif operation == Operation.LOAD:
            if transaction_type in [BusTransaction.FLUSH, BusTransaction.BUS_UPDATE]:
                self.change_block_state(new_block, CacheState.DRAGON_SHARED_CLEAN)
                return True
            elif transaction_type == BusTransaction.BUS_READ_DATA:
                self.change_block_state(new_block, CacheState.DRAGON_EXCLUSIVE)
                return True
        return False


    def resolve_cache_miss(self, transaction):
        tag = self.is_blocking_info['tag']
        set_index = self.is_blocking_info['set_index']
        new_block = CacheBlock(tag, set_index)
        new_block.address = self.is_blocking_info['block_address']
        new_block.address = transaction['block_address']
        resolved = False
        if self.is_blocking and transaction['block_address'] == self.is_blocking_info['block_address']:
            if self.protocol == 'MESI':
                resolved = self.resolve_mesi_miss(transaction, new_block)
            elif self.protocol == 'Dragon':
                resolved = self.resolve_dragon_miss(transaction, new_block)

            if resolved:
                self.is_blocking = False
                self.cache[self.is_blocking_info['set_index']].append(new_block)
                if self.protocol == 'Dragon' and self.is_blocking_info['operation'] == Operation.STORE and transaction['type'] in [BusTransaction.FLUSH, BusTransaction.BUS_UPDATE]:
                    self.request_bus({'type': BusTransaction.BUS_UPDATE, 'block_address': transaction['block_address']})

        return resolved


    def snoop_mesi_other(self, transaction):
        set_index, tag = self.address_handler.get_set_index_and_tag(transaction['block_address'])
        block_address = transaction['block_address']
        block = self.find_block(set_index, tag)
        if not (block or self.is_blocking):
            return # Nothing to contribute or gain
        transaction_type = transaction['type']

        if transaction_type == BusTransaction.BUS_WRITEBACK_ACK and transaction['cpu_id'] == self.cpu_id:
            if transaction['block_address'] == self.pending_writeback_ack_address:
                self.handle_dram_ack()
                return # Time to fetch the block that replaces the evicted block

        if self.resolve_cache_miss(transaction):
            return # Nothing more to do regarding this snoop

        if not block or block.state == CacheState.INVALID:
            return # Nothing more to do regarding this snoop

        if transaction_type == BusTransaction.BUS_READ:
            if not self.shared_line_asserted():
                self.request_bus_snooping({'type': BusTransaction.FLUSH_OPT, 'block_address': block_address})
            self.change_block_state(block, CacheState.MESI_SHARED)
        elif transaction_type == BusTransaction.BUS_READ_X:
            if not self.shared_line_asserted():
                self.request_bus_snooping({'type': BusTransaction.FLUSH_OPT, 'block_address': block_address})
            self.change_block_state(block, CacheState.INVALID)
        elif transaction_type == BusTransaction.BUS_UPGRADE:
            self.change_block_state(block, CacheState.INVALID)
        elif transaction_type == BusTransaction.BUS_WRITEBACK:
            pass

    def snoop_dragon_self(self, transaction):
        NotImplemented

    def snoop_dragon_other(self, transaction):
        set_index, tag = self.address_handler.get_set_index_and_tag(transaction['block_address'])
        block_address = transaction['block_address']
        block = self.find_block(set_index, tag)
        if not (block or self.is_blocking):
            return # Nothing to contribute or gain
        transaction_type = transaction['type']

        if transaction_type == BusTransaction.BUS_WRITEBACK_ACK and transaction['cpu_id'] == self.cpu_id:
            if transaction['block_address'] == self.pending_writeback_ack_address:
                self.handle_dram_ack()
                return # Time to fetch the block that replaces the evicted block

        if self.resolve_cache_miss(transaction):
            return # Nothing more to do regarding this snoop

        if not block or block.state == CacheState.INVALID:
            return # Nothing more to do regarding this snoop
        
        if transaction_type == BusTransaction.BUS_READ:
            if not self.shared_line_asserted():
                self.request_bus_snooping({'type': BusTransaction.FLUSH, 'block_address': block_address})
            if block.state == CacheState.DRAGON_MODIFIED:
                self.change_block_state(block, CacheState.DRAGON_SHARED_MODIFIED)
            elif block.state == CacheState.DRAGON_EXCLUSIVE:
                self.change_block_state(block, CacheState.DRAGON_SHARED_CLEAN)
        elif transaction_type == BusTransaction.BUS_UPDATE:
            if block.state in [CacheState.DRAGON_SHARED_MODIFIED, CacheState.DRAGON_SHARED_CLEAN]:
                self.change_block_state(block, CacheState.DRAGON_SHARED_CLEAN)
            else:
                raise ValueError("BUS_UPDATE should be impossible to receive if the block is not DRAGON_SHARED_MODIFIED or DRAGON_SHARED_CLEAN")

    def snoop_mesi(self, transaction):
        if transaction['cpu_id'] == self.cpu_id and not transaction['from_dram']:
            self.snoop_mesi_self(transaction)
        else:
            self.snoop_mesi_other(transaction)

    def snoop_dragon(self, transaction):
        if transaction['cpu_id'] == self.cpu_id and not transaction['from_dram']:
            self.snoop_dragon_self(transaction)
        else:
            self.snoop_dragon_other(transaction)

    def snoop(self, transaction):
        if self.protocol == 'MESI':
            self.snoop_mesi(transaction)
        elif self.protocol == 'Dragon':
            self.snoop_dragon(transaction)

    def request_bus_snooping(self, transaction):
        transaction_copy = copy.deepcopy(transaction)
        transaction_copy['cpu_id'] = self.cpu_id
        transaction_copy['from_dram'] = False
        self.bus.request(transaction_copy)

    def request_bus(self, transaction):
        transaction_copy = copy.deepcopy(transaction)  # Isolated copy
        if self.is_blocking: 
            raise RuntimeError("CPU is already waiting for the bus, cannot make another request originating from the CPU")
        self.is_blocking = True # Should only block when servicing bus requests originating from the CPU
        transaction_copy['cpu_id'] = self.cpu_id
        transaction_copy['from_dram'] = False
        self.bus.request(transaction_copy)

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
            return instructions
        except FileNotFoundError:
            sys.exit(1)

    def step(self):
        if self.instruction_pointer < len(self.trace):
            self.total_cycles += 1
        
        if self.compute_cycles_remaining > 0:
            self.compute_cycles_remaining -= 1
            self.compute_cycles += 1
            return

        if self.cache.is_blocking:
            self.idle_cycles += 1
            return

        if self.instruction_pointer >= len(self.trace):
            return

        # We only arrive here if there are no compute cycles remaining and no outstanding memory requests
        instruction, value = self.trace[self.instruction_pointer]
        if instruction in [0, 1]: 
            address = int(value, 16) & 0xFFFFFFFF # Ensure 32-bit
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
        cycle_snapshot = {}  # Store state snapshots every 1000 cycles
        while not self.all_programs_finished():
            self.global_cycle += 1
            for cpu in self.cpus:
                cpu.step()
            for cache in self.caches:
                cache.step()
            for cache in self.caches:
                cache.shared_line = False
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