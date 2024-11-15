import sys
import math
import copy
from enum import Enum, auto
from collections import deque, defaultdict
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
    DRAM_READ_ADDRESS = auto() # requesting DRAM for a cache line
    DRAM_READ_DATA = auto() # dram response of aforementioned request
    DRAM_FLUSH = auto() # evicting a dirty block to DRAM
    DRAM_FLUSH_ACK = auto() # acknowledge from DRAM that block is flushed successfully
    # Bus transaction between caches
    READ = auto() # the caches test this before going to main memory
    FLUSH_TEST = auto() # the caches test this before going to main memory
    # MESI-specific bus transactions between caches
    MESI_READ_X = auto()
    MESI_UPGRADE = auto()
    MESI_FLUSH_OPT = auto()
    # Dragon-specific bus transactions between caches
    DRAGON_UPDATE = auto()

class DRAMTransactions(Enum):
    READ_ADDRESS = BusTransaction.DRAM_READ_ADDRESS
    READ_DATS = BusTransaction.DRAM_READ_DATA
    FLUSH = BusTransaction.DRAM_FLUSH
    FLUSH_ACK = BusTransaction.DRAM_FLUSH_ACK

class DRAM:
    def __init__(self, total_access_latency, bus):
        self.bus = bus
        self.total_access_latency = total_access_latency
        self.request_queue = []  # for storing DRAM requests
        self.total_transfer_latency = 8 + 1 # 8 cycles for data and 1 cycle for address or flush_ack

    def request_bus(self, transaction):
        # DRAM is finished processing the request and must out the corresponding response on the bus
        if transaction['type'] == BusTransaction.DRAM_READ_ADDRESS:
            transaction['type'] = BusTransaction.DRAM_READ_DATA
        elif transaction['type'] == BusTransaction.DRAM_FLUSH:
            transaction['type'] = BusTransaction.DRAM_FLUSH_ACK
        transaction['from_dram'] = True
        self.bus.request(transaction)

    def decrement_cycle_counters(self):
        for request in self.request_queue:
            request['cycles_remaining'] -= 1

    def snoop(self, transaction):
        if transaction['type'] in [BusTransaction.DRAM_READ_ADDRESS, BusTransaction.DRAM_FLUSH]:
            transaction['cycles_remaining'] = self.total_access_latency - self.total_transfer_latency
            self.request_queue.append(transaction)

    def step(self):
        # Simulates one clock cycle of the DRAM controller's operation.
        if self.bus.new_transaction:
            self.snoop(self.bus.current_transaction)
        self.decrement_cycle_counters()
        if not any(request['cycles_remaining'] == 0 for request in self.request_queue):
            return
        old_queue_size = len(self.request_queue)
        for request in self.request_queue:
            if request['cycles_remaining'] == 0:
                self.request_bus(request)
                # guaranteed to only delete the request the DRAM responds to, due to the FIFO nature with max 1 incoming request per cycle
                self.request_queue = [req for req in self.request_queue if req['cycles_remaining'] > 0]
                assert len(self.request_queue) == old_queue_size - 1 # using asserts to 'prove' correctness

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
        assert block_address is not None # just to be safe
        index = (block_address >> self.block_offset_bits) & (self.num_sets - 1)
        tag = block_address >> (self.index_bits + self.block_offset_bits)
        return index, tag

    def get_block_address(self, address):
        assert address is not None
        return address >> self.block_offset_bits

class CacheStatistics:
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.private_accesses = 0
        self.shared_accesses = 0
        self.total_cycles = 0
        self.adoptions_in = 0
        self.adoptions_out = 0

    def record_access(self, hit):
        if hit:
            self.hits += 1
        else:
            self.misses += 1

    def get_miss_rate(self):
        total = self.hits + self.misses
        return self.misses / total if total > 0 else 0

    def get_stats(self):
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': 1 - self.get_miss_rate(),
            'private_accesses': self.private_accesses,
            'shared_accesses': self.shared_accesses,
            'adoptions_in': self.adoptions_in,
            'adoptions_out': self.adoptions_out
        }

class Bus:
    def __init__(self, block_size):
        self.dram = None # will be set by Simulator during initialization
        self.caches = None # will be set by Simulator during initialization
        self.queue: deque = deque()
        self.dram_queue: deque = deque()

        self.word_size = 4
        self.block_size = block_size
        
        self.new_transaction = False
        self.current_transaction = None
        self.transfer_cycles_remaining = 0

        self.stats = {
            'invalidations': 0,
            'updates': 0,
            'data_traffic': 0
            }

    def request(self, transaction):
        # entry point for all bus requests, so we assert here that it's a valid request
        assert transaction is not None
        if transaction['type'] in {item.value for item in DRAMTransactions}:
            self.dram_queue.append(transaction)
        else:
            self.queue.append(transaction)

    def grant(self, transaction):
        self.new_transaction = True # only True for first cycle, to trigger cache snooping
        self.current_transaction = transaction
        self.update_traffic_stats(transaction)
        self.transfer_cycles_remaining = self.calculate_transfer_cycles(transaction) - 1 # -1 to account for current cycle
        for cache in self.caches:
            cache.update_shared_line(transaction)
        if transaction['type'] == BusTransaction.FLUSH_TEST:
            self.handle_flush_test()

    def release(self):
        assert self.transfer_cycles_remaining == 0
        self.current_transaction = None

    def handle_flush_test(self):
        # here the bus acts like a simple MCU that just upgrades a flush_test to a flush
        if any(cache.shared_line for cache in self.caches):
            return # the caches handled the flush locally by adopting the evicted cache line
        transaction = self.current_transaction
        transaction['type'] = BusTransaction.DRAM_FLUSH
        self.grant(transaction) # can 'skip the queue' since in principle the MCU just upgrades a flush_test to a flush

    def step(self):
        self.new_transaction = False

        if self.transfer_cycles_remaining > 0:
            self.transfer_cycles_remaining -= 1
            return

        if self.current_transaction:
            self.release()

        # Give priority to DRAM responses and requests
        if self.dram_queue:
            self.grant(self.dram_queue.popleft())
            return
        
        if self.queue:
            self.grant(self.queue.popleft())
            return

    def calculate_transfer_cycles(self, transaction):
        words_in_block = self.block_size // self.word_size
        transaction_type = transaction['type']
        if transaction_type == BusTransaction.DRAM_READ_ADDRESS:
            return 1
        elif transaction_type == BusTransaction.DRAM_READ_DATA:
            return 2 * words_in_block
        elif transaction_type == BusTransaction.DRAM_FLUSH:
            return 2 * words_in_block
        elif transaction_type == BusTransaction.DRAM_FLUSH_ACK:
            return 1
        elif transaction_type == BusTransaction.READ:
            return 1
        elif transaction_type == BusTransaction.FLUSH_TEST:
            return 2 * words_in_block
        elif transaction_type == BusTransaction.MESI_READ_X:
            return 1
        elif transaction_type == BusTransaction.MESI_UPGRADE:
            return 1 
        elif transaction_type == BusTransaction.MESI_FLUSH_OPT:
            return 2 * words_in_block
        elif transaction_type == BusTransaction.DRAGON_UPDATE:
            return 2 * words_in_block
        raise ValueError(f"Unknown transaction type: {transaction_type}")

    def update_traffic_stats(self, transaction: Dict):
        transaction_type = transaction['type']
        if transaction_type == BusTransaction.DRAM_READ_ADDRESS:
            pass 
        elif transaction_type == BusTransaction.DRAM_READ_DATA:
            self.stats['data_traffic'] += self.block_size
        elif transaction_type == BusTransaction.DRAM_FLUSH:
            self.stats['data_traffic'] += self.block_size
        elif transaction_type == BusTransaction.DRAM_FLUSH_ACK:
            pass
        elif transaction_type == BusTransaction.READ:
            pass
        elif transaction_type == BusTransaction.FLUSH_TEST:
            pass
        elif transaction_type == BusTransaction.MESI_READ_X:
            self.stats['invalidations'] += 1
        elif transaction_type == BusTransaction.MESI_UPGRADE:
            self.stats['invalidations'] += 1
        elif transaction_type == BusTransaction.MESI_FLUSH_OPT:
            self.stats['data_traffic'] += self.block_size
        elif transaction_type == BusTransaction.DRAGON_UPDATE:
            self.stats['data_traffic'] += self.block_size
            self.stats['updates'] += 1
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
        self.stats = CacheStatistics()
        self.protocol = protocol 

        self.bus = bus
        self.word_size = 4 # 32 bit architecture
        self.size = size
        self.block_size = block_size
        self.associativity = associativity
        self.num_sets = size // (block_size * associativity)
        self.cache = {i: deque(maxlen=associativity) for i in range(self.num_sets)}
        self.address_handler = AddressHandler(block_size, self.num_sets)

        self.shared_line = False
        self.pending_writeback_ack_address = None
        self.is_resolving_cache_miss = False
        self.cache_miss_info = None

    def find_block(self, set_index, tag):
        for block in self.cache[set_index]:
            if block and block.tag == tag:
                return block
        return None

    def change_block_state(self, block, new_state):
        assert block is not None
        block.state = new_state

    def request_bus(self, transaction):
        if transaction['type'] in [BusTransaction.FLUSH_TEST, BusTransaction.DRAM_FLUSH]:
            assert self.is_resolving_cache_miss
            assert transaction['block_address'] is not None
            self.pending_writeback_ack_address = transaction['block_address']
        transaction['cpu_id'] = self.cpu_id
        transaction['from_dram'] = False
        self.bus.request(transaction)

    def update_lru(self, block):
        set_index = block.set_index
        if self.cache[set_index][-1] != block: # Update only if the block is not already the most recently used
            self.cache[set_index].remove(block)
            self.cache[set_index].append(block)

    def fill_cache(self, block_address, state):
        set_index, tag = self.address_handler.get_set_index_and_tag(block_address)
        block = CacheBlock(tag, set_index)
        block.address = block_address
        block.state = state
        if len(self.cache[set_index]) == self.associativity:
            self.cache[set_index].popleft()
        self.cache[set_index].append(block)

    def access_cache(self, address, operation):
        assert (address is not None and operation in [Operation.LOAD, Operation.STORE])
        block_address = self.address_handler.get_block_address(address)
        set_index, tag = self.address_handler.get_set_index_and_tag(block_address)
        block = self.find_block(set_index, tag)
        if block and block.state != CacheState.INVALID:
            self.process_cache_hit(block, operation, block_address)
        else:
            self.stats.shared_accesses += 1 # misses are always counted as shared accesses
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
            self.stats.private_accesses += 1
        elif block.state == CacheState.MESI_EXCLUSIVE:
            if operation == Operation.STORE:
                self.change_block_state(block, CacheState.MESI_MODIFIED)
            self.stats.private_accesses += 1
        elif block.state == CacheState.MESI_SHARED:
            if operation == Operation.STORE:
                self.request_bus({'type': BusTransaction.MESI_UPGRADE, 'block_address': block_address})
            self.stats.shared_accesses += 1
        else:
            raise ValueError(f"Invalid block state: {block.state}")

    def dragon_hit(self, block, operation, block_address):
        if block.state == CacheState.DRAGON_MODIFIED:
            self.stats.private_accesses += 1
        elif block.state == CacheState.DRAGON_EXCLUSIVE:
            if operation == Operation.STORE:
                self.change_block_state(block, CacheState.DRAGON_MODIFIED)
            self.stats.private_accesses += 1
        # equal logic for Sc and Sm here, state update happens during self-snooping
        elif block.state in [CacheState.DRAGON_SHARED_CLEAN, CacheState.DRAGON_SHARED_MODIFIED]:
            if operation == Operation.STORE:
                self.request_bus({'type': BusTransaction.DRAGON_UPDATE, 'block_address': block_address})
            self.stats.shared_accesses += 1
        else:
            raise ValueError(f"Invalid block state: {block.state}")

    def process_cache_miss(self, set_index, tag, operation, block_address):
        self.is_resolving_cache_miss = True
        self.stats.record_access(False)
        self.cache_miss_info = {
            'set_index': set_index,
            'tag': tag,
            'operation': operation,
            'block_address': block_address
        }
        evicted_address = self.evict_and_writeback_needed(set_index)
        if evicted_address:
            self.request_bus({'type': BusTransaction.FLUSH_TEST, 'block_address': evicted_address})
            return # Will request block after successful writeback
        # No eviction
        self.request_block_on_miss()

    def evict_and_writeback_needed(self, set_index):
        if len(self.cache[set_index]) == self.associativity:
            evicted_block = self.cache[set_index][0]
            if evicted_block is None:
                return False
            evicted_address = copy.deepcopy(evicted_block.address)
            if evicted_block.state in [CacheState.MESI_MODIFIED, CacheState.DRAGON_MODIFIED, CacheState.DRAGON_SHARED_MODIFIED]:
                self.cache[set_index][0] = None
                return evicted_address
            self.cache[set_index][0] = None
        return False

    # successfull writeback due to eviction on cache miss
    # now we need to perform the memory operation that caused the cache miss
    def handle_dram_ack(self):
        assert (self.pending_writeback_ack_address and self.is_resolving_cache_miss)
        self.pending_writeback_ack_address = None
        self.request_block_on_miss()

    def request_block_on_miss(self):
        operation = self.cache_miss_info['operation']
        block_address = self.cache_miss_info['block_address']
        if self.protocol == 'MESI':
            if operation == Operation.STORE:
                transaction_type = BusTransaction.MESI_READ_X
            elif operation == Operation.LOAD:
                transaction_type = BusTransaction.READ
        elif self.protocol == 'Dragon':
            transaction_type = BusTransaction.READ
        self.request_bus({'type': transaction_type, 'block_address': block_address})

    def step(self):
        if self.bus.new_transaction:
            self.snoop(self.bus.current_transaction)

    def snoop(self, transaction):
        if self.protocol == 'MESI':
            self.snoop_mesi(transaction)
        elif self.protocol == 'Dragon':
            self.snoop_dragon(transaction)

    def snoop_mesi(self, transaction):
        if transaction['cpu_id'] == self.cpu_id and not transaction['from_dram']:
            self.snoop_mesi_self(transaction)
        else:
            self.snoop_mesi_other(transaction)

    def snoop_mesi_self(self, transaction):
        set_index, tag = self.address_handler.get_set_index_and_tag(transaction['block_address'])
        block_address = transaction['block_address']
        block = self.find_block(set_index, tag)
        transaction_type = transaction['type']

        if transaction_type in [BusTransaction.READ, BusTransaction.MESI_READ_X] and not self.shared_line_asserted_by_other_cache():
            if self.is_resolving_cache_miss and transaction['block_address'] == self.cache_miss_info['block_address']:
                # still resolving a cache miss and no other cache can help us out
                self.request_bus({'type': BusTransaction.DRAM_READ_ADDRESS, 'block_address': block_address})
        elif transaction_type == BusTransaction.MESI_UPGRADE:
            if block.state == CacheState.INVALID: # this cache lost the data race / bus_upgrade race, need to get new priveleges
                self.process_cache_miss(set_index, tag, Operation.STORE, block_address)
            else:
                self.change_block_state(block, CacheState.MESI_MODIFIED)
        elif transaction_type == BusTransaction.FLUSH_TEST:
            if self.shared_line_asserted_by_other_cache(): # someone adopted our block
                self.stats.adoptions_out += 1
                self.handle_dram_ack()
            else:
                pass # the MCU will turn flush_test into a flush if not adopted

    def snoop_mesi_other(self, transaction):
        set_index, tag = self.address_handler.get_set_index_and_tag(transaction['block_address'])
        block_address = transaction['block_address']
        block = self.find_block(set_index, tag)
        transaction_type = transaction['type']

        if transaction_type == BusTransaction.DRAM_FLUSH_ACK and transaction['cpu_id'] == self.cpu_id:
            if transaction['block_address'] == self.pending_writeback_ack_address:
                self.handle_dram_ack()
                return # time to fetch the block that replaces the evicted block

        if self.is_resolving_cache_miss and not self.pending_writeback_ack_address:
            if transaction['block_address'] == self.cache_miss_info['block_address']:
                if self.resolve_cache_miss(transaction):
                    return # nothing more to do regarding this snoop

        if transaction_type == BusTransaction.FLUSH_TEST: # testing this first so we can do early return
            if self.shared_line:
                self.stats.adoptions_in += 1
                self.fill_cache(block_address, CacheState.MESI_MODIFIED)
                return

        if not block or block.state == CacheState.INVALID:
            return # nothing the cache can help out with

        if transaction_type == BusTransaction.READ:
            if not self.shared_line_asserted_by_other_cache():
                self.request_bus({'type': BusTransaction.MESI_FLUSH_OPT, 'block_address': block_address})
            self.change_block_state(block, CacheState.MESI_SHARED)
        elif transaction_type == BusTransaction.MESI_READ_X:
            if not self.shared_line_asserted_by_other_cache():
                self.request_bus({'type': BusTransaction.MESI_FLUSH_OPT, 'block_address': block_address})
            self.change_block_state(block, CacheState.INVALID)
        elif transaction_type == BusTransaction.MESI_UPGRADE:
            self.change_block_state(block, CacheState.INVALID)

    def snoop_dragon(self, transaction):
        if transaction['cpu_id'] == self.cpu_id and not transaction['from_dram']:
            self.snoop_dragon_self(transaction)
        else:
            self.snoop_dragon_other(transaction)

    def snoop_dragon_self(self, transaction):
        set_index, tag = self.address_handler.get_set_index_and_tag(transaction['block_address'])
        block_address = transaction['block_address']
        block = self.find_block(set_index, tag)
        transaction_type = transaction['type']

        if transaction_type == BusTransaction.READ and not self.shared_line_asserted_by_other_cache():
            if self.is_resolving_cache_miss and transaction['block_address'] == self.cache_miss_info['block_address']:
                self.request_bus({'type': BusTransaction.DRAM_READ_ADDRESS, 'block_address': block_address})
        elif transaction_type == BusTransaction.DRAGON_UPDATE and block:
            self.change_block_state(block, CacheState.DRAGON_SHARED_MODIFIED)
        elif transaction_type == BusTransaction.FLUSH_TEST:
            if self.shared_line_asserted_by_other_cache(): # someone adopted our block
                self.stats.adoptions_out += 1
                self.handle_dram_ack()
            else:
                pass # the MCU will turn flush_test into a flush if not adopted

    def snoop_dragon_other(self, transaction):
        set_index, tag = self.address_handler.get_set_index_and_tag(transaction['block_address'])
        block_address = transaction['block_address']
        block = self.find_block(set_index, tag)
        transaction_type = transaction['type']

        if transaction_type == BusTransaction.DRAM_FLUSH_ACK and transaction['cpu_id'] == self.cpu_id:
            if transaction['block_address'] == self.pending_writeback_ack_address:
                self.handle_dram_ack()
                return # Time to fetch the block that replaces the evicted block

        if self.is_resolving_cache_miss and not self.pending_writeback_ack_address:
            if transaction['block_address'] == self.cache_miss_info['block_address']:
                if self.resolve_cache_miss(transaction):
                    return # nothing more to do regarding this snoop

        if transaction_type == BusTransaction.FLUSH_TEST: # testing this first so we can do early return
            if self.shared_line:
                self.stats.adoptions_in += 1
                self.fill_cache(block_address, CacheState.DRAGON_SHARED_MODIFIED)
                return

        if not block:
            return # nothing the cache can help out with
        
        if transaction_type == BusTransaction.READ:
            if not self.shared_line_asserted_by_other_cache(): # we help out
                self.request_bus({'type': BusTransaction.DRAGON_UPDATE, 'block_address': block_address})
            # ...and change the block state accordingly
            if block.state == CacheState.DRAGON_MODIFIED:
                self.change_block_state(block, CacheState.DRAGON_SHARED_MODIFIED)
            elif block.state == CacheState.DRAGON_EXCLUSIVE:
                self.change_block_state(block, CacheState.DRAGON_SHARED_CLEAN)
        elif transaction_type == BusTransaction.DRAGON_UPDATE:
            self.change_block_state(block, CacheState.DRAGON_SHARED_CLEAN)

    def update_shared_line(self, transaction):
        if transaction['cpu_id'] == self.cpu_id:
            return # transaction is from this cache
        if self.shared_line_asserted_by_other_cache():
            return # shared line already asserted by another cache
        if transaction['type'] in {item.value for item in DRAMTransactions}:
            return # transaction is to or from DRAM, the cache shall not interfere

        block_address = transaction['block_address']
        set_index, tag = self.address_handler.get_set_index_and_tag(block_address)
        block = self.find_block(set_index, tag)

        if transaction['type'] == BusTransaction.FLUSH_TEST:
            if block or len(self.cache[set_index]) < self.associativity:
                self.shared_line = True # the cache can adopt the block if corresponding block already in cache or there is space in set
            return

        if block and block.state != CacheState.INVALID:
            self.shared_line = True # signal that the cache has a valid copy of the block and can share it if necessary when snooping
            # ...in addition, this can give Sm->M in the Dragon protocol if no other cache asserts the line during a bus_update

    def shared_line_asserted_by_other_cache(self):
        return any(cache.shared_line for cache in self.bus.caches if cache.cpu_id != self.cpu_id)

    def resolve_cache_miss(self, transaction):
        assert (self.is_resolving_cache_miss and not self.pending_writeback_ack_address)
        set_index = self.cache_miss_info['set_index']
        tag = self.cache_miss_info['tag']
        set_index = self.cache_miss_info['set_index']
        new_block = CacheBlock(tag, set_index)
        block_address = transaction['block_address']
        new_block.address = block_address
        resolved = False

        if self.protocol == 'MESI':
            resolved = self.resolve_mesi_miss(transaction, new_block)
        elif self.protocol == 'Dragon':
            resolved = self.resolve_dragon_miss(transaction, new_block)

        if resolved:
            self.is_resolving_cache_miss = False
            self.fill_cache(block_address, new_block.state)
            if self.protocol == 'Dragon' and self.cache_miss_info['operation'] == Operation.STORE and transaction['type'] == BusTransaction.DRAGON_UPDATE:
            # the cache has retrieved the block, but must send a bus update to get to Shared Modified
            # but we can un-block the cpu since the cache miss is per definition resolved
                self.request_bus({'type': BusTransaction.DRAGON_UPDATE, 'block_address': transaction['block_address']})

        return resolved

    def resolve_mesi_miss(self, transaction, new_block):
        operation = self.cache_miss_info['operation']
        transaction_type = transaction['type']
        if operation == Operation.STORE:
            if transaction_type in [BusTransaction.MESI_FLUSH_OPT, BusTransaction.DRAM_READ_DATA]:
                self.change_block_state(new_block, CacheState.MESI_MODIFIED)
                return True
        elif operation == Operation.LOAD:
            if transaction_type == BusTransaction.MESI_FLUSH_OPT:
                self.change_block_state(new_block, CacheState.MESI_SHARED)
                return True
            elif transaction_type == BusTransaction.DRAM_READ_DATA:
                self.change_block_state(new_block, CacheState.MESI_EXCLUSIVE)
                return True
        return False

    def resolve_dragon_miss(self, transaction, new_block):
        operation = self.cache_miss_info['operation']
        transaction_type = transaction['type']
        if operation == Operation.STORE:
            if transaction_type == BusTransaction.DRAGON_UPDATE:
                self.change_block_state(new_block, CacheState.DRAGON_SHARED_CLEAN)
                return True
            elif transaction_type == BusTransaction.DRAM_READ_DATA:
                self.change_block_state(new_block, CacheState.DRAGON_MODIFIED)
                return True
        elif operation == Operation.LOAD:
            if transaction_type == BusTransaction.DRAGON_UPDATE:
                self.change_block_state(new_block, CacheState.DRAGON_SHARED_CLEAN)
                return True
            elif transaction_type == BusTransaction.DRAM_READ_DATA:
                self.change_block_state(new_block, CacheState.DRAGON_EXCLUSIVE)
                return True
        return False

class CPU:
    def __init__(self, cpu_id, cache, trace_file):
        self.cpu_id = cpu_id
        self.cache = cache
        self.trace = self.load_trace(trace_file)

        self.finished = False

        self.total_cycles = 0
        self.compute_cycles = 0
        self.idle_cycles = 0
        self.load_store_instructions = 0
        
        self.instruction_pointer = 0
        self.compute_cycles_remaining = 0

        self.stats = {
            'total_cycles': 0,
            'compute_cycles': 0,
            'idle_cycles': 0,
            'load_store_instructions': 0
        }

    def load_trace(self, trace_file):
        instructions = []
        try:
            with open(trace_file, 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    assert len(parts) == 2
                    instruction, value = parts
                    instructions.append((int(instruction), value))
            return instructions
        except FileNotFoundError:
            sys.exit(1)

    def step(self):
        if self.instruction_pointer < len(self.trace):
            self.stats['total_cycles'] += 1
        
        if self.compute_cycles_remaining > 0:
            self.compute_cycles_remaining -= 1
            self.stats['compute_cycles'] += 1
            return

        if self.cache.is_resolving_cache_miss:
            self.stats['idle_cycles'] += 1
            return

        if self.instruction_pointer >= len(self.trace):
            self.finished = True
            return

        # We only arrive here if there are no compute cycles remaining and no outstanding memory requests
        instruction, value = self.trace[self.instruction_pointer]
        if instruction in [0, 1]: 
            address = int(value, 16) & 0xFFFFFFFF # Ensure 32-bit
            operation = Operation.LOAD if instruction == 0 else Operation.STORE 
            self.stats['load_store_instructions'] += 1
            self.cache.access_cache(address, operation)
            self.stats['idle_cycles'] += 1 # Always 1 (cache hit) or more (cache miss) idle cycles during load/store
        elif instruction == 2: 
            cycles = int(value, 16)
            self.compute_cycles_remaining = cycles - 1 # Subtract 1 for the current cycle
            self.stats['compute_cycles'] += 1

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
        return all(cpu.finished for cpu in self.cpus)

    def generate_report(self):
        print("===== Simulation Report =====")
        print(f"Overall Execution Cycles: {self.global_cycle}")
        print(f"Total Data Traffic on Bus: {self.bus.stats['data_traffic']} bytes")
        
        if self.protocol == "MESI":
            print(f"Number of Invalidations: {self.bus.stats['invalidations']}\n")
        elif self.protocol == "Dragon":
            print(f"Number of Updates: {self.bus.stats['updates']}\n")
        
        for cpu in self.cpus:
            cache_stats = cpu.cache.stats.get_stats()
            print(f"--- CPU {cpu.cpu_id} ---")
            print(f"Total Cycles: {cpu.stats['total_cycles']}")
            print(f"Compute Cycles: {cpu.stats['compute_cycles']}")
            print(f"Load/Store Instructions: {cpu.stats['load_store_instructions']}")
            print(f"Idle Cycles: {cpu.stats['idle_cycles']}")
            print(f"Cache Hits: {cache_stats['hits']}")
            print(f"Cache Misses: {cache_stats['misses']}")
            print(f"Hit Rate: {cache_stats['hit_rate']:.2%}")
            print(f"Miss Rate: {(1 - cache_stats['hit_rate']):.2%}")
            print(f"Private Accesses: {cache_stats['private_accesses']}")
            print(f"Shared Accesses: {cache_stats['shared_accesses']}")
            print(f"Incoming adoptions: {cache_stats['adoptions_in']}")
            print(f"Outgoing adoptions: {cache_stats['adoptions_out']}\n")

        print(f"Sum of incoming adoptions: {sum([cache.stats.adoptions_in for cache in self.caches])}")
        print(f"Sum of outgoing adoptions: {sum([cache.stats.adoptions_out for cache in self.caches])}")
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

    assert not (args.cache_size <= 0 or args.associativity <= 0 or args.block_size <= 0)
    assert args.cache_size % (args.block_size * args.associativity) == 0

    trace_folder = '/Users/oysteinweibell/Desktop/MulticoreArch/assignment2/cs4223A2P1/traces'

    simulator = Simulator(args.protocol, f"{trace_folder}/{args.input_file}", args.cache_size, args.associativity, args.block_size)
    simulator.run()

if __name__ == "__main__":
    main()