import os
import struct


def generate_random_seed() -> int:
    seed, = struct.unpack('<I', os.urandom(4))
    return seed
