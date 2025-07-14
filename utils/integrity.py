import hashlib
import zlib

def md5_checksum(file_path: str) -> str:
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def crc32_checksum(file_path: str) -> int:
    prev = 0
    with open(file_path, "rb") as f:
        for line in f:
            prev = zlib.crc32(line, prev)
    return prev & 0xFFFFFFFF
