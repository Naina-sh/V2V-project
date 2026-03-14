from cryptography.fernet import Fernet

# PASTE YOUR GENERATED KEY HERE (INCLUDING b'...')
KEY = b'_btFQGcptuSNTRcNiOfIJEOG1MWuLu5KSvG2yy6qYwA='

cipher = Fernet(KEY)

def encrypt(msg: str) -> bytes:
    return cipher.encrypt(msg.encode())

def decrypt(data: bytes) -> str:
    return cipher.decrypt(data).decode()
