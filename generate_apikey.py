import secrets
import hashlib

def generate_api_token(nbytes=32):
    return secrets.token_hex(nbytes)

def generate_random_sha256():
    random_bytes = secrets.token_bytes(32)
    hash_object = hashlib.sha256(random_bytes)
    return hash_object.hexdigest()

if __name__ == "__main__":
    token = generate_api_token(32)
    print(f"{token}")
    print(f"Length: {len(token)} characters")

    hashed_val = generate_random_sha256()
    print(f"{hashed_val}")
    
    url_safe = secrets.token_urlsafe(32)
    print(f"{url_safe}")