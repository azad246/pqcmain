import os
import base64
import hashlib
import warnings
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives import serialization
from cryptography.fernet import Fernet

PQC_AVAILABLE = False
try:
    import oqs
    PQC_AVAILABLE = True
except ImportError:
    pass

class PQCManager:
    def __init__(self, algorithm='Kyber512'):
        self.algorithm = algorithm
        self.use_pqc = PQC_AVAILABLE
        
        if self.use_pqc:
            try:
                # Test if the specific KEM algorithm is actually supported by the local liboqs binaries
                with oqs.KeyEncapsulation(self.algorithm) as kem:
                    self.kem_name = kem.details['name']
                    self.kem_length_public_key = kem.details['length_public_key']
                    self.kem_length_secret_key = kem.details['length_secret_key']
                    self.kem_length_ciphertext = kem.details['length_ciphertext']
                    self.security_level = kem.details['claimed_nist_level']
                print(f"[PQC Manager] Successfully initialized liboqs with algorithm: {self.kem_name}")
            except Exception as e:
                print(f"[WARNING] liboqs error with algorithm '{self.algorithm}': {e}")
                print("Falling back to RSA simulated PQC.")
                self.use_pqc = False

        if not self.use_pqc:
            self.algorithm = "RSA-2048 (Simulated Fallback)"
            print(f"[PQC Manager] Using SIMULATED Fallback Algorithm: {self.algorithm}")

    def generate_keypair(self):
        """Generates a Public and Secret Keypair."""
        if self.use_pqc:
            with oqs.KeyEncapsulation(self.algorithm) as kem:
                public_key = kem.generate_keypair()
                secret_key = kem.export_secret_key()
                return public_key, secret_key
        else:
            # RSA Fallback logic
            private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
            public_key = private_key.public_key()
            
            pub_bytes = public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            sec_bytes = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
            return pub_bytes, sec_bytes

    def encrypt_data(self, data_bytes, public_key):
        """
        Hybrid Encryption Pipeline:
        1. Establishes a shared symmetric key via Kyber KEM (or RSA encrypting a random session key).
        2. Encrypts the potentially large data payload (e.g. FL weights) using symmetric encryption (Fernet/AES).
        3. Returns the KEM encapsulated payload bundled safely with the encrypted data.
        """
        if self.use_pqc:
            with oqs.KeyEncapsulation(self.algorithm) as kem:
                # Kyber generates a shared secret securely alongside a ciphertext token
                kem_ciphertext, shared_secret = kem.encap_secret(public_key)
                
                # Fernet (AES-128-CBC + HMAC) requires exactly 32 url-safe base64 bytes.
                # Kyber512 output is exactly 32 bytes, but we hash it just to guarantee dimensions across all algos.
                if len(shared_secret) != 32:
                    shared_secret = hashlib.sha256(shared_secret).digest()
                    
                symmetric_key = base64.urlsafe_b64encode(shared_secret)
                cipher = Fernet(symmetric_key)
                encrypted_data = cipher.encrypt(data_bytes)
                
                # Prepend the exact byte-length of the KEM ciphertext (using 4 bytes) to safely unpack later
                kem_len = len(kem_ciphertext).to_bytes(4, byteorder='big')
                return kem_len + kem_ciphertext + encrypted_data
        else:
            # RSA Hybrid Fallback pipeline
            session_key = os.urandom(32)
            symmetric_key = base64.urlsafe_b64encode(session_key)
            cipher = Fernet(symmetric_key)
            encrypted_data = cipher.encrypt(data_bytes)
            
            loaded_public_key = serialization.load_pem_public_key(public_key)
            rsa_ciphertext = loaded_public_key.encrypt(
                session_key,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            rsa_len = len(rsa_ciphertext).to_bytes(4, byteorder='big')
            return rsa_len + rsa_ciphertext + encrypted_data

    def decrypt_data(self, ciphertext_bundle, secret_key):
        """
        Hybrid Decryption Pipeline:
        1. Extracts the KEM/RSA encrypted session key header.
        2. Decapsulates to recover the underlying symmetric shared secret.
        3. Decrypts the actual payload data using Fernet.
        """
        # Unpack the bundle cleanly
        encap_len = int.from_bytes(ciphertext_bundle[:4], byteorder='big')
        encap_ciphertext = ciphertext_bundle[4:4+encap_len]
        encrypted_data = ciphertext_bundle[4+encap_len:]
        
        if self.use_pqc:
            with oqs.KeyEncapsulation(self.algorithm) as kem:
                kem.secret_key = secret_key
                shared_secret = kem.decap_secret(encap_ciphertext)
                
                if len(shared_secret) != 32:
                    shared_secret = hashlib.sha256(shared_secret).digest()
                    
                symmetric_key = base64.urlsafe_b64encode(shared_secret)
                cipher = Fernet(symmetric_key)
                return cipher.decrypt(encrypted_data)
        else:
            # RSA Hybrid Decryption
            loaded_private_key = serialization.load_pem_private_key(secret_key, password=None)
            session_key = loaded_private_key.decrypt(
                encap_ciphertext,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            symmetric_key = base64.urlsafe_b64encode(session_key)
            cipher = Fernet(symmetric_key)
            return cipher.decrypt(encrypted_data)

    def get_algorithm_info(self):
        """Prints extensive details about the active cryptographic backend."""
        print("\n" + "="*50)
        print("PQC-IoT Sentinel Cryptography Specs")
        print("="*50)
        if self.use_pqc:
            print(f"Status:          ACTIVE (liboqs natively running)")
            print(f"Algorithm:       {self.kem_name}")
            print(f"NIST Sec Level:  {self.security_level}")
            print(f"Public Key Size: {self.kem_length_public_key} bytes")
            print(f"Secret Key Size: {self.kem_length_secret_key} bytes")
            print(f"Ciphertext Size: {self.kem_length_ciphertext} bytes")
        else:
            print(f"Status:          SIMULATED (liboqs C-Library Missing)")
            print(f"Algorithm:       {self.algorithm}")
            print(f"NIST Sec Level:  Pre-Quantum (RSA Vulnerable to Shor's)")
            print(f"Public Key Size: 2048 bits")
            print(f"Secret Key Size: 2048 bits")
        print("="*50 + "\n")

# -------------------------------------------------------------------------
# Exposed Module API
# -------------------------------------------------------------------------
_default_manager = PQCManager()

def generate_keypair(algorithm='Kyber512'):
    if algorithm != _default_manager.algorithm and PQC_AVAILABLE:
        # Re-initialize on-the-fly if a different algorithm is requested
        custom_manager = PQCManager(algorithm=algorithm)
        return custom_manager.generate_keypair()
    return _default_manager.generate_keypair()

def encrypt_data(data_bytes, public_key):
    return _default_manager.encrypt_data(data_bytes, public_key)

def decrypt_data(ciphertext, secret_key):
    return _default_manager.decrypt_data(ciphertext, secret_key)

def get_algorithm_info():
    _default_manager.get_algorithm_info()

if __name__ == "__main__":
    get_algorithm_info()
    
    # Built-in self-test to verify crypto integrity
    print("Running Hybrid-Encryption Self-Test...")
    pub, sec = generate_keypair()
    
    message = b"PQC-IoT Sentinel: Secure FL Model Weights Transmission Payload!"
    print(f"Original Message: {message}")
    
    ciphertext = encrypt_data(message, pub)
    print(f"Encrypted Bundle Length: {len(ciphertext)} bytes")
    
    decrypted = decrypt_data(ciphertext, sec)
    print(f"Decrypted Message: {decrypted}")
    
    assert message == decrypted, "[ERROR] Encryption/Decryption mismatch!"
    print("\nSelf-test PASSED: Data encrypted and decrypted flawlessly.")
