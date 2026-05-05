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

# =========================================================================
# DIGITAL SIGNATURES — Dilithium2 (PQC) with RSA-PSS fallback
# =========================================================================

class SignatureManager:
    """
    Post-Quantum digital signature support using Dilithium2 via liboqs.
    Falls back to RSA-PSS (SHA-256) when liboqs is unavailable.

    Signing flow:
      1. SHA-256 hash of the raw weight bytes is computed.
      2. Dilithium2 (or RSA-PSS) signs that digest.
      3. The signature is returned as raw bytes.

    Verification flow:
      1. SHA-256 hash of received bytes is computed.
      2. Dilithium2 (or RSA-PSS) verifies signature against the digest.
      3. Returns True only if the signature is valid — any tampering returns False.
    """

    SIG_ALGORITHM = 'Dilithium2'

    def __init__(self):
        self.use_pqc = False
        if PQC_AVAILABLE:
            try:
                with oqs.Signature(self.SIG_ALGORITHM) as _sig:
                    self.sig_name        = _sig.details['name']
                    self.pub_key_length  = _sig.details['length_public_key']
                    self.sec_key_length  = _sig.details['length_secret_key']
                    self.sig_length      = _sig.details['length_signature']
                self.use_pqc = True
                print(f"[SignatureManager] Dilithium2 available — "
                      f"pub={self.pub_key_length}B  "
                      f"sec={self.sec_key_length}B  "
                      f"sig={self.sig_length}B")
            except Exception as exc:
                print(f"[SignatureManager] Dilithium2 unavailable ({exc}); "
                      f"falling back to RSA-PSS.")
        if not self.use_pqc:
            self.sig_name = 'RSA-PSS-2048 (Fallback)'
            print(f"[SignatureManager] Using {self.sig_name}")

    # ------------------------------------------------------------------
    def generate_signing_keypair(self) -> tuple[bytes, bytes]:
        """Return (signing_public_key, signing_secret_key) as raw bytes."""
        if self.use_pqc:
            with oqs.Signature(self.SIG_ALGORITHM) as sig:
                pub = sig.generate_keypair()
                sec = sig.export_secret_key()
            return pub, sec
        else:
            # RSA-PSS fallback — reuse the serialisation from PQCManager
            sk  = rsa.generate_private_key(public_exponent=65537, key_size=2048)
            pub = sk.public_key().public_bytes(
                serialization.Encoding.PEM,
                serialization.PublicFormat.SubjectPublicKeyInfo,
            )
            sec = sk.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
            return pub, sec

    # ------------------------------------------------------------------
    def sign(self, data_bytes: bytes, secret_key: bytes) -> bytes:
        """
        Sign `data_bytes` with `secret_key`.
        Returns the raw signature bytes.
        """
        digest = hashlib.sha256(data_bytes).digest()
        if self.use_pqc:
            with oqs.Signature(self.SIG_ALGORITHM, secret_key) as sig:
                return sig.sign(digest)
        else:
            sk = serialization.load_pem_private_key(secret_key, password=None)
            return sk.sign(
                digest,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH,
                ),
                hashes.Prehashed(hashes.SHA256()),
            )

    # ------------------------------------------------------------------
    def verify(self, data_bytes: bytes, signature: bytes, public_key: bytes) -> bool:
        """
        Verify `signature` over `data_bytes` using `public_key`.
        Returns True if valid, False for any tampered / mismatched input.
        """
        digest = hashlib.sha256(data_bytes).digest()
        try:
            if self.use_pqc:
                with oqs.Signature(self.SIG_ALGORITHM) as sig:
                    return sig.verify(digest, signature, public_key)
            else:
                pk = serialization.load_pem_public_key(public_key)
                pk.verify(
                    signature,
                    digest,
                    padding.PSS(
                        mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.MAX_LENGTH,
                    ),
                    hashes.Prehashed(hashes.SHA256()),
                )
                return True
        except Exception:
            return False


# =========================================================================
# Exposed Module API
# =========================================================================
_default_manager   = PQCManager()
_signature_manager = SignatureManager()


# --- KEM (Key Encapsulation) API ---

def generate_keypair(algorithm: str = 'Kyber512') -> tuple[bytes, bytes]:
    """Generate a Kyber512 KEM keypair (or RSA fallback)."""
    if algorithm != _default_manager.algorithm and PQC_AVAILABLE:
        return PQCManager(algorithm=algorithm).generate_keypair()
    return _default_manager.generate_keypair()

def encrypt_data(data_bytes: bytes, public_key: bytes) -> bytes:
    """Hybrid-encrypt data with the recipient's KEM public key."""
    return _default_manager.encrypt_data(data_bytes, public_key)

def decrypt_data(ciphertext: bytes, secret_key: bytes) -> bytes:
    """Hybrid-decrypt a KEM-encrypted bundle."""
    return _default_manager.decrypt_data(ciphertext, secret_key)

def get_algorithm_info():
    """Print cryptographic backend details."""
    _default_manager.get_algorithm_info()


# --- Signature (Dilithium2) API ---

def generate_signing_keypair() -> tuple[bytes, bytes]:
    """
    Generate a Dilithium2 signing keypair (or RSA-PSS fallback).
    Returns (signing_public_key, signing_secret_key).
    """
    return _signature_manager.generate_signing_keypair()

def sign_weights(weights_bytes: bytes, private_key: bytes) -> bytes:
    """
    Sign a serialised weight payload with a Dilithium2 private key.

    Parameters
    ----------
    weights_bytes : raw bytes of the serialised model weights
    private_key   : Dilithium2 secret key returned by generate_signing_keypair()

    Returns
    -------
    signature : raw signature bytes to be transmitted alongside the payload
    """
    return _signature_manager.sign(weights_bytes, private_key)

def verify_weights(weights_bytes: bytes, signature: bytes, public_key: bytes) -> bool:
    """
    Verify a Dilithium2 signature over a weight payload.

    Parameters
    ----------
    weights_bytes : the exact same bytes that were signed on the client
    signature     : signature bytes returned by sign_weights()
    public_key    : Dilithium2 public key of the signing client

    Returns
    -------
    True  — payload is authentic and unmodified
    False — signature invalid, payload tampered, or wrong key
    """
    return _signature_manager.verify(weights_bytes, signature, public_key)


# =========================================================================
# Self-test
# =========================================================================

if __name__ == '__main__':
    get_algorithm_info()

    # ── KEM self-test ──────────────────────────────────────────────────
    print("\nRunning Hybrid-Encryption Self-Test...")
    pub, sec = generate_keypair()
    message   = b"PQC-IoT Sentinel: Secure FL Model Weights Transmission Payload!"
    print(f"Original: {message}")
    ciphertext = encrypt_data(message, pub)
    print(f"Encrypted bundle: {len(ciphertext)} bytes")
    decrypted = decrypt_data(ciphertext, sec)
    print(f"Decrypted: {decrypted}")
    assert message == decrypted, "[ERROR] Encryption/Decryption mismatch!"
    print("KEM self-test PASSED.")

    # ── Signature self-test ────────────────────────────────────────────
    print("\nRunning Dilithium2 Signature Self-Test...")
    sig_pub, sig_sec = generate_signing_keypair()
    payload  = b"FL weight bytes example payload"
    sig      = sign_weights(payload, sig_sec)
    print(f"Signature length: {len(sig)} bytes")

    ok = verify_weights(payload, sig, sig_pub)
    assert ok,  "[ERROR] Valid signature rejected!"
    print("Valid signature accepted — PASSED.")

    tampered = payload + b"X"
    nok = verify_weights(tampered, sig, sig_pub)
    assert not nok, "[ERROR] Tampered payload accepted!"
    print("Tampered payload rejected  — PASSED.")
    print("\nAll self-tests PASSED.")

