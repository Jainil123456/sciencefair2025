"""
Seed Manager for AgriGraph AI - Dynamic random seed generation.

Provides three modes for seed generation:
- auto: Uses timestamp + process ID for unique seeds (different data each run)
- fixed: Returns constant seed 42 for reproducibility
- custom: Uses user-provided seed value

This enables flexible control over randomization in the data generation pipeline.
"""

import hashlib
import os
import time
from typing import Dict, Optional


class SeedManager:
    """
    Manages random seed generation and storage for reproducibility and uniqueness.

    Supports three modes:
    - 'auto': Generate unique seed from timestamp and process ID (non-reproducible)
    - 'fixed': Return constant seed 42 (reproducible)
    - 'custom': Return user-provided custom seed (reproducible with custom value)
    """

    # Valid seed range for NumPy and PyTorch (0 to 2^31 - 1)
    SEED_MAX = 2**31 - 1
    FIXED_SEED = 42

    def __init__(self) -> None:
        """Initialize the SeedManager with empty session storage."""
        self._session_seeds: Dict[str, int] = {}
        self._last_auto_seed: Optional[int] = None

    def generate_seed(
        self,
        mode: str = 'auto',
        custom_seed: Optional[int] = None
    ) -> int:
        """
        Generate a random seed based on the specified mode.

        Args:
            mode: Seed generation mode ('auto', 'fixed', or 'custom')
            custom_seed: Seed value to use when mode='custom'

        Returns:
            Integer seed in valid range (0 to 2^31 - 1)

        Raises:
            ValueError: If mode is invalid or custom_seed is invalid
            TypeError: If parameters have incorrect types
        """
        if not isinstance(mode, str):
            raise TypeError(f"mode must be str, not {type(mode).__name__}")

        mode = mode.lower()

        if mode == 'auto':
            return self._generate_auto_seed()
        elif mode == 'fixed':
            return self.FIXED_SEED
        elif mode == 'custom':
            if custom_seed is None:
                raise ValueError(
                    "custom_seed must be provided when mode='custom'"
                )
            return self._validate_seed(custom_seed)
        else:
            raise ValueError(
                f"Invalid mode '{mode}'. Must be 'auto', 'fixed', or 'custom'"
            )

    def _generate_auto_seed(self) -> int:
        """
        Generate a unique seed from timestamp and process ID.

        Uses MD5 hash of (timestamp in microseconds + process ID) to ensure
        consistent integer output in valid range.

        Returns:
            Integer seed in valid range (0 to 2^31 - 1)
        """
        # Get current time in microseconds for high precision
        timestamp_us = int(time.time() * 1_000_000)

        # Get process ID
        process_id = os.getpid()

        # Combine timestamp and process ID
        seed_material = f"{timestamp_us}_{process_id}".encode('utf-8')

        # Hash the material using MD5
        hash_digest = hashlib.md5(seed_material).hexdigest()

        # Convert hex to integer and constrain to valid range
        seed_int = int(hash_digest, 16) % (self.SEED_MAX + 1)

        self._last_auto_seed = seed_int
        return seed_int

    def _validate_seed(self, seed: int) -> int:
        """
        Validate that a seed is an integer in the valid range.

        Args:
            seed: Seed value to validate

        Returns:
            The validated seed

        Raises:
            TypeError: If seed is not an integer
            ValueError: If seed is outside valid range
        """
        if not isinstance(seed, int):
            raise TypeError(f"Seed must be int, not {type(seed).__name__}")

        if seed < 0 or seed > self.SEED_MAX:
            raise ValueError(
                f"Seed must be in range [0, {self.SEED_MAX}], got {seed}"
            )

        return seed

    def store_session_seed(self, session_id: str, seed: int) -> None:
        """
        Store a seed value associated with a session ID.

        Args:
            session_id: Unique identifier for the session
            seed: Seed value to store

        Raises:
            TypeError: If session_id is not a string or seed is not an integer
            ValueError: If seed is outside valid range
        """
        if not isinstance(session_id, str):
            raise TypeError(
                f"session_id must be str, not {type(session_id).__name__}"
            )

        if not session_id.strip():
            raise ValueError("session_id cannot be empty")

        # Validate the seed before storing
        validated_seed = self._validate_seed(seed)

        self._session_seeds[session_id] = validated_seed

    def get_session_seed(self, session_id: str) -> Optional[int]:
        """
        Retrieve a seed value for a given session ID.

        Args:
            session_id: Unique identifier for the session

        Returns:
            The stored seed value, or None if no seed exists for this session

        Raises:
            TypeError: If session_id is not a string
        """
        if not isinstance(session_id, str):
            raise TypeError(
                f"session_id must be str, not {type(session_id).__name__}"
            )

        return self._session_seeds.get(session_id)

    def clear_session_seed(self, session_id: str) -> bool:
        """
        Remove a seed value for a given session ID.

        Args:
            session_id: Unique identifier for the session

        Returns:
            True if seed was removed, False if session_id didn't exist

        Raises:
            TypeError: If session_id is not a string
        """
        if not isinstance(session_id, str):
            raise TypeError(
                f"session_id must be str, not {type(session_id).__name__}"
            )

        if session_id in self._session_seeds:
            del self._session_seeds[session_id]
            return True
        return False

    def clear_all_sessions(self) -> None:
        """Clear all stored session seeds."""
        self._session_seeds.clear()

    def get_last_auto_seed(self) -> Optional[int]:
        """
        Get the last auto-generated seed.

        Returns:
            The last auto-generated seed, or None if no auto seed has been generated
        """
        return self._last_auto_seed

    @property
    def session_count(self) -> int:
        """Get the number of active sessions with stored seeds."""
        return len(self._session_seeds)


# Global singleton instance
seed_manager = SeedManager()
