"""Physical and system-level constants."""

F_NOM_HZ: float = 60.0  # nominal grid frequency [Hz]
F_MIN_HZ: float = 40.0  # minimum valid frequency [Hz]
F_MAX_HZ: float = 80.0  # maximum valid frequency [Hz]
FS_DEFAULT_HZ: float = 10_000.0  # default sampling rate [Hz]
IEEE_FE_LIMIT_MHZ: float = 5.0  # IEEE C37.118 frequency error limit [mHz]
IEEE_RFE_LIMIT_HZS: float = 0.1  # IEEE C37.118 ROCOF error limit [Hz/s]
SCHEMA_VERSION: str = "v1"
