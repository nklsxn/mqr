from dataclasses import dataclass

@dataclass
class Defaults:
    sample_value_fmt: str = '#.5g'
    capability_value_fmt: str = '#.3g'
