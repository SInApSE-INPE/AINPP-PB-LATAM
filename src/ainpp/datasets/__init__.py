from .gsmap import AINPPPBLATAMDataset

# Backwards compatibility: older scripts reference NowcastingDataset.
# Keep a thin alias so legacy entry points continue to work.
NowcastingDataset = AINPPPBLATAMDataset
