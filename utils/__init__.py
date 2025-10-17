from .CSLDaily import CSLDailyDataset
from .CSLNews  import CSLNewsDataset

DATASET_REGISTRY = {
    "CSL_Daily": CSLDailyDataset,
    "CSL_News":  CSLNewsDataset,
}
