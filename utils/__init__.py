from datasets.CSLDaily import CSLDailyDataset
from datasets.CSLNews import CSLNewsDataset

DATASET_REGISTRY = {
    "CSL_Daily": CSLDailyDataset,
    "CSL_News":  CSLNewsDataset,
}
