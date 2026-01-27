from datetime import datetime
from typing import Literal

from datahub.facade import get_history
from datahub.schemas import OHLCVQuery, DatahubFrame
from volanet.dataset.build_dataset import build_training_splits
from volanet.schemas import BuildConfig, LabelConfig, FeatureConfig, DatasetConfig, SplitDatasets, SequenceDataset

def train():
    dh_frame = get_history(OHLCVQuery(ticker="AAPL", start=datetime(2024, 1, 1,), end=datetime(2025,1,8), adjusted=True))
    
    f_cfg = FeatureConfig()
    l_cfg = LabelConfig()
    d_cfg = DatasetConfig()   
    
    cfg = BuildConfig(ticker="AAPL")
    
    training_datasets = build_training_splits(canonical_ohlcv=dh_frame.df, cfg=cfg)
    
    print(dh_frame.df.info())
    print(dh_frame.spec)
    print(training_datasets.train.X)
    print(training_datasets.train.y)

def main() -> None: 
    train();
    
if __name__ == "__main__":
    main();