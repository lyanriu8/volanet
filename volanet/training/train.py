from datetime import datetime

from datahub.facade import get_history
from datahub.schemas import OHLCVQuery, DatahubFrame
from volanet.dataset.build_dataset import build_training_splits


from volanet.schemas import BuildConfig, LabelConfig, FeatureConfig, SplitDatasets, SequenceDataset




def train():
    dh_frame = get_history(OHLCVQuery(ticker="AAPL", start=datetime(2024, 1, 1,), end=datetime(2025,1,8), adjusted=True))
    
    print(dh_frame.df.info())
    print(dh_frame.spec)

def main() -> None: 
    train();
    
if __name__ == "__main__":
    main();