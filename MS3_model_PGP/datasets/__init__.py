from .ptr_dataset import PTRDataset
from .pgp_dataset import PGPDataset

__all__ = {
    'ptr': PTRDataset,
    'pgp': PGPDataset,
}

def build_dataset(config,val=False):
    dataset = __all__[config.method.model_name](
        config=config, is_validation=val
    )
    return dataset
