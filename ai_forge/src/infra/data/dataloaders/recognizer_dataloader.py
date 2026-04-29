import torch

from src.domain.ports.data.dataset_port import DatasetPort
from src.domain.ports.data.dataloader_port import BaseCTCLabelEncoder
from src.infra.data.dataloaders.ctc_collate import build_ctc_collate_fn

class RecognizerDataLoader(torch.utils.data.DataLoader):
    def __init__(
        self,
        dataset: DatasetPort,
        encoder: BaseCTCLabelEncoder,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = False,
        drop_last: bool = False,
        persistent_workers: bool = False,
    ):
        if not isinstance(dataset, torch.utils.data.Dataset):
            raise ValueError("RecognizerDataLoader.__init__: dataset must implement DatasetPort")
        if not isinstance(encoder, BaseCTCLabelEncoder):
            raise ValueError("RecognizerDataLoader.__init__: encoder must implement BaseCTCLabelEncoder")
        if batch_size <= 0:
            raise ValueError("RecognizerDataLoader.__init__: batch_size must be > 0")
        if num_workers < 0:
            raise ValueError("RecognizerDataLoader.__init__: num_workers must be >= 0")
        if not isinstance(shuffle, bool):
            raise ValueError("RecognizerDataLoader.__init__: shuffle must be a boolean")
        if not isinstance(pin_memory, bool):
            raise ValueError("RecognizerDataLoader.__init__: pin_memory must be a boolean")
        if not isinstance(drop_last, bool):
            raise ValueError("RecognizerDataLoader.__init__: drop_last must be a boolean")
        if not isinstance(persistent_workers, bool):
            raise ValueError("RecognizerDataLoader.__init__: persistent_workers must be a boolean")
        if num_workers > 0 and persistent_workers and not torch.utils.data.get_worker_info():
            raise ValueError("RecognizerDataLoader.__init__: persistent_workers=True requires num_workers > 0")
        if num_workers == 0 and persistent_workers:
            raise ValueError("RecognizerDataLoader.__init__: persistent_workers=True requires num_workers > 0")


        collate_fn = build_ctc_collate_fn(encoder)
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin_memory, drop_last=drop_last, persistent_workers=persistent_workers)