import torch
from data_manager.dataset.absa import ABSADataset


# Redistribution data by worker to prevent every worker from loading same data
def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()

    dataset = worker_info.dataset
    worker_id = worker_info.id
    split_size = len(dataset.data) // worker_info.num_workers

    dataset.data = dataset.data[worker_id * split_size: (worker_id + 1) * split_size]


# DataLoader setting
def set_loader(fp, config, meta_data, batch_size, extension="csv"):
    dataset = ABSADataset(fp=fp, config=config, extension=extension, enc_aspect=meta_data["enc_aspect"],
                          enc_aspect2=meta_data["enc_aspect2"], enc_sentiment=meta_data["enc_sentiment"],enc_sentiment_score=meta_data["enc_sentiment_score"],
                          enc_aspect_score=meta_data["enc_aspect_score"] 
                          , batch_size=batch_size)

    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              num_workers=0,
                                              collate_fn=lambda x: x,
                                              worker_init_fn=worker_init_fn)

    return data_loader



