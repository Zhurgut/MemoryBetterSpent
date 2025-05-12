
import torch
from torch import nn 
import numpy as np
from numpy.random import rand

from torch.profiler import profile, ProfilerActivity, record_function, ProfilerAction, schedule

import csv
import os


import train
import layers
import main


current_state = "nothing"



def train_epoch(model, batch_size, opt, epoch):
    global current_state
    model.train(True)
    
    training_loader = torch.utils.data.DataLoader(train.training_datasets[np.random.randint(0, len(train.training_datasets))], batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    
    # loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    loss_fn = nn.CrossEntropyLoss()
    
    train_loss = 0.0
    
    with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=False,
    profile_memory=False,
    with_stack=False,
    schedule=schedule(wait=1, warmup=1, active=100, repeat=0)
    ) as prof:
        
        i = 0
        for X, labels in training_loader:

            with record_function("MEM CPY"):
                X, labels = X.to(train.device).float(), labels.to(train.device).float()
            
            with record_function("OPT_ZERO"):
                opt.zero_grad()
            
            with record_function("FORWARD"):
                logits = model(X)
                loss = loss_fn(logits, labels)
            
            with record_function("BACKWARD"):    
                loss.backward()        
            
            with record_function("OPT_STEP"):
                opt.step()
            
            prof.step()
            
            i += 1
            if i == 50:
                break
            
            # break
    
    
    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    events = prof.events()
    cpu_fw, cpu_bk, gpu_fw, gpu_bk = [], [], [], []
    
    for evt in events:
        k = evt.key
        if k == "FORWARD":
            if evt.cpu_time_total != 0:
                cpu_fw.append(evt.cpu_time_total)
            if evt.device_time_total != 0:
                gpu_fw.append(evt.device_time_total)
        elif k == "BACKWARD":
            if evt.cpu_time_total != 0:
                cpu_bk.append(evt.cpu_time_total)
            if evt.device_time_total != 0:
                gpu_bk.append(evt.device_time_total)
    
    cpu_frwd = np.median(cpu_fw)
    cpu_back = np.median(cpu_bk)
    cuda_frwd = np.median(gpu_fw)
    cuda_back = np.median(gpu_bk)
    
    return round(cpu_frwd, 3), round(cpu_back, 3), round(cuda_frwd, 3), round(cuda_back, 3)
    
    # events = prof.key_averages()
    # evt_map = {evt.key: evt for evt in events}
    
    
    # cpu_frwd = sum(e.cpu_time_total for e in events if e.key == "FORWARD")
    # cpu_back = sum(e.cpu_time_total for e in events if e.key == "BACKWARD")
    
    # cuda_frwd = sum(e.device_time_total for e in events if e.key == "FORWARD")
    # cuda_back = sum(e.device_time_total for e in events if e.key == "BACKWARD")
    
    # return round(cpu_frwd/1000), round(cpu_back/1000), round(cuda_frwd/1000), round(cuda_back/1000)

    


def train_and_profile(model, nr_epochs, batch_size, max_bs):

    _, test_labels = next(iter(torch.utils.data.DataLoader(train.cifar10_test, batch_size=10000)))
    test_labels = test_labels.to(train.device)
    
    _, train_labels = next(iter(torch.utils.data.DataLoader(train.cifar10_train, batch_size=50000)))
    train_labels = train_labels.to(train.device)
    
    
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=nr_epochs, eta_min=0)

    model.train(True)
    return train_epoch(model, batch_size, opt, 0)





def compare_dense_lrl(dim, nr_tf_blocks, nr_heads, rank, bs=128, max_bs=500):
    print("DENSE")
    dense = train_and_profile(main.VisionTransformer(dim, 4, nr_tf_blocks, nr_heads, 10, layers.Dense).to(train.device), 10, bs, max_bs)
    print("LOWRANKLIGHT")
    lrl = train_and_profile(main.VisionTransformer(dim, 4, nr_tf_blocks, nr_heads, 10, layers.LowRankLight, rank).to(train.device), 10, bs, max_bs)
    
    print("cpu forward: dense: ", dense[0], "us,  lowrank-light: ", lrl[0], "us")
    print("cpu backward: dense: ", dense[1], "us,  lowrank-light: ", lrl[1], "us")
    print("gpu forward: dense: ", dense[2], "us,  lowrank-light: ", lrl[2], "us")
    print("gpu backward: dense: ", dense[3], "us,  lowrank-light: ", lrl[3], "us")



def collect(dim, nr_tf_blocks, nr_heads, ranks, batch_sizes=[128, 96, 64, 32, 16], max_bs=500):
    
    results = []
    for bs in batch_sizes:
        
        print(f"Profiling batch size ", bs)
            
        model = main.VisionTransformer(dim, 4, nr_tf_blocks, nr_heads, 10, layers.Dense).to(train.device)
        # model = torch.compile(model)
        cpu_fw, cpu_bk, gpu_fw, gpu_bk = train_and_profile(model, nr_epochs=1, batch_size=bs, max_bs=bs)
        results.append(("dense", main.nr_parameters(model), bs, cpu_fw, cpu_bk, gpu_fw, gpu_bk))
        
        for rank in ranks:

            model = main.VisionTransformer(dim, 4, nr_tf_blocks, nr_heads, 10, layers.LowRankLight, rank).to(train.device)
            cpu_fw, cpu_bk, gpu_fw, gpu_bk = train_and_profile(model, nr_epochs=1, batch_size=bs, max_bs=bs)
            results.append(("lowrank_light", main.nr_parameters(model), bs, cpu_fw, cpu_bk, gpu_fw, gpu_bk))

    os.makedirs(os.path.join(os.path.dirname(__file__), "../../measurements/profiling"), exist_ok=True)
    path = os.path.join(os.path.dirname(__file__), "../../measurements/profiling", f"dense_vs_lrl_{dim}-{nr_tf_blocks}-{nr_heads}.csv")
    with open(path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        writer.writerow(["model", "nr_parameters", "batch_size", "cpu_forward_us", "cpu_backward_us", "gpu_forward_us", "gpu_backward_us"])

        for row in results:
            writer.writerow(row)


# collect(384, 6, 6, [32, 64, 96, 128, 192, 256])
collect(1024, 6, 8, [128, 256, 384, 512, 640, 768, 896])