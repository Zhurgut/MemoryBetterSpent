import torch
from torch import nn
from train import train

import layers
import models
import dataset_loader

import argparse
import json

import time


def nr_parameters(model, seen=None):

    total = 0

    if seen is None:
        seen = set()

    if isinstance(model, layers.MaskedSparse):
        total += (model.mask.sum() + model.bias.numel()).item()
    else:
        for p in model.parameters(recurse=False):
            ptr = p.data_ptr()
            if ptr not in seen:
                seen.add(ptr)
                total += p.numel()

    for child in model.children():
        total += nr_parameters(child, seen)
    
    return total



LAYERS = {
    "dense": layers.Dense,
    "lowrank": layers.LowRank,
    "lowranklight": layers.LowRankLight,
    "monarch": layers.Monarch,
    "kron": layers.Kronecker,
    "kronecker": layers.Kronecker,
    "tt": layers.TT,
    "btt": layers.BTT2,
    "blast": layers.Blast,
    "unstructured": layers.Unstructured
}


# (2) variants only work with sparse layers that support non-square dimensions
MODELS = {
    "mlp": models.MLP_DE,
    "mlp2": models.MLP,
    "b_mlp": models.B_MLP_noIB,
    "b_mlp2": models.B_MLP,
    "vit": models.VisionTransformer_noIB,
    "vit2": models.VisionTransformer,
    "gpt2": models.GPT2,
    "distil_gpt2": models.distilGPT2
}

DATASETS = {
    "cifar10": dataset_loader.cifar10,
    "tiny_imagenet": dataset_loader.tiny_imagenet,
    "wikitext2": dataset_loader.wikitext2
}
    

def main():
    parser = argparse.ArgumentParser(description="Build and train a model")
    
    parser.add_argument("--layer", "-l", type=str, required=True, 
                        choices=LAYERS.keys(),
                        help="Which layer type to use")

    parser.add_argument("--model", "-m", type=str, required=True, 
                        choices=MODELS.keys(),
                        help="Which model architecture to use")
    
    parser.add_argument("--dataset", "-ds", type=str, default="cifar10", 
                        choices=DATASETS.keys(),
                        help="Which dataset to use for training (default cifar10)")
    
    parser.add_argument("--width", "-w", type=int, required=True,
                        help="Width of the model")
    
    parser.add_argument("--depth", "-d", type=int, required=True,
                        help="number of hidden layers, or number of (transformer-) blocks")
    
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-3,
                        help="learning rate for the optimizer, (default 1e-3)")
    
    parser.add_argument("--batch_size", "-bs", type=int, default=1000,
                        help="batch size to use for training, (default 1000)")
    
    parser.add_argument("--epochs", "-e", type=int, default=200,
                        help="the maximum nr of epochs of training, (default 200)")

    parser.add_argument("--weight_decay", "-wd", type=float, default=0.0,
                        help="AdamW weight decay, default 0.0")
    
    parser.add_argument("--lr_decay", type=int, default=1, 
                        choices=[0,1],
                        help="whether to use a cosine schedule to decay to learning rate, (default 1, i.e. true)")
    
    parser.add_argument("--early_stopping", type=int, default=1,
                        choices=[0,1],
                        help="enable early stopping if the program determines that not much can be gained from further training, (default 1, i.e. true)")
    
    parser.add_argument("--init_scale", "-s", type=float, default=1.0,
                        help="modify initialization scale for the weights, default 1.0")
    
    parser.add_argument("--max_bs", type=int, default=50000,
                        help="maximum batchsize the model can cope with for evaluating training and test accuracy. Must lower for ViT")
    
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="probability for dropout layers, default 0.2")
    
        
    parser.add_argument("--params", "-p", nargs="*", type=str,
                        help="""Additional model-specific parameters as key=value pairs (e.g. for -m=tt do: --params rank=3 nr_cores=2)
                        For ViT, pass 'patch_size' and 'nr_heads'""")
    
    
    args = parser.parse_args()
    
    layers.set_scale(args.init_scale)
    layer_fn = LAYERS[args.layer]
    model_fn = MODELS[args.model]
    dataset = DATASETS[args.dataset](args.batch_size, args.max_bs)
    in_dim, out_dim, image_dim, _, _, _, _, _, _, _ = dataset
    # in_dim, out_dim, image_dim = 64*64*3, 200, 64

    p = args.dropout
    
    
    params = {p.split("=")[0]: int(p.split("=")[1]) for p in args.params} if args.params else {}
    
    model = None
    is_gpt2 = False
    if args.model == "mlp" or args.model == "mlp2" or args.model == "b_mlp" or args.model == "b_mlp2":
        
        if layer_fn is layers.LowRank or layer_fn is layers.LowRankLight:
            model = model_fn(in_dim, args.depth, args.width, out_dim, layer_fn, params["rank"], p=p)  
        elif layer_fn is layers.Monarch:
            model = model_fn(in_dim, args.depth, args.width, out_dim, layer_fn, params["nr_blocks"], p=p)
        elif layer_fn is layers.BTT2:
            model = model_fn(in_dim, args.depth, args.width, out_dim, layer_fn, params["rank"], p=p)
        elif layer_fn is layers.TT:
            model = model_fn(in_dim, args.depth, args.width, out_dim, layer_fn, params["nr_cores"], params["rank"], p=p)
        elif layer_fn is layers.Blast:
            model = model_fn(in_dim, args.depth, args.width, out_dim, layer_fn, params["block_size"], params["rank"], p=p)
        else:
            model = model_fn(in_dim, args.depth, args.width, out_dim, layer_fn, p=p)

    elif args.model == "vit" or args.model == "vit2":

        if layer_fn is layers.LowRank or layer_fn is layers.LowRankLight:
            model = model_fn(args.width, image_dim, params["patch_size"], args.depth, params["nr_heads"], out_dim, layer_fn, params["rank"], dropout_p=p)  
        elif layer_fn is layers.Monarch:
            model = model_fn(args.width, image_dim, params["patch_size"], args.depth, params["nr_heads"], out_dim, layer_fn, params["nr_blocks"], dropout_p=p)
        elif layer_fn is layers.BTT2:
            model = model_fn(args.width, image_dim, params["patch_size"], args.depth, params["nr_heads"], out_dim, layer_fn, params["rank"], dropout_p=p)  
        elif layer_fn is layers.TT:
            model = model_fn(args.width, image_dim, params["patch_size"], args.depth, params["nr_heads"], out_dim, layer_fn, params["nr_cores"], params["rank"], dropout_p=p)
        elif layer_fn is layers.Blast:
            model = model_fn(args.width, image_dim, params["patch_size"], args.depth, params["nr_heads"], out_dim, layer_fn, params["block_size"], params["rank"], dropout_p=p)
        else:
            model = model_fn(args.width, image_dim, params["patch_size"], args.depth, params["nr_heads"], out_dim, layer_fn, dropout_p=p)
         
    elif args.model == "gpt2" or args.model == "distil_gpt2":
        

        if layer_fn is layers.LowRank or layer_fn is layers.LowRankLight:
            model = model_fn(layer_fn, params["rank"])  
        elif layer_fn is layers.Monarch:
            model = model_fn(layer_fn, params["nr_blocks"])
        elif layer_fn is layers.BTT2:
            model = model_fn(layer_fn, params["rank"])  
        elif layer_fn is layers.TT:
            model = model_fn(layer_fn, params["nr_cores"], params["rank"])
        elif layer_fn is layers.Blast:
            model = model_fn(layer_fn, params["block_size"], params["rank"])
        elif layer_fn is layers.Unstructured:
            model = model_fn(layer_fn, params["density"])
        else:
            model = model_fn(layer_fn)
        
        is_gpt2 = True
    


    # with torch.no_grad():
    #     model(torch.rand(1, 3, 64, 64))

    # print(nr_parameters(model))
    # exit()


    training_losses, training_accuracies, test_losses, test_accuracies, times = train(
        model, dataset, args.epochs, args.learning_rate, args.weight_decay, bool(args.early_stopping), bool(args.lr_decay), is_gpt2=is_gpt2
    )

    times = [t - times[0] for t in times]
    
    output = {
        "layer": str(layer_fn.__name__),
        "width": args.width,
        "depth": args.depth,
        "nr_parameters": nr_parameters(model),
        "lr": args.learning_rate,
        "batchsize": args.batch_size,
        "nr_epochs": len(training_losses),
        "scale": args.init_scale,
        "time": times,
        "params": params,
        "train_losses": training_losses,
        "test_losses": test_losses,
        "train_accuracies": training_accuracies,
        "test_accuracies": test_accuracies
    }
    
    print(json.dumps(output, indent=4))
    


if __name__ == "__main__":
    main()