import torch
from torch import nn
from train import train

import layers
import models
import datasets

import argparse
import json

import time


def nr_parameters(model):
    if isinstance(model, layers.MaskedSparse):
        return (model.mask.sum() + model.bias.numel()).item()
    
    return sum(p.numel() for p in model.parameters())



LAYERS = {
    "dense": layers.Dense,
    "lowrank": layers.LowRank,
    "lowranklight": layers.LowRankLight,
    "monarch": layers.Monarch,
    "kron": layers.Kronecker,
    "kronecker": layers.Kronecker,
    "tt": layers.TT,
    "btt": layers.BTT,
}


# + variants only work with sparse layers that support non-square dimensions
MODELS = {
    "mlp": models.MLP_DE,
    "mlp+": models.MLP,
    "b-mlp": models.B_MLP_noIB,
    "b-mlp+": models.B_MLP,
    "vit": models.VisionTransformer_noIB,
    "vit+": models.VisionTransformer
}

DATASETS = {
    "cifar10": datasets.cifar10
}
    

def main():
    parser = argparse.ArgumentParser(description="Build and train a model")
    
    parser.add_argument("--layer", "-l", type=str, required=True, 
                        choices=LAYERS.keys(),
                        help="Which layer type to use")
    
    parser.add_argument("--width", "-w", type=int, required=True,
                        help="Width of the model")
    
    parser.add_argument("--depth", "-d", type=int, required=True,
                        help="number of hidden layers")
    
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-3,
                        help="learning rate for the optimizer, (default 1e-3)")
    
    parser.add_argument("--batch_size", "-bs", type=int, default=1000,
                        help="batch size to use for training, (default 1000)")
    
    parser.add_argument("--epochs", "-e", type=int, default=200,
                        help="the maximum nr of epochs of training, (default 200)")
    
    parser.add_argument("--params", "-p", nargs="*", type=str,
                        help="Additional layer-specific parameters as key=value pairs (e.g. for -m=tt, --params rank=3 nr_cores=2)")
    
    parser.add_argument("--weight_decay", "-wd", type=float, default=0.0,
                        help="weight decay, default 0.0")
    
    parser.add_argument("--init_scale", "-s", type=float, default=1.0,
                        help="modify initialization scale for the weights, default 1.0")
    
    parser.add_argument("--max_bs", type=int, default=50000,
                        help="maximum batchsize the model can cope with, for evaluating training and test accuracy. Must lower for ViT")
    
    args = parser.parse_args()
    
    layers.set_scale(args.init_scale)
    
    layer_fn = None
    vit = False
    
    if "vit" in args.layer:
        vit = True
        
    layer_fn = LAYERS[args.layer]
    
    params = {p.split("=")[0]: int(p.split("=")[1]) for p in args.params} if args.params else {}

    if vit:
        if layer_fn is layers.LowRank or layer_fn is layers.LowRankLight:
            model = VisionTransformer(args.width, params["patch_size"], args.depth, params["nr_heads"], 10, layer_fn, params["rank"])  
        elif layer_fn is layers.Monarch:
            model = VisionTransformer(args.width, params["patch_size"], args.depth, params["nr_heads"], 10, layer_fn, params["nr_blocks"])
        elif layer_fn is layers.TT or layer_fn is layers.BTT:
            model = VisionTransformer(args.width, params["patch_size"], args.depth, params["nr_heads"], 10, layer_fn, params["nr_cores"], params["rank"])
        else:
            model = VisionTransformer(args.width, params["patch_size"], args.depth, params["nr_heads"], 10, layer_fn)
        
        model(torch.rand(2, 3, 32, 32))
    else:
        if layer_fn is layers.LowRank or layer_fn is layers.LowRankLight:
            model = create_model(args.depth, args.width, layer_fn, params["rank"])  
        elif layer_fn is layers.Monarch:
            model = create_model(args.depth, args.width, layer_fn, params["nr_blocks"])
        elif layer_fn is layers.TT or layer_fn is layers.BTT:
            model = create_model(args.depth, args.width, layer_fn, params["nr_cores"], params["rank"])
        else:
            model = create_model(args.depth, args.width, layer_fn)      

    start = time.time()
    training_losses, training_accuracies, test_losses, test_accuracies = train(model, args.epochs, args.learning_rate, args.batch_size, args.weight_decay, args.max_bs)
    end = time.time()
    
    output = {
        "layer": str(layer_fn.__name__),
        "width": args.width,
        "depth": args.depth,
        "nr_parameters": nr_parameters(model),
        "lr": args.learning_rate,
        "batchsize": args.batch_size,
        "nr_epochs": len(training_losses),
        "scale": args.init_scale,
        "time": end-start,
        "params": params,
        "train_losses": training_losses,
        "test_losses": test_losses,
        "train_accuracies": training_accuracies,
        "test_accuracies": test_accuracies
    }
    
    print(json.dumps(output, indent=4))
    


if __name__ == "__main__":
    main()