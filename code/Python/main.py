import torch
from torch import nn
from train import train
from layers import *

import argparse
import json


def nr_parameters(model):
    return sum(p.numel() for p in model.parameters())


def create_model(nr_blocks, width, layer_fn, *args):
    # def Block():
    #     return SkipConnection(nn.Sequential(
    #         nn.LayerNorm((width,)), 
    #         # layer_fn(width, *args), 
    #         nn.Linear(width, width),
    #         nn.ReLU(),
    #         # layer_fn(width, *args), 
    #         nn.Linear(width, width)
    #     ))
    def Block():
        return nn.Sequential(
            nn.LayerNorm((width,)),
            layer_fn(width, *args), 
            nn.ReLU()
        )
    
    encoder = nn.Sequential(
        nn.Flatten(), 
        nn.LazyLinear(width), 
        nn.ReLU()
         
        # nn.Conv2d(3, 32, (4, 4), padding=1, stride=2), 
        # nn.MaxPool2d((2, 2)),
        # nn.ReLU(), 
        # nn.Conv2d(32, 32, (4, 4), padding=1, stride=2), 
        # # nn.MaxPool2d((2, 2)),
        
        # nn.ReLU(), 
        # # nn.Conv2d(32, 32, (3, 3)),
        # # nn.ReLU(),
        # # nn.Conv2d(32, 32, (3, 3)),
        # # nn.MaxPool2d((2, 2))
    )
    model = nn.Sequential(
        encoder, 
        *[Block() for b in range(nr_blocks)], 
        nn.Linear(width, 10)
    )
    model(torch.rand(1, 3, 32, 32)) # dummy input to initialize the lazy layer(s), make sure everything works
    
    # print(nr_parameters(model), " parameters")
    
    return model


LAYERS = {
    "dense": Dense,
    "lowrank": LowRank,
    "monarch": Monarch,
    "kron": Kronecker,
    "kronecker": Kronecker,
    "tt": TT
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
    
    parser.add_argument("--batch_size", "-bs", type=int, default=250,
                        help="batch size to use for training, (default 250)")
    
    parser.add_argument("--epochs", "-e", type=int, default=100,
                        help="the maximum nr of epochs of training, (default 100)")
    
    parser.add_argument("--params", "-p", nargs="*", type=str,
                        help="Additional layer-specific parameters as key=value pairs (e.g. for -m=tt, --params rank=3 nr_cores=2)")
    
    args = parser.parse_args()
    
    layer_fn = LAYERS[args.layer]
    
    params = {p.split("=")[0]: int(p.split("=")[1]) for p in args.params} if args.params else {}

    
    if layer_fn is LowRank:
        model = create_model(args.depth, args.width, layer_fn, params["rank"])  
    elif layer_fn is Monarch:
        model = create_model(args.depth, args.width, layer_fn, params["nr_blocks"])
    elif layer_fn is TT:
        model = create_model(args.depth, args.width, layer_fn, params["nr_cores"], params["rank"])
    else:
        model = create_model(args.depth, args.width, layer_fn)      

    training_losses, training_accuracies, test_losses, test_accuracies = train(model, args.epochs, args.learning_rate, args.batch_size)
    
    output = {
        "layer": str(layer_fn.__name__),
        "width": args.width,
        "depth": args.depth,
        "nr_parameters": nr_parameters(model),
        "lr": args.learning_rate,
        "batchsize": args.batch_size,
        "nr_epochs": len(training_losses),
        "params": params,
        "train_losses": training_losses,
        "test_losses": test_losses,
        "train_accuracies": training_accuracies,
        "test_accuracies": test_accuracies
    }
    
    print(json.dumps(output, indent=4))
    
    # model = create_model(args.depth, args.width, layer_fn)

# model = create_model(1, 256, Dense)

# start = time.time()
# train(model, 100, 1e-4, 250)
# end = time.time()

# print(end-start)



# model = create_model(3, 64, LowRank, 5)
# train_epoch(model)

# model = create_model(3, 64, Monarch, 8)
# train_epoch(model)

# model = create_model(3, 64, TT, 2, 2)
# train_epoch(model)

# model = create_model(3, 64, Kronecker)
# train_epoch(model)



if __name__ == "__main__":
    main()