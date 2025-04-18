import time

import torch
import torch.nn as nn

from bigptq import BRAGPTQ
from binary import Binarization
from modelutils import find_layers
import json
import os


downloads_dir = "./downloads"
def get_model(model_name):
    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    
    model_path = os.path.join(downloads_dir, f"DOWNLOAD_{model_name}")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)  # Ensure directories exist
    
    if os.path.exists(model_path):
        print(f"Loading pretrained model from {model_path}")
        model = torch.load(model_path)
    else:
        print(f"Downloading and saving model: {model_name}")
        if "opt" in model_name:
            from transformers import OPTForCausalLM
            model = OPTForCausalLM.from_pretrained(model_name, torch_dtype="auto",cache_dir=downloads_dir)
            model.seqlen = model.config.max_position_embeddings
        elif "llama" in model_name:
            from transformers import LlamaForCausalLM
            model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype="auto",cache_dir=downloads_dir)
            model.seqlen = 2048
        else:
            raise ValueError("Unsupported model type")
        
        #torch.save(model, model_path)
        print(f"Model saved to {model_path}")
    
    return model

'''
The function is employed to calibrate and quantize models layer by layer.
'''
@torch.no_grad()
def quant_sequential(model, dataloader, dev):
    print("Starting ...")

    for name, module in model.named_modules():
        module.global_name = args.model + name

    use_cache = model.config.use_cache
    model.config.use_cache = False

    if "opt" in args.model:
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(
            dev
        )
        if (
            hasattr(model.model.decoder, "project_out")
            and model.model.decoder.project_out
        ):
            model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
        if (
            hasattr(model.model.decoder, "project_in")
            and model.model.decoder.project_in
        ):
            model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
    elif "llama" in args.model:
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev # To capture attention mask corresponding to each sample for one layer?
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module): # Cache["i"] stores index of attention mask, and Cache["attention_mask"] stores attention mask itself
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp # inputs holds a key for each attention mask
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev)) # Pass first batch through the model
            # This should capture attention masks into inps
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    if "opt" in args.model:
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
        if (
            hasattr(model.model.decoder, "project_out")
            and model.model.decoder.project_out
        ):
            model.model.decoder.project_out = model.model.decoder.project_out.cpu()
        if (
            hasattr(model.model.decoder, "project_in")
            and model.model.decoder.project_in
        ):
            model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    elif "llama" in args.model:
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]

    print("Ready.")
    
    for i in range(len(layers)):

        layer = layers[i].to(dev)

        subset = find_layers(layer) # Just returns all modules in layer i?

        gptq = {}
        for name in subset:
            if (
                not (args.minlayer <= i < args.maxlayer and args.quant_only in name)
            ) == (not args.invert):
                continue
            braq_quantizer = Binarization(
                subset[name].weight,
                method=args.low_quant_method,
                groupsize=groupsize,
                corr_damp = args.corr_damp,
                lam = args.lam
            ) # Quantizer for each module of layer i
            gptq[name] = BRAGPTQ(
                subset[name],
                braq_quantizer,
                salient_metric=args.salient_metric,
                disable_gptq=args.disable_gptq,
            ) #?

        def add_batch(name):
            def tmp(_, inp, out):
                gptq[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in gptq:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        for h in handles:
            h.remove()

        for name in gptq:
            print(i, name)
            print("Quantizing ...")
            info = gptq[name].fasterquant(
                percdamp=args.percdamp, 
                blocksize=args.blocksize,
            )
            gptq[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        layers[i] = layer.cpu()
        del layer
        del gptq
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
'''
    opt braq       ptb : ppl37.62 :  
    opt robq       ptb : ppl31.48 :  
    opt mestrobq   ptb : ppl17.42 :  
    opt medianbraq ptb : ppl700   :
    opt orb        ptb : ppl6000  :
    opt whor       ptb : ppl1000  :
    opt arb   arb(0.5) : ppl500   :
    opt arb   arb(0.9) : ppl33.39 :
    opt arb   arb(0.8) : ppl45 :
    opt crb            : ppl17.32 :

    llama braq     ptb : ppl97 
    llama mestrobq ptb : ppl52.6
    llama crb      ptb : pp55

    
    opt braq  wikitext : ppl41
    opt crb   wikitext : ppl12
    
    llama braq     wikitext  : pp18

    -- above measures used incorrect crb --

    opt1.3B braq ptb                        : ppl 73.81
    opt1.3B crb  ptb                        : ppl 87.83
    opt1.3B crb_stable  ptb                 : ppl 82
    opt1.3B crb_stable_v2  ptb              : ppl 81
    opt1.3B crb_stable_v3  ptb              : ppl 75
    opt1.3B crb_stable_v4  ptb              : ppl 73.28
    opt1.3B crb_stable_v4 cordamp0.2 ptb    : ppl 83 
    opt1.3B crb_stable_v5           ptb     : ppl 65.59 [!]
    opt1.3B crb_stable_v6           ptb     : ppl 63.11 [!]

    opt1.3B braq wikitext2                  : ppl 61.275
    opt1.3B crb  wikitext2                  : ppl 50.70
    opt1.3B crb_stable_v6  wikitext2        : ppl 53.13 [!]
    
    opt2.7B crb  wikitext2                  : ppl 71.49

    opt2.7B braq wikitext2                  : ppl 61.275 ?
    opt2.7B crb  wikitext2                  : ppl 44
    opt2.7B crb_stable_v6    wikitext       : ppl 67 [-]
    opt2.7B crb_stable_v7    wikitext       : ppl 47.34


    opt6.7B braq ptb                        : ppl 35 
    opt6.7B crbv6 ptb                       : ppl 35 [-]
    opt6.7B crbv7 ptb                       : ppl 34.9

    opt6.7b braq             wikitext       : ppl35.84
    opt6.7b crb_stable_v6    wikitext       : ppl36.429  [-]
    opt6.7b crb_stable_v7    wikitext       : ppl

'''

if __name__ == "__main__":
    import argparse
    from datautils import *

    def list_of_ints(arg):
        return list(map(int, arg.split(',')))
    
    def list_of_floats(arg):
        return list(map(float, arg.split(',')))

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model", type=str, help="model to load; for example `huggyllama/llama-7b`."
    )
    parser.add_argument(
        "dataset",
        type=str,
        choices=["wikitext2", "ptb", "c4"],
        help="Where to extract calibration data from.",
    )
    parser.add_argument(
        "low_quant_method",
        type=str,
        choices=['fp16','rtn',"xnor", "sign", "no", "2bit", "4bit", "prune", "braq",'robq','mestrobq','medianbraq','orb','whor','arb','bhor','jrb','crb','odr','new','ahor','crbv8','crbv9','crbv10','crbog'],
        help="quantization method; `xnor` is the method using XNOR to adapt hardware calculation; `prune` is the method used in sparseGPTQ; braq is the method used in BiLLM",
    )
    parser.add_argument("--load_quantized", action="store_true")
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for sampling the calibration data."
    )
    parser.add_argument(
        "--nsamples", type=int, default=128, help="Number of calibration data samples."
    )
    parser.add_argument(
        "--percdamp",
        type=float,
        default=0.01,
        help="Percent of the average Hessian diagonal to use for dampening.",
    )
    parser.add_argument(
        "--blocksize",
        type=int,
        default=128,
        help="Blocksize to use for adaptive mask selection.",
    )
    parser.add_argument(
        "--salient_metric",
        type=str,
        default="magnitude",
        choices=["magnitude", "hessian"],
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="set the device to use for quantization.",
    )
    parser.add_argument(
        "--disable_gptq",
        action="store_true",
        help="disable GPTQ for quantization.",
    )
    parser.add_argument(
        "--minlayer", type=int, default=-1, help="Quant all layers with id >= this."
    )
    parser.add_argument(
        "--maxlayer", type=int, default=1000, help="Quant all layers with id < this."
    )
    parser.add_argument(
        "--quant_only",
        type=str,
        default="",
        help="Quant only layers that contain this text.",
    )
    parser.add_argument("--invert", action="store_true", help="Invert subset.")
    parser.add_argument(
        "--save",
        action="store_true",
    )
    parser.add_argument(
        "--log_wandb", action="store_true", help="Whether to log to wandb."
    )
    parser.add_argument(
        "--just_download", action="store_true"
    )
    
    parser.add_argument(
        "--corr_damp", type = float, default = 0.1
    )
    parser.add_argument(
        "--lam", type = float, default = 1e-5
    )
    parser.add_argument(
        "--skip_ppl_save",
        action="store_true"
    )
    args = parser.parse_args()
    groupsize = args.blocksize

    device = args.device
    save_title = f"{args.model}_{args.dataset}_{args.low_quant_method}_{groupsize}_{args.salient_metric}"
    save_file = "./output/" + save_title.replace("/", "_") + ".pt"
    if args.load_quantized:
        model = get_model(save_file) # 1 : Get Model
        model.eval()
    else: # braq
        model = get_model(args.model)
        model.eval()
        tick = time.time()

        dataloader, testloader = get_loaders(
            args.dataset,
            nsamples=args.nsamples,
            seed=args.seed,
            model=args.model,
            seqlen=model.seqlen,
        )

        if(args.just_download):
            print(f"Just download flag set, exiting")
            exit()
        #quant_sequential(model, dataloader, device)
        print("quantization time:", time.time() - tick, "s")


    '''
    if args.save:
        save_path = os.path.dirname(save_file)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        model.save_pretrained(save_file)
    '''


    for dataset in [args.dataset]:#["wikitext2", "ptb", "c4"]:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, seqlen=model.seqlen, model=args.model
        )
        print(dataset)
        if "opt" in args.model:
            from eval_ppl_utils import opt_eval
            
            ppl = opt_eval(model, testloader, device, dataset, args.log_wandb, save_title, save = not args.skip_ppl_save )
           
            ''' FOR ABLATION STUDY '''
            # Define the path to the JSON file
            results_path = "./output/ablation_results.json"

            # Ensure the results directory exists
            os.makedirs(os.path.dirname(results_path), exist_ok=True)

            # Load existing results or initialize an empty dict if file doesn't exist or is empty
            try:
                with open(results_path, "r") as f:
                    results = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                results = {}

            # Create the key and update the results with the new perplexity value
            key = f"{args.model}_{args.dataset}_{args.low_quant_method}_{groupsize}_{args.salient_metric}_{args.corr_damp}_{args.lam}"
            results[key] = ppl

            # Save the updated results back to the JSON file
            with open(results_path, "w") as f:
                json.dump(results, f, indent=4)


        elif "llama" in args.model:
            from eval_ppl_utils import llama_eval

            llama_eval(model, testloader, device, dataset, args.log_wandb, save_title)
