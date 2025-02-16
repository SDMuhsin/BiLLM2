@torch.no_grad()
def quant_sequential(model, dataloader, dev):
    """
    Orchestrates the sequential (layer-by-layer) quantization of the model using GPTQ-like strategies.
    This function:
      1) Intercepts the forward pass of the first layer to gather calibration/statistics data.
      2) Iterates over each layer in the model:
         - Applies the GPTQ-based binarization/quantization method to that layer's weights,
           focusing on either 'salient' or 'non-salient' partitions (as specified in the Binarization object).
         - Re-injects the quantized layer to verify and propagate outputs for the next layer.
      3) Ensures that the binarization steps align with the high-level theory:
         - Salient partition → 2-bit expansion (residual binarization).
         - Non-salient partition → possibly 1-bit binarization (split into sparse/concentrated).
    The function uses caching and hooking to minimize memory usage and gather precise per-layer input data.

    Args:
        model: The HuggingFace/transformer model to be quantized.
        dataloader: A DataLoader providing sample inputs for calibration/stats gathering.
        dev: The device on which to perform computations (e.g., CUDA or CPU).

    Returns:
        None. The model's layers are modified in-place (quantized).
    """

    # -- 0) Preparations: label modules, turn off cache to simplify capturing data
    print("Starting ...")

    for name, module in model.named_modules():
        # Tag each module with a global name (for debugging/logging).
        module.global_name = args.model + name

    use_cache = model.config.use_cache
    model.config.use_cache = False

    # -- 1) Move certain embedding modules to the device, depending on model type.
    #    This is needed to ensure that we can pass data forward without errors or partial CPU/GPU conflicts.
    if "opt" in args.model:
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
    elif "llama" in args.model:
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)

    # -- 2) Create a tensor to store intermediate layer inputs (calibration data).
    #    We want to feed multiple samples (args.nsamples) of length model.seqlen,
    #    capturing the hidden states dimension (model.config.hidden_size).
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), 
        dtype=dtype, 
        device=dev
    )

    # We'll store the attention mask here. 'cache' is a dict for convenience.
    cache = {"i": 0, "attention_mask": None}

    # -- 3) Define a 'Catcher' module that intercepts the forward pass of the first layer.
    #    When the first layer is called, it will store the input in 'inps' and
    #    the attention mask in 'cache["attention_mask"]', then raise an exception to exit early.
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            # Save the layer input (hidden states) for calibration.
            inps[cache["i"]] = inp
            cache["i"] += 1
            # Capture the attention mask for later usage.
            cache["attention_mask"] = kwargs["attention_mask"]
            # Raise an exception to break out of the forward pass once we have our stats.
            raise ValueError

    # Replace the first layer with the 'Catcher' wrapper, so we can grab input stats.
    layers[0] = Catcher(layers[0])

    # -- 4) Run a single pass of the model on samples from the dataloader to fill 'inps' with actual data.
    #    We only do this for the first layer. The 'Catcher' will store everything and throw ValueError to break.
    for batch in dataloader:
        try:
            model(batch[0].to(dev)) 
            # We only need to do this once per sample until we fill up inps,
            # the forward pass is aborted by the Catcher after storing data.
        except ValueError:
            pass

    # Now restore the original first layer (remove the Catcher).
    layers[0] = layers[0].module

    # Move these embedding modules back to CPU to free GPU memory (cleanup step).
    layers[0] = layers[0].cpu()
    if "opt" in args.model:
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.cpu()
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    elif "llama" in args.model:
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    # Prepare an 'outs' buffer. We'll store layer outputs here after quantization.
    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]

    print("Ready.")
    
    # -- 5) Main loop: for each layer, do GPTQ-based quantization (including braq/binary).
    #    We will successively quantize each layer, compute outputs for the dataset,
    #    and pass those outputs as inputs to the next layer.
    for i in range(len(layers)):

        # Move the current layer to GPU for processing
        layer = layers[i].to(dev)

        # 'find_layers' presumably enumerates submodules of 'layer' we want to quantize.
        # For example, we might only quantize certain linear modules or key-value-projection modules.
        subset = find_layers(layer)

        # 'gptq' dictionary will map module names to GPTQ objects. Each GPTQ object
        # handles the binarization/quantization logic for that submodule's weights.
        gptq = {}
        for name in subset:
            # 'quant_only in name' can be used to filter which submodules get quantized;
            # 'args.invert' might invert that selection logic.
            if (not (args.minlayer <= i < args.maxlayer and args.quant_only in name)) == (not args.invert):
                continue

            # We instantiate the binarization (or multi-bit) object with the selected method (braq, cabr, etc.),
            # which references the code you've seen for 'high_order_residual' or new methods.
            braq_quantizer = Binarization(
                subset[name].weight,
                method=args.low_quant_method,
                groupsize=groupsize,
            )

            # Wrap that in a GPTQ object that handles accumulation of stats & final quantization.
            gptq[name] = BRAGPTQ(
                subset[name],
                braq_quantizer,
                salient_metric=args.salient_metric,
                disable_gptq=args.disable_gptq,
            )

        # -- 6) Hooks to accumulate layer input/output data for each submodule. 
        #    This is used by GPTQ to approximate Hessian or to track gradients/residuals.
        def add_batch(name):
            def tmp(_, inp, out):
                # For GPTQ approach, we store the input and output of the submodule
                # so we can solve for binarization/quantization parameters that best fit.
                gptq[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gptq:
            # Register a forward hook on each submodule so we can log the input & output data.
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        # -- 7) We feed the calibration samples (captured in 'inps') through this layer
        #    to gather stats for the quantization. 'outs' collects the layer outputs.
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        # Remove the hooks after capturing the necessary data.
        for h in handles:
            h.remove()

        # -- 8) Perform the quantization using the GPTQ logic for each submodule:
        #    'fasterquant' is presumably where the main binarization or multi-bit quant actually happens:
        #    1) Possibly builds a Hessian approximation or merges sub-blocks
        #    2) Calls braq or related method for the salient portion
        #    3) Calls single-bit binarization for non-salient portion
        #    4) Updates submodule weights in place with the quantized values.
        for name in gptq:
            print(i, name)
            print("Quantizing ...")
            info = gptq[name].fasterquant(
                percdamp=args.percdamp, 
                blocksize=args.blocksize,
            )
            # info might hold stats about the final quantization (error metrics, etc.).

            # Free any large memory structures (like stored batches or Hessian approximations).
            gptq[name].free()

        # -- 9) We run the layer forward again with the newly quantized weights
        #    to get the final outputs, which will serve as the input to the next layer.
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        # Move the now-quantized layer back to CPU to reduce GPU memory usage
        layers[i] = layer.cpu()
        del layer
        del gptq
        torch.cuda.empty_cache()

        # -- 10) Swap 'inps' and 'outs' so that what we just computed as outputs 
        #     becomes the input for the next layer iteration.
        inps, outs = outs, inps

    # Restore the original cache setting
    model.config.use_cache = use_cache

