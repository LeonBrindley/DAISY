import torch
import torch.nn as nn

def add_dropout(model, layer_name_str, dropout_rate=0.5):
    """
    Adds a Dropout layer before each layer in the model whose name contains the given layer_name_str.
    
    Parameters:
    model (torch.nn.Module): The model to modify.
    layer_name_str (str): The substring to search for in layer names.
    dropout_rate (float): The dropout rate to use.
    
    Returns:
    torch.nn.Module: The modified model.
    """
    
    for name, module in model.named_children():
        # Recursively apply dropout to submodules
        if len(list(module.children())) > 0:
            add_dropout(module, layer_name_str, dropout_rate)
        
        # Check if the layer name contains the specified string
        if layer_name_str in name:
            # Replace the module with a Sequential containing Dropout + original module
            setattr(model, name, nn.Sequential(
                nn.Dropout(p=dropout_rate),
                module
            ))

    return model