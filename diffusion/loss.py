import mlx.core as mx


def diffusion_loss(model_output, noise, loss_type="l2"):
    """
    Calculate the loss between the model output and the noise.
    
    Args:
        model_output: The predicted noise from the model
        noise: The actual noise added
        loss_type: The type of loss to use ('l1' or 'l2')
    """
    if loss_type == "l1":
        loss = mx.abs(noise - model_output)
    elif loss_type == "l2":
        loss = mx.square(noise - model_output)
    else:
        raise ValueError(f"Unrecognized loss type {loss_type}")
    
    return mx.mean(loss)
