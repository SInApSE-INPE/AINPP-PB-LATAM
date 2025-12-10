from src.models.unet import UNet

def get_model(config):
    model_name = config.model.name
    
    if model_name == 'unet':
        return UNet(
            n_channels=config.model.in_channels,
            n_classes=config.model.out_channels
        )
    # Add other models here (e.g., resnet, inception)
    # elif model_name == 'resnet':
    #     return ResNet(...)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
