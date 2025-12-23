from src.models.unet import UNet

def get_model(config):
    model_name = config.model.name
    
    if model_name == 'unet':
        return UNet(config)
    # Add other models here (e.g., resnet, inception)
    # elif model_name == 'resnet':
    #     return ResNet(...)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
