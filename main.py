import torch
from remapper import Translator2dto1d, TranslatorOnlyResNet2dto1d, convert

if __name__ == "__main__":
    from torchvision.models import efficientnet_b0, mobilenet_v2, resnet18

    model = resnet18()
    model = convert(model, translator=TranslatorOnlyResNet2dto1d())
    print(model(torch.zeros(1, 3, 224)).shape)

    model = mobilenet_v2()
    model = convert(model, translator=Translator2dto1d())
    print(model(torch.zeros(1, 3, 224)).shape)

    model = efficientnet_b0()
    model = convert(model, translator=Translator2dto1d())
    print(model(torch.zeros(1, 3, 224)).shape)
