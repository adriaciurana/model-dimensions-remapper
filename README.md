# Model Dimensions Remapper

This small project enables the modification of the dimensions of an existing model. For instance, it allows you to adapt some models to work with 1D or 3D data.

it's not automatic in all the architectures, for instance for the ViT, you need to reformulate the `forward` function.


## Overview

In the realm of deep learning models, it is common to encounter scenarios where the input dimensions need to be adjusted to accommodate specific data formats like Audio or Video. This project offers a solution by providing the capability to modify the dimensions of an existing model effortlessly.

## Key Features

- **Dimension Flexibility**: Modify the dimensions of a model to suit different input data shapes.
- **Adaptability**: Easily transform a model designed for a specific dimensionality (e.g., 2D) to work seamlessly with other dimensions (e.g., 1D or 3D).
- **Compatibility**: Ideal for scenarios where pre-existing models need to be adapted to diverse input data formats.

## How to Use

1. Clone the repository to your local machine.
   ```bash
   git clone https://github.com/adriaciurana/model-dimensions-remapper.git
   ```

2. Follow the installation instructions in the provided documentation.

3. Integrate the project into your existing codebase.

4. Modify the dimensions of your model as needed for the desired input data format.

5. Continue using the adapted model in your applications.

## Requirements

- Python 3.x
- Dependencies: einops

## Example

```python
from remapper import convert, Translator2dto1d

model = mobilenet_v2()
model = convert(model, translator=Translator2dto1d())
print(model(torch.zeros(1, 3, 224)).shape)
```

## Contributions

Contributions are encouraged! If you have ideas for improvements or new features, please submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).