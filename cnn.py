import torch


class SimpleCNN(torch.nn.Module):
    def __init__(self, n_in_channels: int = 3, n_hidden_layers: int = 3, n_kernels: int = 32, kernel_size: int = 7):
        """Simple CNN with `n_hidden_layers`, `n_kernels`, and `kernel_size` as hyperparameters"""
        super().__init__()

        cnn = []
        for i in range(n_hidden_layers):
            cnn.append(torch.nn.Conv2d(
                in_channels=n_in_channels,
                out_channels=n_kernels,
                kernel_size=kernel_size,
                padding=int(kernel_size / 2)
            ))
            cnn.append(torch.nn.ReLU())
            n_in_channels = n_kernels
        self.hidden_layers = torch.nn.Sequential(*cnn)

        self.output_layer = torch.nn.Conv2d(
            in_channels=n_in_channels,
            out_channels=3,
            kernel_size=kernel_size,
            padding=int(kernel_size / 2)
        )

    def forward(self, x):
        """Apply CNN to input `x` of shape (N, n_channels, X, Y), where N=n_samples and X, Y are spatial dimensions"""
        cnn_out = self.hidden_layers(x)  # apply hidden layers (N, n_in_channels, X, Y) -> (N, n_kernels, X, Y)
        pred = self.output_layer(cnn_out)  # apply output layer (N, n_kernels, X, Y) -> (N, 3, X, Y)
        return pred


# if __name__ == "__main__":
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     # Input minibatch with 4 samples: 100 by 100 images and 3 color channels
#     input_tensor = torch.arange(4 * 3 * 100 * 100, dtype=torch.float32, device=device).reshape((4, 3, 100, 100))
#
#     image_nn = SimpleCNN()
#     print(image_nn)
#     print("\nApplying ImageNN")
#     print(f"input tensor shape: {input_tensor.shape}")
#     output_tensor = image_nn(input_tensor)
#     print(f"output tensor shape: {output_tensor.shape}")
