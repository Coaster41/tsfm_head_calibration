from torch import nn

class ResidualBlock(nn.Module):
  """TimesFM residual block."""

  def __init__(
      self,
      input_dims,
      hidden_dims,
      output_dims,
  ):
    super(ResidualBlock, self).__init__()
    self.input_dims = input_dims
    self.hidden_dims = hidden_dims
    self.output_dims = output_dims

    # Hidden Layer
    self.hidden_layer = nn.Sequential(
        nn.Linear(input_dims, hidden_dims),
        nn.SiLU(),
    )

    # Output Layer
    self.output_layer = nn.Linear(hidden_dims, output_dims)
    # Residual Layer
    self.residual_layer = nn.Linear(input_dims, output_dims)

  def forward(self, x):
    hidden = self.hidden_layer(x)
    output = self.output_layer(hidden)
    residual = self.residual_layer(x)
    return output + residual