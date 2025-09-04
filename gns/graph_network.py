from typing import List, Tuple
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


def build_mlp(
    input_size: int,
    hidden_layer_sizes: List[int],
    output_size: int = None,
    output_activation: nn.Module = nn.Identity,
    activation: nn.Module = nn.ReLU,
) -> nn.Module:
    """Build a MultiLayer Perceptron.

    Args:
      input_size: Size of input layer.
      layer_sizes: An array of input size for each hidden layer.
      output_size: Size of the output layer.
      output_activation: Activation function for the output layer.
      activation: Activation function for the hidden layers.

    Returns:
      mlp: An MLP sequential container.
    """
    # Size of each layer
    layer_sizes = [input_size] + hidden_layer_sizes
    if output_size:
        layer_sizes.append(output_size)

    # Number of layers
    nlayers = len(layer_sizes) - 1

    # Create a list of activation functions and
    # set the last element to output activation function
    act = [activation for i in range(nlayers)]
    act[-1] = output_activation

    # Create a torch sequential container
    mlp = nn.Sequential()
    for i in range(nlayers):
        mlp.add_module("NN-" + str(i), nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        mlp.add_module("Act-" + str(i), act[i]())

    return mlp

def initialize_film(film_layer, hidden_dim):
  last_linear = film_layer.film_generator[-2]  # Last Linear layer (before output)
  nn.init.zeros_(last_linear.weight)  # Zero weight
  nn.init.zeros_(last_linear.bias)    # Zero bias
  last_linear.bias.data[:hidden_dim] = 1.0  # Gamma starts at 1, beta starts at 0


class FiLM(nn.Module):
  def __init__(self, hidden_dim: int, mlp_hidden_dim: int, nmlp_layers: int):
      """
      FiLM module (Feature-wise Linear Modulation).

      This version applies FiLM modulation using:
        - h concatenated with upsampled conditioning input (cond MLP)
        - LayerNorm on MLP outputs

      Args:
          hidden_dim (int): Dimension of hidden features h.
          mlp_hidden_dim (int): Hidden layer size in FiLM generator and cond MLPs.
          nmlp_layers (int): Number of layers in the FiLM generator MLP.
      """
      super(FiLM, self).__init__()
      self.hidden_dim = hidden_dim

      # Conditioning MLP
      self.cond_mlp = build_mlp(
          input_size=1,  # scalar cond input
          hidden_layer_sizes=[mlp_hidden_dim],
          output_size=hidden_dim,
          activation=nn.Tanh,
      )
      self.use_cond_mlp = True

      # FiLM generator
      film_input_dim = hidden_dim 
      self.film_generator = build_mlp(
          input_size=film_input_dim,
          hidden_layer_sizes=[mlp_hidden_dim for _ in range(nmlp_layers)],
          output_size=hidden_dim * 2,
      )

      # Initialize FiLM parameters
      initialize_film(self, hidden_dim)

  def forward(self, h: torch.Tensor, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Forward pass for FiLM.

    Args:
        h (torch.Tensor): Hidden state of shape [B, hidden_dim].
        cond (torch.Tensor): Conditioning input of shape [B, 1].

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - gamma: Scaling tensor of shape [B, hidden_dim].
            - beta: Shifting tensor of shape [B, hidden_dim].
    """
    # Upsample cond through cond MLP
    cond_up = self.cond_mlp(cond)

    # Element-wise modulation
    film_input = h * cond_up

    # Generate gamma and beta
    gamma_beta = self.film_generator(film_input)

    gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
    return gamma, beta


class Encoder(nn.Module):
    """Graph network encoder. Encode nodes and edges states to an MLP. The Encode:
    :math: `\mathcal{X} \rightarrow \mathcal{G}` embeds the particle-based state
    representation, :math: `\mathcal{X}`, as a latent graph, :math:
    `G^0 = encoder(\mathcal{X})`, where :math: `G = (V, E, u), v_i \in V`, and
    :math: `e_{i,j} in E`
    """

    def __init__(
        self,
        nnode_in_features: int,
        nnode_out_features: int,
        nedge_in_features: int,
        nedge_out_features: int,
        nmlp_layers: int,
        mlp_hidden_dim: int,
    ):
        """The Encoder implements nodes features :math: `\varepsilon_v` and edge
        features :math: `\varepsilon_e` as multilayer perceptrons (MLP) into the
        latent vectors, :math: `v_i` and :math: `e_{i,j}`, of size 128.

        Args:
          nnode_in_features: Number of node input features (for 2D = 30, calculated
            as [10 = 5 times steps * 2 positions (x, y) +
            4 distances to boundaries (top/bottom/left/right) +
            16 particle type embeddings]).
          nnode_out_features: Number of node output features (latent dimension of
            size 128).
          nedge_in_features: Number of edge input features (for 2D = 3, calculated
            as [2 (x, y) relative displacements between 2 particles + distance
            between 2 particles]).
          nedge_out_features: Number of edge output features (latent dimension of
            size 128).
          nmlp_layer: Number of hidden layers in the MLP (typically of size 2).
          mlp_hidden_dim: Size of the hidden layer (latent dimension of size 128).

        """
        super(Encoder, self).__init__()
        # Encode node features as an MLP
        self.node_fn = nn.Sequential(
            *[
                build_mlp(
                    nnode_in_features,
                    [mlp_hidden_dim for _ in range(nmlp_layers)],
                    nnode_out_features,
                ),
                nn.LayerNorm(nnode_out_features),
            ]
        )
        # Encode edge features as an MLP
        self.edge_fn = nn.Sequential(
            *[
                build_mlp(
                    nedge_in_features,
                    [mlp_hidden_dim for _ in range(nmlp_layers)],
                    nedge_out_features,
                ),
                nn.LayerNorm(nedge_out_features),
            ]
        )

    def forward(self, x: torch.tensor, edge_features: torch.tensor):
        """The forward hook runs when the Encoder class is instantiated

        Args:
          x: Particle state representation as a torch tensor with shape
            (nparticles, nnode_input_features)
          edge_features: Edge features as a torch tensor with shape
            (nparticles, nedge_input_features)

        """
        return self.node_fn(x), self.edge_fn(edge_features)


class InteractionNetwork(MessagePassing):
    def __init__(
        self,
        nnode_in: int,
        nnode_out: int,
        nedge_in: int,
        nedge_out: int,
        nmlp_layers: int,
        mlp_hidden_dim: int,
        film_module: nn.Module = None,
    ):
        """InteractionNetwork derived from torch_geometric MessagePassing class

        Args:
          nnode_in: Number of node inputs (latent dimension of size 128).
          nnode_out: Number of node outputs (latent dimension of size 128).
          nedge_in: Number of edge inputs (latent dimension of size 128).
          nedge_out: Number of edge output features (latent dimension of size 128).
          nmlp_layer: Number of hidden layers in the MLP (typically of size 2).
          mlp_hidden_dim: Size of the hidden layer (latent dimension of size 128).

        """
        # Aggregate features from neighbors
        super(InteractionNetwork, self).__init__(aggr="add")
        # Node MLP
        self.node_fn = nn.Sequential(
            *[
                build_mlp(
                    nnode_in + nedge_out,
                    [mlp_hidden_dim for _ in range(nmlp_layers)],
                    nnode_out,
                ),
                nn.LayerNorm(nnode_out),
            ]
        )
        # Edge MLP
        self.edge_fn = nn.Sequential(
            *[
                build_mlp(
                    nnode_in + nnode_in + nedge_in,
                    [mlp_hidden_dim for _ in range(nmlp_layers)],
                    nedge_out,
                ),
                nn.LayerNorm(nedge_out),
            ]
        )

        # Store the passed FiLM module
        self.film = film_module
        self.use_film = self.film is not None

    def forward(
        self, x: torch.tensor, edge_index: torch.tensor, edge_features: torch.tensor, cond: torch.Tensor = None
    ):
        """The forward hook runs when the InteractionNetwork class is instantiated

        Args:
          x: Particle state representation as a torch tensor with shape
            (nparticles, nnode_input_features)
          edge_index: A torch tensor list of source and target nodes with shape
            (2, nedges)
          edge_features: Edge features as a torch tensor with shape
            (nedges, nedge_in=latent_dim of 128)
          cond (torch.Tensor): Conditioning input of shape [B, 1] for FiLM.

        Returns:
          tuple: Updated node and edge features
        """
        # Save particle state and edge features
        x_residual = x
        edge_features_residual = edge_features
        # Start propagating messages.
        # Takes in the edge indices and all additional data which is needed to
        # construct messages and to update node embeddings.
        # Call PyG propagate() method:
        # 1. Message phase - compute messages for each edge
        # 2. Aggregate phase - aggregate messages for each node
        # 3. Update phase - updates only the node features
        # Update uses the message from step 1 and any original arguments passed to
        # propagate() to update the node embeddings. This is why we need to store
        # the updated edge features to return them from the update() method.
        x, edge_features = self.propagate(
            edge_index=edge_index, x=x, edge_features=edge_features, cond=cond
        )

        return x + x_residual, edge_features + edge_features_residual

    def message(
        self, x_i: torch.tensor, x_j: torch.tensor, edge_features: torch.tensor, cond: torch.Tensor = None
    ) -> torch.tensor:
        """Constructs message from j to i of edge :math:`e_{i, j}`. Tensors :obj:`x`
        passed to :meth:`propagate` can be mapped to the respective nodes :math:`i`
        and :math:`j` by appending :obj:`_i` or :obj:`_j` to the variable name,
        i.e., :obj:`x_i` and :obj:`x_j`.

        Args:
          x_i: Particle state representation as a torch tensor with shape
            (nparticles, nnode_in=latent_dim of 128) at node i
          x_j: Particle state representation as a torch tensor with shape
            (nparticles, nnode_in=latent_dim of 128) at node j
          edge_features: Edge features as a torch tensor with shape
            (nedges, nedge_in=latent_dim of 128)
          cond (torch.Tensor): Conditioning input of shape [B, 1] for FiLM.

        """
        h = torch.cat([x_i, x_j, edge_features], dim=-1)

        # Define mlp and norm by extracting from self.edge_fn
        mlp, norm = self.edge_fn[0], self.edge_fn[1]

        prev_h = None
        for name, layer in mlp.named_children():
            # Save the input to the linear layer
            if name.startswith("NN-"):
                prev_h = h.clone()

            h = layer(h)

            # Apply FiLM ONLY after a hidden linear layer ('NN-1', 'NN-2', etc.)
            if name.startswith("NN-") and self.use_film:
                if name != "NN-0":  # Don't apply to the first layer
                    gamma, beta = self.film(prev_h, cond)
                    h = gamma * h + beta

        h = norm(h)
        self._edge_features = h
        return self._edge_features

    def update(
        self, x_updated: torch.tensor, x: torch.tensor, edge_features: torch.tensor, cond: torch.Tensor = None
    ):
        """Update the particle state representation

        Args:
          x: Particle state representation as a torch tensor with shape
            (nparticles, nnode_in=latent_dim of 128)
          x_updated: Updated particle state representation as a torch tensor with
            shape (nparticles, nnode_in=latent_dim of 128)
          edge_features: Edge features as a torch tensor with shape
            (nedges, nedge_out=latent_dim of 128)
          cond (torch.Tensor): Conditioning input of shape [B, 1] for FiLM.

        Returns:
          tuple: Updated node and edge features
        """
        # Concat node features with a final shape of
        # [nparticles, latent_dim (or nnode_in) *2]
        # This gets called later, after message() and aggregate()
        # Update modified from MessagePassing takes the output of aggregation
        # as first argument and any argument which was initially passed to
        # propagate hence we need to return the stored value of edge_features
        x_updated = torch.cat([x_updated, x], dim=-1)
        mlp, norm = self.node_fn[0], self.node_fn[1]

        prev_x = None
        for name, layer in mlp.named_children():
            # Save the input to the linear layer
            if name.startswith("NN-"):
                prev_x = x_updated.clone()

            x_updated = layer(x_updated)

            # Apply FiLM ONLY after a linear layer ('NN-')
            if name.startswith("NN-") and self.use_film:
                # The original code applies it to hidden layers, so we skip the first one (NN-0)
                if name not in ["NN-0"]:
                    gamma, beta = self.film(prev_x, cond)
                    x_updated = gamma * x_updated + beta

        x_updated = norm(x_updated)
        return x_updated, self._edge_features


class Processor(MessagePassing):
    """The Processor: :math: `\mathcal{G} \rightarrow \mathcal{G}` computes
    interactions among nodes via :math: `M` steps of learned message-passing, to
    generate a sequence of updated latent graphs, :math: `G = (G_1 , ..., G_M )`,
    where :math: `G^{m+1| = GN^{m+1} (G^m )`. It returns the final graph,
    :math: `G^M = PROCESSOR(G^0)`. Message-passing allows information to
    propagate and constraints to be respected: the number of message-passing
    steps required will likely scale with the complexity of the interactions.

    """

    def __init__(
        self,
        nnode_in: int,
        nnode_out: int,
        nedge_in: int,
        nedge_out: int,
        nmessage_passing_steps: int,
        nmlp_layers: int,
        mlp_hidden_dim: int,
        use_film: bool,
        film_mp_blocks: list[int],
    ):
        """Processor derived from torch_geometric MessagePassing class. The
        processor uses a stack of :math: `M GNs` (where :math: `M` is a
        hyperparameter) with identical structure, MLPs as internal edge and node
        update functions, and either shared or unshared parameters. We use GNs
        without global features or global updates (i.e., an interaction network),
        and with a residual connections between the input and output latent node
        and edge attributes.

        Args:
          nnode_in: Number of node inputs (latent dimension of size 128).
          nnode_out: Number of node outputs (latent dimension of size 128).
          nedge_in: Number of edge inputs (latent dimension of size 128).
          nedge_out: Number of edge output features (latent dimension of size 128).
          nmessage_passing_steps: Number of message passing steps.
          nmlp_layer: Number of hidden layers in the MLP (typically of size 2).
          mlp_hidden_dim: Size of the hidden layer (latent dimension of size 128).
          use_film: Whether to use FiLM conditioning.
          film_mp_blocks: A list of message passing layer indices to apply FiLM to.

        """
        super(Processor, self).__init__(aggr="max")

        if use_film:
            self.film_module = FiLM(
                hidden_dim=mlp_hidden_dim,
                mlp_hidden_dim=mlp_hidden_dim,
                nmlp_layers=nmlp_layers,
            )
        else:
            self.film_module = None

        self.film_mp_blocks = film_mp_blocks

        # Create a stack of M Graph Networks GNs.
        self.gnn_stacks = nn.ModuleList()
        for idx in range(nmessage_passing_steps):
            # If use_film is true and the current layer index is in film_layers,
            # pass the central FiLM module. Otherwise, pass None.
            current_film_module = self.film_module if use_film and idx in self.film_mp_blocks else None
            
            self.gnn_stacks.append(
                InteractionNetwork(
                    nnode_in=nnode_in,
                    nnode_out=nnode_out,
                    nedge_in=nedge_in,
                    nedge_out=nedge_out,
                    nmlp_layers=nmlp_layers,
                    mlp_hidden_dim=mlp_hidden_dim,
                    film_module=current_film_module,  # Pass the module instance
                )
            )

    def forward(
        self, x: torch.tensor, edge_index: torch.tensor, edge_features: torch.tensor,  cond: torch.tensor = None
    ):
        """The forward hook runs through GNN stacks when class is instantiated.

        Args:
          x: Particle state representation as a torch tensor with shape
            (nparticles, latent_dim)
          edge_index: A torch tensor list of source and target nodes with shape
            (2, nedges)
          edge_features: Edge features as a torch tensor with shape
            (nparticles, latent_dim)
          cond (torch.Tensor): Conditioning input of shape [B, 1] for FiLM.

        """
        for idx, gnn in enumerate(self.gnn_stacks):
            if idx in self.film_mp_blocks:
                x, edge_features = gnn(x, edge_index, edge_features, cond=cond)
            else:
                x, edge_features = gnn(x, edge_index, edge_features)
        return x, edge_features


class Decoder(nn.Module):
    """The Decoder: :math: `\mathcal{G} \rightarrow \mathcal{Y}` extracts the
    dynamics information from the nodes of the final latent graph,
    :math: `y_i = \delta v (v_i^M)`

    """

    def __init__(
        self, nnode_in: int, nnode_out: int, nmlp_layers: int, mlp_hidden_dim: int
    ):
        """The Decoder coder's learned function, :math: `\detla v`, is an MLP.
        After the Decoder, the future position and velocity are updated using an
        Euler integrator, so the :math: `yi` corresponds to accelerations,
        :math: `\"{p}_i`, with 2D or 3D dimension, depending on the physical domain.

        Args:
          nnode_in: Number of node inputs (latent dimension of size 128).
          nnode_out: Number of node outputs (particle dimension).
          nmlp_layer: Number of hidden layers in the MLP (typically of size 2).
          mlp_hidden_dim: Size of the hidden layer (latent dimension of size 128).
        """
        super(Decoder, self).__init__()
        self.node_fn = build_mlp(
            nnode_in, [mlp_hidden_dim for _ in range(nmlp_layers)], nnode_out
        )

    def forward(self, x: torch.tensor):
        """The forward hook runs when the Decoder class is instantiated

        Args:
          x: Particle state representation as a torch tensor with shape
            (nparticles, nnode_in)

        """
        return self.node_fn(x)


class EncodeProcessDecode(nn.Module):
    def __init__(
        self,
        nnode_in_features: int,
        nnode_out_features: int,
        nedge_in_features: int,
        latent_dim: int,
        nmessage_passing_steps: int,
        nmlp_layers: int,
        mlp_hidden_dim: int,
        use_film: bool,
        film_mp_blocks: list[int],
    ):
        """Encode-Process-Decode function approximator for learnable simulator.

        Args:
          nnode_in_features: Number of node input features (for 2D = 30,
            calculated as [10 = 5 times steps * 2 positions (x, y) +
            4 distances to boundaries (top/bottom/left/right) +
            16 particle type embeddings]).
          nnode_out_features:  Number of node outputs (particle dimension).
          nedge_in_features: Number of edge input features (for 2D = 3,
            calculated as [2 (x, y) relative displacements between 2 particles +
            distance between 2 particles]).
          latent_dim: Size of latent dimension (128)
          nmlp_layer: Number of hidden layers in the MLP (typically of size 2).
          mlp_hidden_dim: Size of the hidden layer (latent dimension of size 128).
          use_film: Whether to use FiLM conditioning.
          film_mp_blocks: A list of message passing layer indices to apply FiLM to.

        """
        super(EncodeProcessDecode, self).__init__()
        self._encoder = Encoder(
            nnode_in_features=nnode_in_features,
            nnode_out_features=latent_dim,
            nedge_in_features=nedge_in_features,
            nedge_out_features=latent_dim,
            nmlp_layers=nmlp_layers,
            mlp_hidden_dim=mlp_hidden_dim,
        )
        self._processor = Processor(
            nnode_in=latent_dim,
            nnode_out=latent_dim,
            nedge_in=latent_dim,
            nedge_out=latent_dim,
            nmessage_passing_steps=nmessage_passing_steps,
            nmlp_layers=nmlp_layers,
            mlp_hidden_dim=mlp_hidden_dim,
            use_film=use_film,
            film_mp_blocks=film_mp_blocks
        )
        self._decoder = Decoder(
            nnode_in=latent_dim,
            nnode_out=nnode_out_features,
            nmlp_layers=nmlp_layers,
            mlp_hidden_dim=mlp_hidden_dim,
        )

    def forward(
        self, x: torch.tensor, edge_index: torch.tensor, edge_features: torch.tensor,  cond: torch.tensor = None
    ):
        """The forward hook runs at instatiation of EncodeProcessorDecode class.

        Args:
          x: Particle state representation as a torch tensor with shape
            (nparticles, nnode_in_features)
          edge_index: A torch tensor list of source and target nodes with shape
            (2, nedges)
          edge_features: Edge features as a torch tensor with shape
            (nedges, nedge_in_features)
          cond (torch.Tensor): Conditioning input of shape [B, 1] for FiLM. 

        Returns:
          x: Particle state representation as a torch tensor with shape
            (nparticles, nnode_out_features)
        """
        x, edge_features = self._encoder(x, edge_features)
        x, edge_features = self._processor(x, edge_index, edge_features, cond=cond)
        x = self._decoder(x)
        return x
