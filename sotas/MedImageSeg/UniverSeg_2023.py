from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, Callable

import einops as E
import torch
from torch import nn

from pydantic import validate_arguments

size2t = Union[int, Tuple[int, int]]

from torch.nn import init
"""
https://github.com/JJGO/UniverSeg
"""
"""
@inproceedings{butoi2023universeg,
  title={Universeg: Universal medical image segmentation},
  author={Butoi, Victor Ion and Ortiz, Jose Javier Gonzalez and Ma, Tianyu and Sabuncu, Mert R and Guttag, John and Dalca, Adrian V},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={21438--21451},
  year={2023}
}
"""
"""
UniverSeg是一个few-shot分割模型，它的初始化和输入数据有以下几个关键点需要注意：
1. 模型初始化
model = UniverSeg(
    in_channels=1,      # 输入图像的通道数
    num_classes=4,      # 分割输出的类别数
    num_support=5,      # 支持集样本数量
    encoder_blocks=[64, 64, 64, 64]  # 可选，编码器块的通道配置
)
---------------------------------------------------------------------------
2.输入数据要求
# 目标图像 (待分割的图像)
x = torch.randn(B, in_channels, H, W)  # [batch_size, channels, height, width]

# 支持集图像 (用于few-shot学习的样本图像)
support_images = torch.randn(B, in_channels, num_support, H, W)  
# [batch_size, channels, num_support, height, width]

# 支持集标签 (支持集图像对应的分割标签)
support_labels = torch.randint(0, 2, (B, 1, num_support, H, W)).float()
# [batch_size, 1, num_support, height, width]
---------------------------------------------------------------------------
3.特别注意事项：
支持集数量(num_support)必须在模型初始化时指定，并与输入数据匹配
support_images和support_labels的num_support维度必须相同
support_labels通常是二值的(0或1)，表示分割掩码
所有输入的空间维度(H, W)必须一致
in_channels必须与输入图像的通道数匹配
"""
def initialize_weight(
    weight: torch.Tensor,
    distribution: Optional[str],
    nonlinearity: Optional[str] = "LeakyReLU",
) -> None:
    """Initialize the weight tensor with a chosen distribution and nonlinearity.

    Args:
        weight (torch.Tensor): The weight tensor to initialize.
        distribution (Optional[str]): The distribution to use for initialization. Can be one of "zeros",
            "kaiming_normal", "kaiming_uniform", "kaiming_normal_fanout", "kaiming_uniform_fanout",
            "glorot_normal", "glorot_uniform", or "orthogonal".
        nonlinearity (Optional[str]): The type of nonlinearity to use. Can be one of "LeakyReLU", "Sine",
            "Tanh", "Silu", or "Gelu".

    Returns:
        None
    """

    if distribution is None:
        return

    if nonlinearity:
        nonlinearity = nonlinearity.lower()
        if nonlinearity == "leakyrelu":
            nonlinearity = "leaky_relu"

    if nonlinearity == "sine":
        warnings.warn("sine gain not implemented, defaulting to tanh")
        nonlinearity = "tanh"

    if nonlinearity is None:
        nonlinearity = "linear"

    if nonlinearity in ("silu", "gelu"):
        nonlinearity = "leaky_relu"

    gain = 1 if nonlinearity is None else init.calculate_gain(nonlinearity)

    if distribution == "zeros":
        init.zeros_(weight)
    elif distribution == "kaiming_normal":
        init.kaiming_normal_(weight, nonlinearity=nonlinearity)
    elif distribution == "kaiming_uniform":
        init.kaiming_uniform_(weight, nonlinearity=nonlinearity)
    elif distribution == "kaiming_normal_fanout":
        init.kaiming_normal_(weight, nonlinearity=nonlinearity, mode="fan_out")
    elif distribution == "kaiming_uniform_fanout":
        init.kaiming_uniform_(weight, nonlinearity=nonlinearity, mode="fan_out")
    elif distribution == "glorot_normal":
        init.xavier_normal_(weight, gain=gain)
    elif distribution == "glorot_uniform":
        init.xavier_uniform_(weight, gain)
    elif distribution == "orthogonal":
        init.orthogonal_(weight, gain)
    else:
        raise ValueError(f"Unsupported distribution '{distribution}'")


def initialize_bias(
    bias: torch.Tensor,
    distribution: Optional[float] = 0,
    nonlinearity: Optional[str] = "LeakyReLU",
    weight: Optional[torch.Tensor] = None,
) -> None:
    """Initialize the bias tensor with a constant or a chosen distribution and nonlinearity.

    Args:
        bias (torch.Tensor): The bias tensor to initialize.
        distribution (Optional[float]): The constant value to initialize the bias to.
        nonlinearity (Optional[str]): The type of nonlinearity to use when initializing the bias.
        weight (Optional[torch.Tensor]): The weight tensor to use when initializing the bias.

    Returns:
        None
    """

    if distribution is None:
        return

    if isinstance(distribution, (int, float)):
        init.constant_(bias, distribution)
    else:
        raise NotImplementedError(f"Unsupported distribution '{distribution}'")


def initialize_layer(
    layer: nn.Module,
    distribution: Optional[str] = "kaiming_normal",
    init_bias: Optional[float] = 0,
    nonlinearity: Optional[str] = "LeakyReLU",
) -> None:
    """Initialize the weight and bias tensors of a linear or convolutional layer.

    Args:
        layer (nn.Module): The layer to initialize.
        distribution (Optional[str]): The distribution to use for weight initialization.
        init_bias (Optional[float]): The value to use for bias initialization.
        nonlinearity (Optional[str]): The type of nonlinearity to use when initializing the layer.

    Returns:
        None
    """

    assert isinstance(
        layer, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)
    ), f"Can only be applied to linear and conv layers, given {layer.__class__.__name__}"

    initialize_weight(layer.weight, distribution, nonlinearity)
    if layer.bias is not None:
        initialize_bias(
            layer.bias, init_bias, nonlinearity=nonlinearity, weight=layer.weight
        )


def reset_conv2d_parameters(
    model: nn.Module,
    init_distribution: Optional[str],
    init_bias: Optional[float],
    nonlinearity: Optional[str],
) -> None:
    """Reset the parameters of all convolutional layers in the model.

    Args:
        model (nn.Module): The model to reset the convolutional layers of.
        init_distribution (Optional[str]): The distribution to use for weight initialization.
        init_bias (Optional[float]): The value to use for bias initialization.
        nonlinearity (Optional[str]): The type of nonlinearity to use when initializing the layers.

    Returns:
        None
    """

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            initialize_layer(
                module,
                distribution=init_distribution,
                init_bias=init_bias,
                nonlinearity=nonlinearity,
            )
            

"""
Module containing utility functions for validating arguments using Pydantic.

Functions:
    - as_2tuple(val: size2t) -> Tuple[int, int]: Convert integer or 2-tuple to 2-tuple format.
    - validate_arguments_init(class_) -> class_: Decorator to validate the arguments of the __init__ method using Pydantic.
"""

from typing import Any, Dict, Tuple, Union

from pydantic import validate_arguments

size2t = Union[int, Tuple[int, int]]
Kwargs = Dict[str, Any]


def as_2tuple(val: size2t) -> Tuple[int, int]:
    """
    Convert integer or 2-tuple to 2-tuple format.

    Args:
        val (Union[int, Tuple[int, int]]): The value to convert.

    Returns:
        Tuple[int, int]: The converted 2-tuple.

    Raises:
        AssertionError: If val is not an integer or a 2-tuple with length 2.
    """
    if isinstance(val, int):
        return (val, val)
    assert isinstance(val, (list, tuple)) and len(val) == 2
    return tuple(val)


def validate_arguments_init(class_):
    """
    Decorator to validate the arguments of the __init__ method using Pydantic.

    Args:
        class_ (Any): The class to decorate.

    Returns:
        class_: The decorated class with validated __init__ method.
    """
    class_.__init__ = validate_arguments(class_.__init__)
    return class_


def vmap(module: Callable, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    """
    Applies the given module over the initial batch dimension and the second (group) dimension.

    Args:
        module: a callable that is applied over the batch dimension and the second (group) dimension.
                must support batch operations
        x: tensor of shape (batch_size, group_size, ...).
        args: positional arguments to pass to `module`.
        kwargs: keyword arguments to pass to `module`.

    Returns:
        The output tensor with the same shape as the input tensor.
    """
    batch_size, group_size, *_ = x.shape
    grouped_input = E.rearrange(x, "B S ... -> (B S) ...")
    grouped_output = module(grouped_input, *args, **kwargs)
    output = E.rearrange(
        grouped_output, "(B S) ... -> B S ...", B=batch_size, S=group_size
    )
    return output


def vmap_fn(fn: Callable) -> Callable:
    """
    Returns a callable that applies the input function over the initial batch dimension and the second (group) dimension.

    Args:
        fn: function to apply over the batch dimension and the second (group) dimension.

    Returns:
        A callable that applies the input function over the initial batch dimension and the second (group) dimension.
    """

    def vmapped_fn(*args, **kwargs):
        return vmap(fn, *args, **kwargs)

    return vmapped_fn


class Vmap(nn.Module):
    def __init__(self, module: nn.Module):
        """
        Applies the given module over the initial batch dimension and the second (group) dimension.

        Args:
            module: module to apply over the batch dimension and the second (group) dimension.
        """
        super().__init__()
        self.vmapped = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the given module over the initial batch dimension and the second (group) dimension.

        Args:
            x: tensor of shape (batch_size, group_size, ...).

        Returns:
            The output tensor with the same shape as the input tensor.
        """
        return vmap(self.vmapped, x)


def vmap_cls(module_type: type) -> Callable:
    """
    Returns a callable that applies the input module type over the initial batch dimension and the second (group) dimension.

    Args:
        module_type: module type to apply over the batch dimension and the second (group) dimension.

    Returns:
        A callable that applies the input module type over the initial batch dimension and the second (group) dimension.
    """

    def vmapped_cls(*args, **kwargs):
        module = module_type(*args, **kwargs)
        return Vmap(module)

    return vmapped_cls



class CrossConv2d(nn.Conv2d):
    """
    Compute pairwise convolution between all element of x and all elements of y.
    x, y are tensors of size B,_,C,H,W where _ could be different number of elements in x and y
    essentially, we do a meshgrid of the elements to get B,Sx,Sy,C,H,W tensors, and then
    pairwise conv.
    Args:
        x (tensor): B,Sx,Cx,H,W
        y (tensor): B,Sy,Cy,H,W
    Returns:
        tensor: B,Sx,Sy,Cout,H,W
    """
    """
    CrossConv2d is a convolutional layer that performs pairwise convolutions between elements of two input tensors.

    Parameters
    ----------
    in_channels : int or tuple of ints
        Number of channels in the input tensor(s).
        If the tensors have different number of channels, in_channels must be a tuple
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple of ints
        Size of the convolutional kernel.
    stride : int or tuple of ints, optional
        Stride of the convolution. Default is 1.
    padding : int or tuple of ints, optional
        Zero-padding added to both sides of the input. Default is 0.
    dilation : int or tuple of ints, optional
        Spacing between kernel elements. Default is 1.
    groups : int, optional
        Number of blocked connections from input channels to output channels. Default is 1.
    bias : bool, optional
        If True, adds a learnable bias to the output. Default is True.
    padding_mode : str, optional
        Padding mode. Default is "zeros".
    device : str, optional
        Device on which to allocate the tensor. Default is None.
    dtype : torch.dtype, optional
        Data type assigned to the tensor. Default is None.

    Returns
    -------
    torch.Tensor
        Tensor resulting from the pairwise convolution between the elements of x and y.

    Notes
    -----
    x and y are tensors of size (B, Sx, Cx, H, W) and (B, Sy, Cy, H, W), respectively,
    The function does the cartesian product of the elements of x and y to obtain a tensor
    of size (B, Sx, Sy, Cx + Cy, H, W), and then performs the same convolution for all 
    (B, Sx, Sy) in the batch dimension. Runtime and memory are O(Sx * Sy).

    Examples
    --------
    >>> x = torch.randn(2, 3, 4, 32, 32)
    >>> y = torch.randn(2, 5, 6, 32, 32)
    >>> conv = CrossConv2d(in_channels=(4, 6), out_channels=7, kernel_size=3, padding=1)
    >>> output = conv(x, y)
    >>> output.shape  #(2, 3, 5, 7, 32, 32)
    """

    @validate_arguments
    def __init__(
        self,
        in_channels: size2t,
        out_channels: int,
        kernel_size: size2t,
        stride: size2t = 1,
        padding: size2t = 0,
        dilation: size2t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:

        if isinstance(in_channels, (list, tuple)):
            concat_channels = sum(in_channels)
        else:
            concat_channels = 2 * in_channels

        super().__init__(
            in_channels=concat_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise convolution between all elements of x and all elements of y.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of size (B, Sx, Cx, H, W).
        y : torch.Tensor
            Input tensor of size (B, Sy, Cy, H, W).

        Returns
        -------
        torch.Tensor
            Tensor resulting from the cross-convolution between the elements of x and y.
            Has size (B, Sx, Sy, Co, H, W), where Co is the number of output channels.
        """
        B, Sx, *_ = x.shape
        _, Sy, *_ = y.shape

        xs = E.repeat(x, "B Sx Cx H W -> B Sx Sy Cx H W", Sy=Sy)
        ys = E.repeat(y, "B Sy Cy H W -> B Sx Sy Cy H W", Sx=Sx)

        xy = torch.cat([xs, ys], dim=3,)

        batched_xy = E.rearrange(xy, "B Sx Sy C2 H W -> (B Sx Sy) C2 H W")
        batched_output = super().forward(batched_xy)

        output = E.rearrange(
            batched_output, "(B Sx Sy) Co H W -> B Sx Sy Co H W", B=B, Sx=Sx, Sy=Sy
        )
        return output

def get_nonlinearity(nonlinearity: Optional[str]) -> nn.Module:
    if nonlinearity is None:
        return nn.Identity()
    if nonlinearity == "Softmax":
        # For Softmax, we need to specify the channel dimension
        return nn.Softmax(dim=1)
    if hasattr(nn, nonlinearity):
        return getattr(nn, nonlinearity)()
    raise ValueError(f"nonlinearity {nonlinearity} not found")


@validate_arguments_init
@dataclass(eq=False, repr=False)
class ConvOp(nn.Sequential):

    in_channels: int
    out_channels: int
    kernel_size: size2t = 3
    nonlinearity: Optional[str] = "LeakyReLU"
    init_distribution: Optional[str] = "kaiming_normal"
    init_bias: Union[None, float, int] = 0.0

    def __post_init__(self):
        super().__init__()
        self.conv = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            padding_mode="zeros",
            bias=True,
        )

        if self.nonlinearity is not None:
            self.nonlin = get_nonlinearity(self.nonlinearity)

        reset_conv2d_parameters(
            self, self.init_distribution, self.init_bias, self.nonlinearity
        )


@validate_arguments_init
@dataclass(eq=False, repr=False)
class CrossOp(nn.Module):

    in_channels: size2t
    out_channels: int
    kernel_size: size2t = 3
    nonlinearity: Optional[str] = "LeakyReLU"
    init_distribution: Optional[str] = "kaiming_normal"
    init_bias: Union[None, float, int] = 0.0

    def __post_init__(self):
        super().__init__()

        self.cross_conv = CrossConv2d(
            in_channels=as_2tuple(self.in_channels),
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
        )

        if self.nonlinearity is not None:
            self.nonlin = get_nonlinearity(self.nonlinearity)

        reset_conv2d_parameters(
            self, self.init_distribution, self.init_bias, self.nonlinearity
        )

    def forward(self, target, support):
        interaction = self.cross_conv(target, support).squeeze(dim=1)

        if self.nonlinearity is not None:
            interaction = vmap(self.nonlin, interaction)

        new_target = interaction.mean(dim=1, keepdims=True)

        return new_target, interaction


@validate_arguments_init
@dataclass(eq=False, repr=False)
class CrossBlock(nn.Module):

    in_channels: size2t
    cross_features: int
    conv_features: Optional[int] = None
    cross_kws: Optional[Dict[str, Any]] = None
    conv_kws: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        super().__init__()

        conv_features = self.conv_features or self.cross_features
        cross_kws = self.cross_kws or {}
        conv_kws = self.conv_kws or {}

        self.cross = CrossOp(self.in_channels, self.cross_features, **cross_kws)
        self.target = Vmap(ConvOp(self.cross_features, conv_features, **conv_kws))
        self.support = Vmap(ConvOp(self.cross_features, conv_features, **conv_kws))

    def forward(self, target, support):
        target, support = self.cross(target, support)
        target = self.target(target)
        support = self.support(support)
        return target, support


@validate_arguments_init
@dataclass(eq=False, repr=False)
class UniverSeg(nn.Module):
    in_channels: int = 1
    num_classes: int = 1
    encoder_blocks: List[size2t] = None
    decoder_blocks: Optional[List[size2t]] = None
    num_support: int = 5  # 添加支持样本数量参数

    def __post_init__(self):
        super().__init__()
        
        if self.encoder_blocks is None:
            self.encoder_blocks = [64, 64, 64, 64]

        self.downsample = nn.MaxPool2d(2, 2)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        self.enc_blocks = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()

        encoder_blocks = list(map(as_2tuple, self.encoder_blocks))
        decoder_blocks = self.decoder_blocks or encoder_blocks[-2::-1]
        decoder_blocks = list(map(as_2tuple, decoder_blocks))

        block_kws = dict(cross_kws=dict(nonlinearity=None))

        # 修改输入通道配置
        support_channels = self.in_channels + 1  # 每个support sample的通道数(image + label)
        in_ch = (self.in_channels, support_channels)
        out_channels = self.num_classes
        out_activation = None

        # Encoder
        skip_outputs = []
        for (cross_ch, conv_ch) in encoder_blocks:
            block = CrossBlock(in_ch, cross_ch, conv_ch, **block_kws)
            in_ch = conv_ch
            self.enc_blocks.append(block)
            skip_outputs.append(in_ch)

        # Decoder
        skip_chs = skip_outputs[-2::-1]
        for (cross_ch, conv_ch), skip_ch in zip(decoder_blocks, skip_chs):
            block = CrossBlock(in_ch + skip_ch, cross_ch, conv_ch, **block_kws)
            in_ch = conv_ch
            self.dec_blocks.append(block)

        self.out_conv = ConvOp(
            in_ch, out_channels, kernel_size=1, nonlinearity=out_activation,
        )

    def forward(self, target_image, support_images, support_labels):
        # 确保输入维度正确
        B, C, H, W = target_image.shape
        assert C == self.in_channels, f"Expected {self.in_channels} channels, got {C}"
        assert support_images.shape[1] == self.in_channels
        assert support_images.shape[2] == self.num_support
        
        # 重新排列target
        target = target_image.unsqueeze(1)  # [B, 1, C, H, W]
        
        # 合并support images和labels
        support = []
        for i in range(self.num_support):
            support_img = support_images[:, :, i, :, :]  # [B, C, H, W]
            support_label = support_labels[:, :, i, :, :]  # [B, 1, H, W]
            support.append(torch.cat([support_img, support_label], dim=1))  # [B, C+1, H, W]
        support = torch.stack(support, dim=1)  # [B, num_support, C+1, H, W]

        pass_through = []

        for i, encoder_block in enumerate(self.enc_blocks):
            target, support = encoder_block(target, support)
            if i == len(self.encoder_blocks) - 1:
                break
            pass_through.append((target, support))
            target = vmap(self.downsample, target)
            support = vmap(self.downsample, support)

        for decoder_block in self.dec_blocks:
            target_skip, support_skip = pass_through.pop()
            target = torch.cat([vmap(self.upsample, target), target_skip], dim=2)
            support = torch.cat([vmap(self.upsample, support), support_skip], dim=2)
            target, support = decoder_block(target, support)

        target = E.rearrange(target, "B 1 C H W -> B C H W")
        target = self.out_conv(target)

        return target




if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 测试参数
    batch_size = 4
    img_size = 256
    in_channels = 1
    num_classes = 4
    num_support = 5
    
    # 创建测试输入
    x = torch.randn(batch_size, in_channels, img_size, img_size).to(device)
    support_images = torch.randn(batch_size, in_channels, num_support, img_size, img_size).to(device)
    support_labels = torch.randint(0, 2, (batch_size, 1, num_support, img_size, img_size)).float().to(device)
    
    # 创建模型实例
    model = UniverSeg(
        in_channels=in_channels,
        num_classes=num_classes,
        num_support=num_support
    ).to(device)
    
    # 测试前向传播
    y = model(x, support_images, support_labels)
    print(f"Input shape: {x.shape}")
    print(f"Support images shape: {support_images.shape}")
    print(f"Support labels shape: {support_labels.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        
