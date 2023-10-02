import torch
import torch.nn as nn
import torch.nn.functional as F


def insert_attn_forward_fn(module, attn, insert_after='bn2'):
    if insert_after == 'conv2':
        target_forward_fn = denselayer_attn_forward
    elif insert_after == 'bn2':
        target_forward_fn = bnact_attn_forward
        module = module.norm2
    elif insert_after == 'bn1':
        target_forward_fn = bnact_attn_forward
        module = module.norm1
    else:
        raise NotImplementedError('The defined insert position is not implemented yet.')

    module.add_module('attn', attn)
    bound_method = target_forward_fn.__get__(module, module.__class__)
    setattr(module, 'forward', bound_method)


def denselayer_attn_forward(self, x):  # noqa: F811
        if isinstance(x, torch.Tensor):
            prev_features = [x]
        else:
            prev_features = x

        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("Memory Efficient not supported in JIT")
            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bottleneck_fn(prev_features)

        new_features = self.conv2(self.norm2(bottleneck_output))
        new_features = self.attn(new_features)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


def bnact_attn_forward(self, x):
        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked.add_(1)  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        x = F.batch_norm(
            x,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight,
            self.bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )
        x = self.drop(x)
        x = self.attn(x)
        x = self.act(x)
        return x