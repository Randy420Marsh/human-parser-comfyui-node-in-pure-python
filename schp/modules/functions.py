import torch
import torch.distributed as dist
import torch.autograd as autograd
from torch.autograd.function import once_differentiable


# Activation names
ACT_LEAKY_RELU = "leaky_relu"
ACT_RELU = "relu"
ACT_ELU = "elu"
ACT_NONE = "none"


class InPlaceABN(autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, running_mean, running_var,
                training=True, momentum=0.1, eps=1e-05, activation="leaky_relu", slope=0.01):
        ctx.training = training
        ctx.momentum = momentum
        ctx.eps = eps
        ctx.activation = activation
        ctx.slope = slope
        ctx.affine = weight is not None and bias is not None
        
        if training:
            mean = x.mean(dim=[0, 2, 3], keepdim=True)
            var = x.var(dim=[0, 2, 3], keepdim=True)
            
            running_mean = (1 - momentum) * running_mean + momentum * mean
            running_var = (1 - momentum) * running_var + momentum * var * x.size(0) / (x.size(0) - 1)
        else:
            mean = running_mean
            var = running_var
        
        x_norm = (x - mean.view(1, -1, 1, 1)) / torch.sqrt(var.view(1, -1, 1, 1) + eps)
        
        if ctx.affine:
            x_norm = x_norm * weight.view(1, -1, 1, 1) + bias.view(1, -1, 1, 1)
        
        # Apply activation
        if ctx.activation == ACT_LEAKY_RELU:
            x_norm = torch.where(x_norm > 0, x_norm, x_norm * ctx.slope)
        elif ctx.activation == ACT_RELU:
            x_norm = torch.where(x_norm > 0, x_norm, torch.zeros_like(x_norm))
        elif ctx.activation == ACT_ELU:
            x_norm = torch.where(x_norm > 0, x_norm, (torch.exp(x_norm) - 1))
        
        ctx.save_for_backward(x_norm, var, weight, bias)
        ctx.mark_non_differentiable(running_mean, running_var)
        
        return x_norm, running_mean, running_var
    
    @staticmethod
    @once_differentiable
    def backward(ctx, dz):
        z, var, weight, bias = ctx.saved_tensors
        
        # Undo activation
        if ctx.activation == ACT_LEAKY_RELU:
            dz = torch.where(z > 0, dz, dz * ctx.slope)
        elif ctx.activation == ACT_RELU:
            dz = torch.where(z > 0, dz, torch.zeros_like(dz))
        elif ctx.activation == ACT_ELU:
            dz = torch.where(z > 0, dz, dz * torch.exp(z))
        
        # Compute gradients
        if ctx.training:
            edz, eydz = dz.mean(dim=[0, 2, 3]), (dz * z).mean(dim=[0, 2, 3])
        else:
            edz, eydz = dz.new_zeros(dz.size(1)), dz.new_zeros(dz.size(1))
        
        dx = dz * (1 / torch.sqrt(var + ctx.eps))
        
        if ctx.affine:
            dweight = eydz if ctx.affine else None
            dbias = edz if ctx.affine else None
        else:
            dweight = None
            dbias = None
        
        return dx, dweight, dbias, None, None, None, None, None, None


class InPlaceABNSync(autograd.Function):
    @classmethod
    def forward(cls, ctx, x, weight, bias, running_mean, running_var,
                training=True, momentum=0.1, eps=1e-05, activation="leaky_relu", slope=0.01, equal_batches=True):
        ctx.training = training
        ctx.momentum = momentum
        ctx.eps = eps
        ctx.activation = activation
        ctx.slope = slope
        ctx.affine = weight is not None and bias is not None

        ctx.world_size = dist.get_world_size() if dist.is_initialized() else 1

        if training:
            mean = x.mean(dim=[0, 2, 3], keepdim=True)
            var = x.var(dim=[0, 2, 3], keepdim=True)

            if ctx.world_size > 1:
                if equal_batches:
                    batch_size = x.size(0) * ctx.world_size
                else:
                    batch_size = x.new_tensor([x.size(0)], dtype=torch.long)
                    dist.all_reduce(batch_size, dist.ReduceOp.SUM)
                    batch_size = batch_size.item()

                factor = x.size(0) / batch_size

                mean_all = mean.clone() * factor
                dist.all_reduce(mean_all, dist.ReduceOp.SUM)
                mean = mean_all

                var_all = (var + (mean - mean_all) ** 2) * factor
                dist.all_reduce(var_all, dist.ReduceOp.SUM)
                var = var_all

                running_mean = (1 - momentum) * running_mean + momentum * mean
                running_var = (1 - momentum) * running_var + momentum * var * batch_size / (batch_size - 1)
            else:
                running_mean = (1 - momentum) * running_mean + momentum * mean
                running_var = (1 - momentum) * running_var + momentum * var * x.size(0) / (x.size(0) - 1)
        else:
            mean = running_mean
            var = running_var

        x_norm = (x - mean.view(1, -1, 1, 1)) / torch.sqrt(var.view(1, -1, 1, 1) + eps)

        if ctx.affine:
            x_norm = x_norm * weight.view(1, -1, 1, 1) + bias.view(1, -1, 1, 1)

        # Apply activation
        if ctx.activation == ACT_LEAKY_RELU:
            x_norm = torch.where(x_norm > 0, x_norm, x_norm * ctx.slope)
        elif ctx.activation == ACT_RELU:
            x_norm = torch.where(x_norm > 0, x_norm, torch.zeros_like(x_norm))
        elif ctx.activation == ACT_ELU:
            x_norm = torch.where(x_norm > 0, x_norm, (torch.exp(x_norm) - 1))

        ctx.save_for_backward(x_norm, var, weight, bias)
        ctx.mark_non_differentiable(running_mean, running_var)

        return x_norm, running_mean, running_var
                    
    @staticmethod
    @once_differentiable
    def backward(ctx, dz):
        z, var, weight, bias = ctx.saved_tensors

        # Undo activation
        if ctx.activation == ACT_LEAKY_RELU:
            dz = torch.where(z > 0, dz, dz * ctx.slope)
        elif ctx.activation == ACT_RELU:
            dz = torch.where(z > 0, dz, torch.zeros_like(dz))
        elif ctx.activation == ACT_ELU:
            dz = torch.where(z > 0, dz, dz * torch.exp(z))

        # Compute gradients
        if ctx.training:
            edz_local = dz.mean(dim=[0, 2, 3])
            eydz_local = (dz * z).mean(dim=[0, 2, 3])

            if ctx.world_size > 1:
                edz = dz.new_zeros(dz.size(1))
                eydz = dz.new_zeros(dz.size(1))

                edz_local *= ctx.world_size
                eydz_local *= ctx.world_size

                dist.all_reduce(edz, dist.ReduceOp.SUM)
                dist.all_reduce(eydz, dist.ReduceOp.SUM)
            else:
                edz = edz_local
                eydz = eydz_local
        else:
            edz = dz.new_zeros(dz.size(1))
            eydz = dz.new_zeros(dz.size(1))

        dx = dz * (1 / torch.sqrt(var + ctx.eps))

        if ctx.affine:
            dweight = eydz_local if ctx.affine else None
            dbias = edz_local if ctx.affine else None
        else:
            dweight = None
            dbias = None

        return dx, dweight, dbias, None, None, None, None, None, None


inplace_abn = InPlaceABN.apply
inplace_abn_sync = InPlaceABNSync.apply

__all__ = ["inplace_abn", "inplace_abn_sync", "ACT_LEAKY_RELU", "ACT_RELU", "ACT_ELU", "ACT_NONE"]
