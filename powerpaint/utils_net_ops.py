import torch
from .utils_nets import group_norm_affine_part

def batch_run(sample, func, norm=None, offload=False, key="", var_mean={}):

    if norm:
        assert isinstance(norm, torch.nn.GroupNorm)
        num_groups = norm.num_groups
        c = sample[0].shape[1]
        channel_in_group = int(c / num_groups)  # 16

        if key and len(key) and key in var_mean:
            var, mean = var_mean[key]
        else:
            # print(f"Computing var_mean for {key}")
            vars = []
            means = []
            areas = []
            for tile in sample:
                _, c, h, w = tile.shape  # [1, 512, 144, 144]
                fp32_tile = tile.float()
                tile_reshaped = fp32_tile.contiguous().view(num_groups, channel_in_group, h, w)
                var, mean = torch.var_mean(tile_reshaped, dim=[1, 2, 3], unbiased=False)
                vars.append(var)
                means.append(mean)
                areas.append(h * w)

            var = torch.vstack(vars)
            mean = torch.vstack(means)  # b, c
            areas = torch.tensor(areas, dtype=torch.float32, device=var.device) / max(areas) # b
            areas = areas.unsqueeze(1) / torch.sum(areas)  # b, 1
            var = torch.sum(var * areas, dim=0)  # (c,)
            mean = torch.sum(mean * areas, dim=0)
            var_mean[key] = (var, mean)

    results = []
    while sample:
        tile = sample.pop(0)
        tile = tile.to(device="cuda")  # [1, 4, 144, 144]
        if norm:
            # normalize
            mean = mean.to(device="cuda", dtype=torch.float32).reshape((num_groups, 1, 1, 1))
            var = var.to(device="cuda", dtype=torch.float32).reshape((num_groups, 1, 1, 1))
            fp32_tile = tile.float()
            tile_reshaped = fp32_tile.contiguous().view(num_groups, channel_in_group, *tile.shape[2:])
            tile_normed = (tile_reshaped - mean) / torch.sqrt(var + 1e-2) 
            tile = tile_normed.view(*tile.shape)
            tile = group_norm_affine_part(tile, norm)
            tile = tile.to(dtype=torch.float16)
        result = func(tile)
        if offload:
            result = result.cpu()
        results.append(result)
    return results

def memory_efficient_add_scale(sample, hidden_states, scale):
    results = []
    while sample:
        s = sample.pop(0)
        h = hidden_states.pop(0)
        results.append((s+h)/scale)
    assert len(hidden_states) == 0
    return results


def clone_list(sample):
    clone = []
    for s in sample:
        clone.append(s.detach().clone())
    return clone
