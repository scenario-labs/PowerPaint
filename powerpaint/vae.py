from functools import partial
from PIL import Image

import torch

from diffusers import AutoencoderKL

from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.models.autoencoders.vae import DecoderOutput, DiagonalGaussianDistribution

from diffusers.models.unets.unet_2d_blocks import DownEncoderBlock2D
from diffusers.models.modeling_outputs import AutoencoderKLOutput

from .utils_nets import up_sample, down_sample, resnet_step1, resnet_step2, attention
from .utils_tiles import crop_valid_region, split_tiles
from .utils_tiles import may_offload
from .utils_net_ops import batch_run, memory_efficient_add_scale, clone_list
from .utils_logs import CustomLogger
clog = CustomLogger()

# DECODER PART
def decoder_decode(sample, vae, offload=False, var_mean={}, debug=False):
    decoder = vae.decoder
    clog.print("vae init", check=[sample], debug=debug)

    # 1. sample = decoder.conv_in(sample)
    sample = batch_run(sample, decoder.conv_in, offload=offload)
    clog.print("step1", check=[sample], debug=debug)

    # 2. middle UNetMidBlock2D: sample = decoder.mid_block(sample)
    mid_block = decoder.mid_block

    # 2.a sample = mid_block.resnets[0](sample, None)
    resnet = mid_block.resnets[0]

    hidden_states = clone_list(sample)
    hidden_states = batch_run(
        hidden_states, partial(resnet_step1, resnet),
        norm=resnet.norm1,
        key="decoder-mid_block-resnet0.norm1",
        var_mean=var_mean,
        offload=offload,
    )

    hidden_states = batch_run(
        hidden_states, partial(resnet_step2, resnet),
        norm=resnet.norm2,
        key="decoder-mid_block-resnet0.norm2",
        var_mean=var_mean,
        offload=offload,
    )

    assert resnet.conv_shortcut is None
    sample = memory_efficient_add_scale(sample, hidden_states, resnet.output_scale_factor)
    clog.print("resnet0", check=[sample], debug=debug)
    # end resnet0

    # 2b. AttnProcessor2_0
    assert len(mid_block.attentions) == 1
    attn = mid_block.attentions[0]
    # sample = attn(sample)
    hidden_states = clone_list(sample)
    hidden_states = batch_run(
        hidden_states, partial(attention, attn),
        norm=attn.group_norm,
        key="decoder-mid_block-attn",
        var_mean=var_mean,
        offload=offload,
    )
    sample = memory_efficient_add_scale(sample, hidden_states, attn.rescale_output_factor)
    clog.print("step1", check=[sample], debug=debug)
    # end AttnProcessor2_0

    # 2c. sample = resnet(sample, None) for other resnets
    for j, resnet in enumerate(mid_block.resnets[1:]):
        hidden_states = clone_list(sample)
        hidden_states = batch_run(
            hidden_states, partial(resnet_step1, resnet),
            norm=resnet.norm1,
            key=f"decoder-mid_block-resnet{j+1}.norm1",
            var_mean=var_mean,
            offload=offload,
        )

        hidden_states = batch_run(
            hidden_states, partial(resnet_step2, resnet),
            norm=resnet.norm2,
            key=f"decoder-mid_block-resnet{j+1}.norm2",
            var_mean=var_mean,
            offload=offload,
        )

        assert resnet.conv_shortcut is None
        sample = memory_efficient_add_scale(sample, hidden_states, resnet.output_scale_factor)
        clog.print(f"midblock-resnet{j}", check=[sample], debug=debug)

    # 3. up blocks sample = up_block(sample)
    for k, up_block in enumerate(decoder.up_blocks):
        # 3a. sample = resnet(sample, None) for upblock resnets
        for j, resnet in enumerate(up_block.resnets):
            hidden_states = clone_list(sample)
            hidden_states = batch_run(
                hidden_states, partial(resnet_step1, resnet),
                norm=resnet.norm1,
                key=f"decoder-up_block{k}-resnet{j}.norm1",
                var_mean=var_mean,
                offload=offload,
            )

            hidden_states = batch_run(
                hidden_states, partial(resnet_step2, resnet),
                norm=resnet.norm2,
                key=f"decoder-up_block{k}-resnet{j}.norm2",
                var_mean=var_mean,
                offload=offload,
            )

            if resnet.conv_shortcut is not None:
                sample = batch_run(sample, resnet.conv_shortcut, offload=offload)

            sample = memory_efficient_add_scale(sample, hidden_states, resnet.output_scale_factor)
            clog.print(f"Upblock{k} resnet{j}", check=[sample], debug=debug)

        # 3b. Upblock upsample
        sample = batch_run(sample, partial(up_sample, up_block), offload=offload)
        clog.print(f"upblock{k} upsample", check=[sample], debug=debug)

    # 4. post-process
    sample = batch_run(
        sample, lambda x: decoder.conv_out(decoder.conv_act(x)),
        norm=decoder.conv_norm_out,
        key=f"decoder-conv_norm_out",
        var_mean=var_mean,
        offload=offload,
    )
    clog.print(f"postprocess", check=[sample], debug=debug)
    return sample


def decode(z, vae, offload=False):
    z = batch_run(z, vae.post_quant_conv, offload=offload)
    # dec = vae.decoder(z)
    dec = decoder_decode(z, vae)
    decoded = DecoderOutput(sample=dec).sample
    return decoded


def tile_decode(latents, vae, var_mean={}, debug=False):
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    N, _, height, width = latents.shape
    assert N == 1, "Only one image to upscale"
    in_bboxes, out_bboxes = split_tiles(height, width, vae_scale_factor, is_decoder=True)
    tiles = []
    for w_start, w_end, h_start, h_end in in_bboxes:
        tile = latents[:, :, h_start:h_end, w_start:w_end].cpu()
        tiles.append(tile)

    if len(var_mean):
        # run sequentially each tile one after the other in order to avoid useless memory transfers
        if debug:
            print("Sequential run, no offload")
        decoded_tiles = []
        for tile in tiles:
            tile = batch_run([tile], vae.post_quant_conv, offload=False)[0]
            decoded_tile = decoder_decode(
                [tile], vae, offload=False, var_mean=var_mean, debug=debug,
            )[0]
            # if offload:
            #     decoded_tile = decoded_tile.detach().cpu()
            decoded_tiles.append(decoded_tile)
    else:
        # offload params: decides if the tiles should be returned back to CPU while the next tile is run
        # does not depend if var_mean is known or not
        offload = may_offload(latents.shape[2] * latents.shape[3])
        if debug:
            print(f"Encoder offload: {offload}")

        # run and stop at each norm to compute the var_mean
        tiles = batch_run(tiles, vae.post_quant_conv, offload=offload)
        decoded_tiles = decoder_decode(
            tiles, vae, offload=offload, var_mean=var_mean, debug=debug,
        )  # [n_tiles, 3, tile_output_height, tile_output_width]

    result = torch.zeros(
        (N, decoded_tiles[0].shape[1], height * vae_scale_factor, width * vae_scale_factor),
        device=decoded_tiles[0].device,
        requires_grad=False
    )
    for tile, out_bbox, in_bbox in zip(decoded_tiles, out_bboxes, in_bboxes):
        w_start, w_end, h_start, h_end = out_bbox
        result[:, :, h_start:h_end, w_start:w_end] = crop_valid_region(tile, in_bbox, out_bbox, vae_scale_factor, is_decoder=True)

    return result


# ENCODER PART

def encoder_encode(sample, vae, offload=False, var_mean={}, debug=False):
    encoder = vae.encoder

    # 1. sample = encoder.conv_in(sample)
    sample = batch_run(sample, encoder.conv_in, offload=offload)
    clog.print("step1", check=[sample], debug=debug)

    # 2. down
    for i, down_block in enumerate(encoder.down_blocks):
        # print(f"Downblock{i}")
        assert isinstance(down_block, DownEncoderBlock2D)
        # sample = down_block(sample)
        for j, resnet in enumerate(down_block.resnets):
            # sample = resnet(sample, temb=None)
            hidden_states = clone_list(sample)
            hidden_states = batch_run(
                hidden_states, partial(resnet_step1, resnet),
                norm=resnet.norm1,
                key=f"encoder-down_block{i}-resnet{j}.norm1",
                var_mean=var_mean,
                offload=offload,
            )

            hidden_states = batch_run(
                hidden_states, partial(resnet_step2, resnet),
                key=f"encoder-down_block{i}-resnet{j}.norm2",
                var_mean=var_mean,
                norm=resnet.norm2,
                offload=offload,
            )

            if resnet.conv_shortcut is not None:
                sample = batch_run(sample, resnet.conv_shortcut, offload=offload)

            sample = memory_efficient_add_scale(sample, hidden_states, resnet.output_scale_factor)
            clog.print(f"down_block{i}-resnet{j}", check=[sample], debug=debug)

        # 3b. Downblock downsample
        sample = batch_run(sample, partial(down_sample, down_block), offload=offload)
        clog.print(f"down_block{i} down_sample", check=[sample], debug=debug)


    # 3. middle sample = encoder.mid_block(sample)
    mid_block = encoder.mid_block

    # 2.a sample = mid_block.resnets[0](sample, None)
    resnet = mid_block.resnets[0]

    hidden_states = clone_list(sample)

    hidden_states = batch_run(
        hidden_states, partial(resnet_step1, resnet),
        norm=resnet.norm1,
        key="encoder-mid_block-resnet0.norm1",
        var_mean=var_mean,
        offload=offload,
    )

    hidden_states = batch_run(
        hidden_states, partial(resnet_step2, resnet),
        key="encoder-mid_block-resnet0.norm2",
        var_mean=var_mean,
        norm=resnet.norm2,
        offload=offload,
    )

    assert resnet.conv_shortcut is None
    sample = memory_efficient_add_scale(sample, hidden_states, resnet.output_scale_factor)
    clog.print("resnet0", check=[sample], debug=debug)
    # end resnet0

    # 2b. AttnProcessor2_0
    assert len(mid_block.attentions) == 1
    attn = mid_block.attentions[0]

    # sample = attn(sample, temb=None)
    hidden_states = clone_list(sample)
    hidden_states = batch_run(
        hidden_states, partial(attention, attn),
        norm=attn.group_norm,
        key="encoder-mid_block-attn",
        var_mean=var_mean,
        offload=offload,
    )
    sample = memory_efficient_add_scale(sample, hidden_states, attn.rescale_output_factor)
    clog.print("step1", check=[sample], debug=debug)
    # end AttnProcessor2_0

    for j, resnet in enumerate(mid_block.resnets[1:]):
        hidden_states = clone_list(sample)

        hidden_states = batch_run(
            hidden_states, partial(resnet_step1, resnet),
            norm=resnet.norm1,
            key=f"encoder-mid_block-resnet{j+1}.norm1",
            var_mean=var_mean,
            offload=offload,
        )

        hidden_states = batch_run(
            hidden_states, partial(resnet_step2, resnet),
            key=f"encoder-mid_block-resnet{j + 1}.norm2",
            var_mean=var_mean,
            norm=resnet.norm2,
            offload=offload,
        )

        if resnet.conv_shortcut is not None:
            sample = batch_run(sample, resnet.conv_shortcut, offload=offload)

        sample = memory_efficient_add_scale(sample, hidden_states, resnet.output_scale_factor)
        clog.print(f"mid_block-resnet{j}", check=[sample], debug=debug)


    # 4. post-process
    sample = batch_run(
        sample, lambda x: encoder.conv_out(encoder.conv_act(x)),
        norm=encoder.conv_norm_out,
        key="encoder-conv_norm_out",
        var_mean=var_mean,
        offload=offload,
    )
    clog.print(f"postprocess", check=[sample], debug=debug)

    return sample


def encode(z, vae, offload=False):
    h = encoder_encode(z, vae)
    moments = batch_run(h, vae.quant_conv, offload=offload)
    posterior = DiagonalGaussianDistribution(moments)
    return (posterior,)


def tile_encode(z, vae, var_mean={}, debug=False):
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    N, _, height, width = z.shape
    assert N == 1, "Only one image to upscale"
    in_bboxes, out_bboxes = split_tiles(height, width, vae_scale_factor, is_decoder=False)
    tiles = []
    for w_start, w_end, h_start, h_end in in_bboxes:
        tile = z[:, :, h_start:h_end, w_start:w_end].cpu()
        tiles.append(tile)

    if len(var_mean):
        # run sequentially each tile one after the other in order to avoid useless memory transfers
        if debug:
            print("Sequential run, no offload")
        encoded_tiles = []
        for tile in tiles:
            encoded_tile = encoder_encode(
                [tile], vae, offload=False, var_mean=var_mean
            )[0]
            # encoded_tile = batch_run([encoded_tile], vae.post_quant_conv, offload=False)[0]
            encoded_tiles.append(encoded_tile)

    else:
        offload = may_offload((height // vae_scale_factor) * (width // vae_scale_factor))
        if debug:
            print(f"Encoder offload: {offload}")
        encoded_tiles = encoder_encode(tiles, vae, offload=offload, var_mean=var_mean)  # [n_tiles, 3, tile_output_height, tile_output_width]

    result = torch.zeros(
        (N, encoded_tiles[0].shape[1], height // vae_scale_factor, width // vae_scale_factor),
        device=encoded_tiles[0].device,
        dtype=encoded_tiles[0].dtype,
        requires_grad=False
    )

    for tile, out_bbox, in_bbox in zip(encoded_tiles, out_bboxes, in_bboxes):
        w_start, w_end, h_start, h_end = out_bbox
        croped_tile = crop_valid_region(tile, in_bbox, out_bbox, vae_scale_factor, is_decoder=False)
        result[:, :, h_start:h_end, w_start:w_end] = croped_tile

    clog.print(f"tiles merged", check=[result], debug=debug)

    moments = vae.quant_conv(result.to("cuda"))
    clog.print(f"quant_conv", check=[moments], debug=debug)
    posterior = DiagonalGaussianDistribution(moments)
    return AutoencoderKLOutput(latent_dist=posterior)


if __name__ == "__main__":
    import torch.nn.functional as F
    latents_file = "latents.npy"
    latents_file = "latents_4x.npy"
    latents_file = "latents_emmanuel.npy"

    vae = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path="stabilityai/sd-vae-ft-mse",
    ).to(dtype=torch.float16, device="cuda")

    vae.requires_grad_(False)

    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor, do_convert_rgb=True)

    # # VAE Decoder
    # latents = torch.from_numpy(np.load(latents_file)).to(dtype=torch.float16)
    # print(f"Latent shape: {latents.shape}")  # [1, 4, 128, 256]
    #
    # # latents = F.upsample(latents, scale_factor=4)
    #
    # with torch.no_grad():
    #     # 1. Original
    #     # vae.enable_tiling(True)
    #     # decoded_img = vae.decode(latents.to("cuda"), return_dict=False)[0]
    #
    #     # 2. hack without tiles
    #     # decoded_img = decode(latents, vae)
    #
    #     # 3. hack with tiles
    #     decoded_img = tile_decode(latents, vae, var_mean={})
    #
    # print(f"Image shape: {decoded_img.shape}")  # [1, 3, 1024, 2048]
    # decoded_img = image_processor.postprocess(decoded_img)
    # decoded_img[0].save("result.png")

    # VAE encoder
    image = Image.open("emmanuel_upscaled.png")
    image = image_processor.preprocess(image, width=2048, height=2048).to(dtype=torch.float16, device="cuda")
    # image = F.upsample(image, scale_factor=4)
    # h = vae.encoder(image)
    h = tile_encode(image, vae)
    #np.save("encoder_latents.npy", h.cpu())

    clog.summary()
