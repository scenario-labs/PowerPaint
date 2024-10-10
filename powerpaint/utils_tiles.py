import math
import torch


def get_rcmd_dec_tsize():
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties("cuda").total_memory // 2**20
        if   total_memory > 30*1000: DECODER_TILE_SIZE = 256
        elif total_memory > 16*1000: DECODER_TILE_SIZE = 192
        elif total_memory > 12*1000: DECODER_TILE_SIZE = 128
        elif total_memory >  8*1000: DECODER_TILE_SIZE = 96
        else:                        DECODER_TILE_SIZE = 64
    else:                            DECODER_TILE_SIZE = 64
    return DECODER_TILE_SIZE


def get_rcmd_enc_tsize():
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties("cuda").total_memory // 2**20
        if   total_memory > 16*1000: ENCODER_TILE_SIZE = 3072
        elif total_memory > 12*1000: ENCODER_TILE_SIZE = 2048
        elif total_memory >  8*1000: ENCODER_TILE_SIZE = 1536
        else:                        ENCODER_TILE_SIZE = 960
    else:                            ENCODER_TILE_SIZE = 512
    return ENCODER_TILE_SIZE


def get_best_tile_size(lowerbound, upperbound):
    """
    Get the best tile size for GPU memory
    """
    divider = 32
    while divider >= 2:
        remainer = lowerbound % divider
        if remainer == 0:
            return lowerbound
        candidate = lowerbound - remainer + divider
        if candidate <= upperbound:
            return candidate
        divider //= 2
    return lowerbound


def find_best_tile_dim(dim, pad, is_decoder=True):
    max_tile_size = get_rcmd_dec_tsize() if is_decoder else get_rcmd_enc_tsize()
    num_tiles = math.ceil((dim - 2 * pad) / max_tile_size)
    num_tiles = max(num_tiles, 1)
    real_tile_size = math.ceil((dim - 2 * pad) / num_tiles)
    real_tile_size = get_best_tile_size(real_tile_size, max_tile_size)
    return real_tile_size, num_tiles


def split_tiles(height, width, vae_scale_factor, is_decoder=True):
    """
    Tool function to split into tiles
    @param h: height of the latents
    @param w: width of the latents
    @return: tile_input_bboxes, tile_output_bboxes
    """
    tile_input_bboxes, tile_output_bboxes = [], []
    if is_decoder:
        pad = 11
    else:
        pad = 32
    tile_height, num_height_tiles = find_best_tile_dim(height, pad, is_decoder=is_decoder)  # 96
    tile_width, num_width_tiles = find_best_tile_dim(width, pad, is_decoder=is_decoder)  # 112
    print(f"VAE tile size(hxw): {tile_height}x{tile_width}")
    print(f"Num VAE tiles: {num_height_tiles}x{num_width_tiles}={num_width_tiles*num_height_tiles}")
    for i in range(num_height_tiles):
        for j in range(num_width_tiles):
            # bbox: [x1, x2, y1, y2]
            # the padding is is unnecessary for image borders. So we directly start from (32, 32)
            input_bbox = [
                pad + j * tile_width,
                min(pad + (j + 1) * tile_width, width),
                pad + i * tile_height,
                min(pad + (i + 1) * tile_height, height),
            ]

            # if the output bbox is close to the image boundary, we extend it to the image boundary
            output_bbox = [
                input_bbox[0] if input_bbox[0] > pad else 0,
                input_bbox[1] if input_bbox[1] < width - pad else width,
                input_bbox[2] if input_bbox[2] > pad else 0,
                input_bbox[3] if input_bbox[3] < height - pad else height,
            ]

            # scale to get the final output bbox
            output_bbox = [x * vae_scale_factor if is_decoder else x // vae_scale_factor for x in output_bbox]
            tile_output_bboxes.append(output_bbox)

            # indistinguishable expand the input bbox by pad pixels
            tile_input_bboxes.append([
                max(0, input_bbox[0] - pad),
                min(width, input_bbox[1] + pad),
                max(0, input_bbox[2] - pad),
                min(height, input_bbox[3] + pad),
            ])

    return tile_input_bboxes, tile_output_bboxes


def crop_valid_region(x, input_bbox, target_bbox, vae_scale_factor, is_decoder=True):
    """
    Crop the valid region from the tile
    @param x: input tile
    @param input_bbox: original input bounding box
    @param target_bbox: output bounding box
    @param scale: scale factor
    @return: cropped tile
    """
    padded_bbox = [i * vae_scale_factor if is_decoder else i // vae_scale_factor for i in input_bbox]
    margin = [target_bbox[i] - padded_bbox[i] for i in range(4)]
    return x[:, :, margin[2]:x.size(2)+margin[3], margin[0]:x.size(3)+margin[1]]


def may_offload(tile_area):
    """
    # Measured VRAM occupations on A10 #
    # image size | latent size | vram (with offload) | vram (without offload)
    # 2000       | 256         | 3.6 (22s)           | 8.63 (5s)
    # 4000       | 512         | 7.16 (50s)          | 19.7 (11s)
    # 8000       | 1024        | 7.16 (298s)         | OOM on A10

    The offload parameter decides if the tiles should be returned back to CPU while the next tile is run
    does not depend if var_mean is known or not

    :return:
        bool
    """
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties("cuda").total_memory // 2**20
        print(f"Total memory available: {total_memory}")
        if tile_area > 512 * 512 and total_memory < 30 * 1000:
            print("Activating tile offload")
            return True
        return False

    return True
