import math
from typing import Dict, List, Optional, Union

import numpy as np

from transformers.image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
    infer_channel_dimension_format,
    is_scaled_image,
    make_flat_list_of_images,
    make_list_of_images,
    load_image,
    valid_images,
    to_numpy_array,
)
from transformers.image_processing_utils import BatchFeature, BaseImageProcessor
from transformers.image_transforms import (
    convert_to_rgb,
    resize,
    to_channel_dimension_format,    
)
from transformers.utils import TensorType, logging
from transformers.models.auto import AutoImageProcessor


logger = logging.get_logger(__name__)


def smart_resize(
    height: int, width: int, factor: int = 32, min_pixels: int = 16 * 16 * 4 * 64, max_pixels: int = 16 * 16 * 4 * 2048
):
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")

    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    
    assert h_bar % factor == 0, f"h_bar:{h_bar} must be divisible by factor:{factor}"
    assert w_bar % factor == 0, f"w_bar:{w_bar} must be divisible by factor:{factor}"
    return h_bar, w_bar


def resize_small_images(image, min_size=32):
    # `image` is a numpy array here (see to_numpy_array in _preprocess);
    # get_image_size returns (height, width).
    height, width = get_image_size(image)
    factor = 1.
    if height < min_size:
        factor = min_size / height
    if width < min_size:
        factor = max(factor, min_size / width)

    if factor > 1.:
        new_height = round(height * factor + 0.5)
        new_width = round(width * factor + 0.5)
        # transformers' numpy resize takes size=(height, width); matches the
        # resize() call in _preprocess and avoids PIL's (width, height) ordering.
        image = resize(image, size=(new_height, new_width))
    return image


class AXVEImageProcessor(BaseImageProcessor):

    model_input_names = ["pixel_values_images", "image_grid_hw"]

    def __init__(
        self,
        do_resize: bool = True,
        size: Dict[str, int] = None,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = True,
        min_pixels: int = 16 * 16 * 4 * 64,
        max_pixels: int = 16777216,
        patch_size: int = 16,
        merge_size: int = 2,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        if size is not None:
            print("size will be ignored. use min_pixels and max_pixels instead.")
        
        if min_pixels is None or max_pixels is None:
            raise ValueError("min_pixels and max_pixels must be provided")
            
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.size = size

        self.do_resize = do_resize
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_STANDARD_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_STANDARD_STD

        self.patch_size = patch_size
        self.merge_size = merge_size
        self.do_convert_rgb = do_convert_rgb

    def get_size_of_processed_image(self, image): 
        factor = self.patch_size * self.merge_size
        image = resize_small_images(image, min_size=factor)
        input_data_format = infer_channel_dimension_format(image)
        height, width = get_image_size(image, channel_dim=input_data_format)
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=factor,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        return resized_height, resized_width

    def _preprocess(
        self,
        images: ImageInput,
        do_resize: Optional[bool] = None,
        size: Optional[dict[str, int]] = None,
        resample: PILImageResampling = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        do_convert_rgb: Optional[bool] = None,
        patch_size: Optional[int] = None,
        merge_size: Optional[int] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        images = make_list_of_images(images)

        if do_convert_rgb:
            images = [convert_to_rgb(image) for image in images]

        # All transformations expect numpy arrays.
        images = [to_numpy_array(image) for image in images]

        if do_rescale and is_scaled_image(images[0]):
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )
        if input_data_format is None:
            # We assume that all images have the same channel dimension format.
            input_data_format = infer_channel_dimension_format(images[0])
      
        patches_per_turn = []
        hw_grids_per_turn = []
        for image in images:
            if do_resize:
                image = resize_small_images(image, min_size=patch_size * merge_size)
                height, width = get_image_size(image, channel_dim=input_data_format)
                resized_height, resized_width = smart_resize(
                    height,
                    width,
                    factor=patch_size * merge_size,
                    min_pixels=self.min_pixels,
                    max_pixels=self.max_pixels,
                )
                image = resize(
                    image, size=(resized_height, resized_width), resample=resample, input_data_format=input_data_format
                )

            if do_rescale:
                image = self.rescale(image, scale=rescale_factor, input_data_format=input_data_format)

            if do_normalize:
                image = self.normalize(
                    image=image, mean=image_mean, std=image_std, input_data_format=input_data_format
                )

            image = to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)
            
            channel = image.shape[0]
            grid_h, grid_w = resized_height // self.patch_size, resized_width // self.patch_size
            patches = image.reshape(
                1,
                channel,
                grid_h // self.merge_size,
                self.merge_size,
                self.patch_size,
                grid_w // self.merge_size,
                self.merge_size,
                self.patch_size,
            )
            patches = patches.transpose(0, 2, 5, 3, 6, 1, 4, 7)
            patches = patches.reshape(-1, channel, patch_size, patch_size)
            patches = patches.reshape(
                grid_h * grid_w, channel * self.patch_size * self.patch_size
            )     

            hw_grid = (grid_h, grid_w)

            assert patches.shape[0] == hw_grid[0] * hw_grid[1], f"patches.shape[0]: {patches.shape[0]} != hw_grid[0] * hw_grid[1]: {hw_grid[0] * hw_grid[1]}"

            patches_per_turn.append(patches)
            hw_grids_per_turn.append(hw_grid)
        
        return np.concatenate(patches_per_turn, axis=0), hw_grids_per_turn

    def preprocess(
        self,
        images: ImageInput,
        do_resize: Optional[bool] = None,
        size: Optional[dict[str, int]] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        resample: Optional[PILImageResampling] = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        do_convert_rgb: Optional[bool] = None,
        patch_size: Optional[int] = None,
        merge_size: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        min_pixels = min_pixels if min_pixels is not None else self.min_pixels
        max_pixels = max_pixels if max_pixels is not None else self.max_pixels

        if size is not None:
            raise ValueError("size is not supported. use min_pixels and max_pixels instead")

        do_resize = do_resize if do_resize is not None else self.do_resize

        resample = resample if resample is not None else self.resample
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        patch_size = patch_size if patch_size is not None else self.patch_size
        merge_size = merge_size if merge_size is not None else self.merge_size
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb

        if images is not None:
            images = self.fetch_images(images)
            images = make_flat_list_of_images(images)

        if images is not None and not valid_images(images):
            raise ValueError("Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, or torch.Tensor")

        data = {}
        if images is not None:
            pixel_values, grid_hws = [], []            
            for image in images:
                patches, grid_hw = self._preprocess(
                    image,
                    do_resize=do_resize,
                    size=size,
                    resample=resample,
                    do_rescale=do_rescale,
                    rescale_factor=rescale_factor,
                    do_normalize=do_normalize,
                    image_mean=image_mean,
                    image_std=image_std,
                    patch_size=patch_size,
                    merge_size=merge_size,
                    data_format=data_format,
                    do_convert_rgb=do_convert_rgb,
                    input_data_format=input_data_format,
                )
                pixel_values.extend(patches)
                grid_hws += grid_hw
            pixel_values = np.array(pixel_values)
            grid_hws = np.array(grid_hws)
            data.update({"pixel_values_images": pixel_values, "image_grid_hw": grid_hws})

        return BatchFeature(data=data, tensor_type=return_tensors)

    def get_number_of_image_patches(self, height: int, width: int, images_kwargs=None):

        images_kwargs = images_kwargs or {}
        min_pixels = images_kwargs.get("min_pixels", self.min_pixels)
        max_pixels = images_kwargs.get("max_pixels", self.max_pixels)
        patch_size = images_kwargs.get("patch_size", self.patch_size)
        merge_size = images_kwargs.get("merge_size", self.merge_size)

        factor = patch_size * merge_size
        resized_height, resized_width = smart_resize(
            height, width, factor, min_pixels=min_pixels, max_pixels=max_pixels
        )
        grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
        return grid_h * grid_w    

    def fetch_images(self, image_url_or_urls: Union[str, list[str], list[list[str]]]):

        if isinstance(image_url_or_urls, list):
            return [self.fetch_images(x) for x in image_url_or_urls]
        elif isinstance(image_url_or_urls, str):
            return load_image(image_url_or_urls)
        elif valid_images(image_url_or_urls):
            return image_url_or_urls
        else:
            raise TypeError(f"only a single or a list of entries is supported but got type={type(image_url_or_urls)}")