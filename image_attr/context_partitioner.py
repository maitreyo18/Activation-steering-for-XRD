import numpy as np
from numpy.typing import NDArray
from typing import Optional, List
from abc import ABC, abstractmethod
from .utils import split_text
from PIL import Image
import torch

class BaseContextPartitioner(ABC):
    """
    A base class for partitioning a context into sources.

    Attributes:
        context (str):
            The context to partition.

    Methods:
        num_sources(self) -> int:
            Property. The number of sources within the context.
        split_context(self) -> None:
            Split the context into sources.
        get_source(self, index: int) -> str:
            Get a represention of the source corresponding to a given index.
        get_context(self, mask: Optional[NDArray] = None) -> str:
            Get a version of the context ablated according to the given mask.
        sources(self) -> List[str]:
            Property. A list of all sources within the context.
    """

    def __init__(self, context: str) -> None:
        self.context = context

    @property
    @abstractmethod
    def num_sources(self) -> int:
        """The number of sources."""

    @abstractmethod
    def split_context(self) -> None:
        """Split the context into sources."""

    @abstractmethod
    def get_source(self, index: int) -> str:
        """Get a represention of the source corresponding to a given index."""

    @abstractmethod
    def get_context(self, mask: Optional[NDArray] = None):
        """Get a version of the context ablated according to the given mask."""

    @property
    def sources(self) -> List[str]:
        """A list of all sources."""
        return [self.get_source(i) for i in range(self.num_sources)]

class SimpleImagePartitioner(BaseContextPartitioner):
    def __init__(self, image: Image, source_type: str = "patch", image_size: int = 224) -> None:
        super().__init__('')
        
        image = image.resize((image_size, image_size))      
        image = np.array(image)

        # Handle grayscale images by converting to RGB
        if len(image.shape) == 2:  # Grayscale image
            image = np.stack([image] * 3, axis=-1)  # Convert to RGB by repeating the channel
        elif image.shape[-1] == 4:  # RGBA image
            image = image[..., :3]  # Remove alpha channel

        self.img = np.transpose(image, (2,0,1))
        assert self.img.shape[0] == 3 and len(self.img.shape) == 3
        
        self.img_size = image_size
        self.p = 16 # patch size
        self.patch_index = int(self.img_size/self.p)
        self.in_chans = self.img.shape[0]
 
        self.source_type = source_type
        self._cache = {}
   
    def set_p(self, p):
        self.p = p
        self.patch_index = int(self.img_size/self.p)

    def patchify(self):
    
        assert self.img.shape[1] == self.img.shape[2] and self.img.shape[1] % self.p == 0
        h = w = self.img.shape[1] // self.p
        x = np.reshape(self.img, (self.in_chans, h, self.p, w, self.p))

        x = np.einsum('chpwq->hwpqc', x)
        x = np.reshape(x, (h * w, self.p ** 2 * self.in_chans)) 

        self._cache["parts"] = x
    
    def unpatchify(self, x):
     
        h = w = int(x.shape[0] ** .5)
        assert h * w == x.shape[0]
        x = np.reshape(x, (h, w, self.p, self.p, self.in_chans))
        x = np.einsum('hwpqc->chpwq', x)
        imgs = np.reshape(x, (self.in_chans, h*self.p, h*self.p))
        
        imgs = np.transpose(imgs, (1, 2, 0))
        imgs = Image.fromarray(imgs)
        return imgs

    @property
    def parts(self):
        if self._cache.get("parts") is None:
            self.patchify()
        return self._cache["parts"]

    @property
    def num_sources(self) -> int:
        return self.parts.shape[0]

    def get_source(self, index: int) -> np.array:
        return self.parts[index]

    def get_image(self, mask: Optional[NDArray] = None):
        if mask is None:
            mask = np.ones(self.num_sources, dtype=np.uint8)
        mask = mask.astype(np.uint8)
        parts = np.array(self.parts) * mask[:, None]
        return self.unpatchify(parts)
    
    def visualize_attr(self, attr_index: Optional[NDArray] = None, flip=False):
        assert attr_index.shape[0] != 0
        attr = np.zeros(self.num_sources, dtype=np.uint8)
        attr[attr_index] = 1
        
        if flip:
            attr = 1 - attr
        parts = np.array(self.parts) * attr[:, None]
        return self.unpatchify(parts)

    def split_context(self):
        return self.patchify()

    def get_context(self, mask: Optional[NDArray] = None):
        return self.get_image(mask)

    @property
    def sources(self) -> List[np.array]:
        """A list of all sources."""
        return [self.get_source(i) for i in range(self.num_sources)]

class SimpleContextPartitioner(BaseContextPartitioner):
    """
    A simple context partitioner that splits the context into sources based on
    a separator.
    """

    def __init__(self, context: str, source_type: str = "sentence") -> None:
        super().__init__(context)
        self.source_type = source_type
        self._cache = {}

    def split_context(self):
        """Split text into parts and cache the parts and separators."""
        parts, separators, _ = split_text(self.context, self.source_type)
        self._cache["parts"] = parts
        self._cache["separators"] = separators

    @property
    def parts(self):
        if self._cache.get("parts") is None:
            self.split_context()
        return self._cache["parts"]

    @property
    def separators(self):
        if self._cache.get("separators") is None:
            self.split_context()
        return self._cache["separators"]

    @property
    def num_sources(self) -> int:
        return len(self.parts)

    def get_source(self, index: int) -> str:
        return self.parts[index]

    def get_context(self, mask: Optional[NDArray] = None):
        if mask is None:
            mask = np.ones(self.num_sources, dtype=bool)
        separators = np.array(self.separators)[mask]
        parts = np.array(self.parts)[mask]
        context = ""
        for i, (separator, part) in enumerate(zip(separators, parts)):
            if i > 0:
                context += separator
            context += part
        return context
