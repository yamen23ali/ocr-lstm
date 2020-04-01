import torch
from PIL import Image

class ImageTensorPadding(object):
    """ 
        Pad 2D image to a specific height and width

        Args:
            padded_image_height (int): The new image height
            padded_image_width (int): The new image width

        Attributes:
            padded_image_height (int): The new image height
            padded_image_width (int): The new image width
    """

    def __init__(self, padded_image_height, padded_image_width):

        self.padded_image_height = padded_image_height
        self.padded_image_width = padded_image_width

    def __call__(self, image):
        """
        Apply the transformation

        Args:
            image (:obj: `Image`): The image to apply the transformation on

        Returns:
            (:obj: `Image`): The new padded image
        """
        padded_image = torch.ones(1, self.padded_image_height, self.padded_image_width)
        padded_image[0, 0:image.shape[1], 0:image.shape[2]] = image

        return padded_image

class ImageThumbnail(object):

    """ 
        Resize an image by getting its thumbnail in order to preserve the aspect ratio

        Args:
            thumbnail_height (int): The new image height
            thumbnail_width (int): The new image width

        Attributes:
            thumbnail_height (int): The new image height
            thumbnail_width (int): The new image width
    """

    def __init__(self, thumbnail_height, thumbnail_width):

        self.thumbnail_height = thumbnail_height
        self.thumbnail_width = thumbnail_width

    def __call__(self, image):
        """
        Apply the transformation

        Args:
            image (:obj: `Image`): The image to apply the transformation on

        Returns:
            (:obj: `Image`): The new thumbnail image
        """
        image.thumbnail((self.thumbnail_width, self.thumbnail_height), Image.ANTIALIAS)

        return image


class UnfoldImage(object):
    """ 
        Unfold 2D image ( i.e. Convert a text line image into frames)

        Args:
            unfold_dimension (int): The dimension to do the unfolding on
            frame_size (int): The size of each frame during the unfolding

        Attributes:
            unfold_dimension (int): The dimension to do the unfolding on
            frame_size (int): The size of each frame during the unfolding
    """

    def __init__(self, unfold_dimension, frame_size):
        self.unfold_dimension = unfold_dimension
        self.frame_size = frame_size

    def __call__(self, image):
        """
        Apply the transformation

        Args:
            image (:obj: `Image`): The image to apply the transformation on

        Returns:
            (:obj: `Image`): The unfolded image frames
        """

        # Image[0] jsut to remove the extra dimension we get from ToTensor() transformation
        frames = image[0].unfold(self.unfold_dimension, self.frame_size, self.frame_size)

        # Put it in this shape : (frames, frame_height, frame_width)
        frames = frames.permute(1, 0, 2) 

        # Convert each frame ( 2D image ) into one vector
        # Here the shape becomes (frames, frame_height * frame_width)
        frames = frames.reshape(frames.shape[0], -1)

        return frames

class FixLineImage(object):
    """
    Fix text line image that have height > width 
    """
    def __init__(self):
        pass

    def __call__(self, image):
        """
        Apply the transformation

        Args:
            image (:obj: `Image`): The image to apply the transformation on

        Returns:
            (:obj: `Image`): The fixed image
        """
        size = image.size
        if size[0] < size[1]:
            im = Image.new('1',(size[1] + 1, size[1]),1)
            im.paste(image, (0,0))
            image = im

        return image