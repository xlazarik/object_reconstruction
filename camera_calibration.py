import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS


class ImageIntrinsicMetadata:

    def __init__(self, filepath):
        self.exif_data = {}
        self.read_exif_from_img(filepath)

    def extract_exif_data(self, raw_exif_data):
        for tag_id in raw_exif_data:
            tag = TAGS.get(tag_id, tag_id)
            content = raw_exif_data.get(tag_id)
            self.exif_data[tag] = content
            # print(f'{tag:25}: {content}')

    def read_exif_from_img(self, filepath):
        with Image.open(filepath) as im:
            exif = im.getexif()
            self.extract_exif_data(exif)
            self.extract_exif_data(exif.get_ifd(0x8769))

    def get_camera_focal_length(self):
        # Assuming a standard 35mm sensor size
        # sensor_size_mm = 35.0

        # focal length in mm
        focal_length_mm = self.exif_data['FocalLengthIn35mmFilm']
        # focal_length_mm = self.exif_data['FocalLength']

        return focal_length_mm

    def get_camera_principal_point(self):
        principal_point_x = self.exif_data['ExifImageWidth'] / 2
        principal_point_y = self.exif_data['ExifImageHeight'] / 2

        return (principal_point_x, principal_point_y)

    def get_intrinsic_matrix(self):
        fp_x = self.exif_data['ExifImageWidth'] * self.get_camera_focal_length() / 36.0
        # fp_y = self.exif_data['ExifImageHeight'] * self.get_camera_focal_length() / 24.0
        o_x, o_y = self.get_camera_principal_point()

        intrinsic_matrix = np.array(
            [[fp_x, 0, o_x],
             [0, fp_x, o_y],
             [0, 0, 1]])
        return intrinsic_matrix


# TODO: perform chessboard camera calibration
class ImageIntrinsicManual:

    def __init__(self, calibration_photo_directory_path):
        self.focal_length = (0.0, 0.0)
        self.principal_point = (0.0, 0.0)
        self.calibrate_camera(calibration_photo_directory_path)

    def calibrate_camera(self):
        pass

    def get_camera_focal_lengths(self):
        pass

    def get_camera_principal_point(self):
        pass

    def get_intrinsic_matrix(self):
        fp_x, fp_y = self.get_camera_focal_lengths()
        o_x, o_y = self.get_camera_principal_point()

        intrinsic_matrix = np.array(
            [[fp_x, 0, o_x],
             [0, fp_x, o_y],
             [0, 0, 1]])
        return intrinsic_matrix


###############################################################################
# TESTING THE FUNCTIONALITY
###############################################################################
if __name__ == "__main__":
    filepath = 'C:/Users/vladi/Desktop/COLMAP-3.8-windows-cuda/WORKSPACE_eydrops_box/dense/0/leftIMG.jpg'

    img_data = ImageIntrinsicMetadata(filepath)
    f = img_data.get_camera_focal_length()
    ox, oy = img_data.get_camera_principal_point()

    print(img_data.get_intrinsic_matrix())
