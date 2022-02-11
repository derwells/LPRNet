import os


class RawLicensePlate():
    def __init__(
        self, 
        fname
    ):
        self.fname = fname

        self.area = None
        self.tilt = None
        self.bbox = None
        self.vert = None
        self.lpn = None
        self.brightness = None
        self.blur = None
        self.extension = None

        self.clean()
    
    def clean(self):
        self.fname, self.extension = os.path.splitext(
            self.fname
        )
        (
            self.area,
            self.tilt,
            self.bbox,
            self.vert,
            self.lpn,
            self.brightness,
            self.blur
        ) = self.fname.split('-')

        self.tilt = self.clean_coords(self.tilt)
        self.bbox = self.clean_coords(self.bbox)
        self.vert = self.clean_coords(self.vert)

    def clean_coords(self, coords):
        coords_split = coords.split("_")
        cleaned = []
        for coord in coords_split:
            tmp = coord.split("&")
            tmp = map(int, tmp)
            tmp = list(tmp)
            cleaned.append(tmp)

        return cleaned


class CroppedLicensePlate():
    def __init__(
        self, 
        fname
    ):
        self.fname = fname

        self.idx = None
        self.lpn = None

        self.extension = None

        self.clean()

    def clean(self):
        name, self.extension = os.path.splitext(
            self.fname
        )
        idx, lpn = name.split('-')
        self.idx = int(idx)
        
        lpn = lpn.split('_')
        self.lpn = list(
            map(int, lpn)
        )
