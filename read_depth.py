#The code in this file reads the depth from the depth map. 
#This depth map has been created using LIDAR and is specific to the KITTI dataset. The code in this file has been taken and adapted
# from "https://s3.eu-central-1.amazonaws.com/avg-kitti/devkit_scene_flow.zip" which is part of the official KITTI dataset website.

from PIL import Image
import numpy as np
import pandas as pd

def depth_read(filename):
    # loads depth map from png file
    # and returns it as a numpy array
    

    depth_png = np.array(Image.open(filename), dtype=int)
    # make sure to have a proper 16bit depth map here.. not 8bit!
    assert(np.max(depth_png) > 255)

    depth = depth_png.astype(float) / 256.   # in meters
    depth[depth_png == 0] = -1.
    return depth

if __name__ == "__main__":
    
    image_name="0000000010.png"
    depth_filepath_left="dataset/depth/image_02/"+image_name
    map=depth_read(depth_filepath_left)
    print(map)
    
    df = pd.DataFrame (map)
    filepath = 'depth_map.xlsx'
    df.to_excel(filepath, index=False)