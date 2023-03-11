import io
import numpy as np

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

op_colors = [[255,     0,    85],
    [255,     0,     0], 
    [255,    85,     0], 
    [255,   170,     0], 
    [255,   255,     0], 
    [170,   255,     0], 
     [85,   255,     0], 
      [0,   255,     0], 
    [255,     0,     0], 
      [0,   255,    85], 
      [0,   255,   170], 
      [0,   255,   255], 
      [0,   170,   255], 
      [0,    85,   255], 
      [0,     0,   255], 
    [255,     0,   170], 
    [170,     0,   255], 
    [255,     0,   255], 
     [85,     0,   255], 
      [0,     0,   255], 
      [0,     0,   255], 
      [0,     0,   255], 
      [0,   255,   255], 
      [0,   255,   255],
      [0,   255,   255]]

def overlay_kp(pimg, keypoints, vis_only=False): # visualize the 2D keypoints
    """
    pimg: PIL.Image
    keypoints: (N, 3) or (N, 2)
    vis_only: only visualize the visible keypoints
    """
    draw = ImageDraw.Draw(pimg)
    radius = 4
    num_keypoints = keypoints.shape[0]
    for k in range(num_keypoints):
        if vis_only and keypoints[k,2] == 0:
            continue
        leftUpPoint = (keypoints[k,0].item()-radius, keypoints[k,1].item()-radius)
        rightDownPoint = (keypoints[k,0].item()+radius, keypoints[k,1].item()+radius)
        color = op_colors[k % 25]
        draw.ellipse([leftUpPoint, rightDownPoint], fill=(color[0],color[1],color[2], 128), width=1)
    return pimg


def plot_3d_joints(img, pose_3D):
  joint_names_h36m = ['hip','right_up_leg','right_leg','right_foot','left_up_leg','left_leg', 'left_foot','spine1','neck', 'head','head-top','left-arm','left_forearm','left_hand','right_arm','right_forearm','right_hand']
  bones_h36m = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [9, 10], [8, 14], [14, 15], [15, 16], [8, 11], [11, 12], [12, 13]]
  img = np.array(img)
  fig = plt.figure(0)
  ax_2D = fig.add_subplot(121)
  ax_2D.imshow(img)
  # display 3D pose
  ax_3D = fig.add_subplot(122, projection='3d')
  for bone in bones_h36m:
      ax_3D.plot(pose_3D[bone,0], pose_3D[bone,2], pose_3D[bone,1], linewidth=3)
  ax_3D.invert_zaxis()

  img_buf = io.BytesIO()
  plt.savefig(img_buf, format='png')
  im = Image.open(img_buf)
  # im.show()
  # img_buf.close()
  return im