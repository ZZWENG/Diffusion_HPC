"""
Down-sample a video and save the cropped frames to disk.

python process_video.py 43_07_video.mp4 data/43_07_fps3
"""
import cv2
import os
import glob
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('--path', required=True, help='')
parser.add_argument('--fps', required=True, help='')

def process(path, video_name, fps):
    out_dir = os.path.join(path, os.path.splitext(video_name)[0], 'images')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    count = 0
    vidcap = cv2.VideoCapture(os.path.join(path, video_name))
    video_fps = round(vidcap.get(cv2.CAP_PROP_FPS))
    print('Original video has fps', video_fps, '. Downsampling to fps', fps)
    
    t = time.time()
    while True:
        if fps != video_fps:
            vidcap.set(cv2.CAP_PROP_POS_MSEC, count*int(1000/fps))
        success, image = vidcap.read()
        if not success: break
        cv2.imwrite(os.path.join(out_dir, "frame%06d.jpg" % count), image)
        count += 1
    print('Saved {} frames in {} mins.'.format(count, (time.time() - t)/60))


if __name__ == '__main__':
    args = parser.parse_args()
    for video_path in glob.glob(os.path.join(args.path, '*.mkv')):
        print('Preprocessing', video_path)
        video_name = os.path.split(video_path)[1]
        process(args.path, video_name, int(args.fps))


