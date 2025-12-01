# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import math
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from lib.render.mesh import load_obj_mesh, compute_normal
from lib.render.camera import Camera
from lib.render.gl.geo_render import GeoRender
from lib.render.gl.color_render import ColorRender
import trimesh

import cv2
import os
import argparse
import matplotlib.pyplot as plt

width = 512
height = 512

def add_frame_number(img, frame_num):
    """Add frame number text to top right corner of image"""
    # Convert to uint8 if needed
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    
    # Make a copy to avoid modifying the original
    img = img.copy()
    text = f"t={(frame_num) / 7.5:.3f} ms"
    font = cv2.FONT_HERSHEY_COMPLEX  # Closest to Times New Roman/serif font
    font_scale = 1.0
    font_thickness = 1
    text_color = (0, 0, 0, 255)  # Black text with full alpha for RGBA
    bg_color = (255, 255, 255, 255)  # White background with full alpha for RGBA
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
    
    # Position in top right corner with padding
    padding = 30
    x = img.shape[1] - text_width - padding
    y = text_height + padding
    
    # Draw background rectangle with more padding
    cv2.rectangle(img, 
                  (x - 15, y - text_height - 15), 
                  (x + text_width + 15, y + baseline + 15), 
                  bg_color, -1)
    
    # Draw text
    cv2.putText(img, text, (x, y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
    
    return img

def make_rotate(rx, ry, rz):
    sinX = np.sin(rx)
    sinY = np.sin(ry)
    sinZ = np.sin(rz)

    cosX = np.cos(rx)
    cosY = np.cos(ry)
    cosZ = np.cos(rz)

    Rx = np.zeros((3, 3))
    Rx[0, 0] = 1.0
    Rx[1, 1] = cosX
    Rx[1, 2] = -sinX
    Rx[2, 1] = sinX
    Rx[2, 2] = cosX

    Ry = np.zeros((3, 3))
    Ry[0, 0] = cosY
    Ry[0, 2] = sinY
    Ry[1, 1] = 1.0
    Ry[2, 0] = -sinY
    Ry[2, 2] = cosY

    Rz = np.zeros((3, 3))
    Rz[0, 0] = cosZ
    Rz[0, 1] = -sinZ
    Rz[1, 0] = sinZ
    Rz[1, 1] = cosZ
    Rz[2, 2] = 1.0

    R = np.matmul(np.matmul(Rz, Ry), Rx)
    return R

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file_dir', type=str, required=True)
parser.add_argument('-ww', '--width', type=int, default=512)
parser.add_argument('-hh', '--height', type=int, default=512)
parser.add_argument('-g', '--geo_render', action='store_true', help='default is normal rendering')
parser.add_argument('-n', '--num', type=int, default=100)
parser.add_argument('-s', '--start', type=int, default=0)
parser.add_argument('-d', '--debug', action='store_true', help='show debug plots of first frame')


args = parser.parse_args()

if args.geo_render:
    renderer = GeoRender(width=args.width, height=args.height)
else:
    renderer = ColorRender(width=args.width, height=args.height)
    
cam = Camera(width=1.0, height=args.height/args.width)
cam.ortho_ratio = 1.2
cam.near = -100
cam.far = 10

obj_files = []
for (root,dirs,files) in os.walk(args.file_dir, topdown=True): 
    for file in files:
        if '.obj' in file:
            obj_files.append(os.path.join(root, file))
            
# NEW: sort and cut file list
obj_files.sort()
obj_files = obj_files[args.start:args.num+args.start]
print(obj_files)


''' Render bottom view'''
for i, obj_path in enumerate(obj_files):
    print(obj_path)
    obj_file = obj_path.split('/')[-1]
    obj_fold = obj_path.split('/')[-2]
    obj_root = obj_path.replace(obj_file, '')
    file_name = obj_file[:-4]

    if not os.path.exists(obj_path):
        continue
    mesh = trimesh.load(obj_path)
    vertices = mesh.vertices
    faces = mesh.faces

    '''bottom view'''
    R = make_rotate(math.radians(270), 0, 0)
    vertices = np.matmul(vertices, R.T)
    vertices = vertices*0.5

    normals = compute_normal(vertices, faces)

    if args.geo_render:
        renderer.set_mesh(vertices, faces, normals, faces)
    else:
        renderer.set_mesh(vertices, faces, 0.5 * normals + 0.5, faces)

    cam.center = np.array([0, 0, 0])
    cam.eye = np.array([2.0 * math.sin(math.radians(0)), 0, 2.0 * math.cos(math.radians(0))]) + cam.center

    renderer.set_camera(cam)
    renderer.display()
    img = renderer.get_color(0)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
    
    img = add_frame_number(img, i)
    
    # Debug: show first frame
    if args.debug and i == 0:
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA))
        plt.title(f'Bottom View - Frame {i}')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(obj_root, 'debug_bottom.png'), dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Debug: Image shape: {img.shape}, dtype: {img.dtype}, min: {img.min()}, max: {img.max()}")
    
    
    cv2.imwrite(os.path.join(obj_root, 'bot_%04d.png' % i), img)

file_name.replace(" ", "_")
file_name = "_".join(file_name.split())
print(file_name)
cmd = 'ffmpeg -framerate 30 -i ' + obj_root + '/bot_%04d.png -vcodec libx264 -y -pix_fmt yuv420p -refs 16 ' + os.path.join(obj_root, str(obj_fold) + '_bottom.mp4')
os.system(cmd)
#cmd = 'rm %s/rot_*.png' % obj_root
#os.system(cmd)


''' Render 90° side view'''
for i, obj_path in enumerate(obj_files):
    print(obj_path)
    obj_file = obj_path.split('/')[-1]
    obj_fold = obj_path.split('/')[-2]
    obj_root = obj_path.replace(obj_file, '')
    file_name = obj_file[:-4]

    if not os.path.exists(obj_path):
        continue
    mesh = trimesh.load(obj_path)
    vertices = mesh.vertices
    faces = mesh.faces

    '''90° side view'''
    R = make_rotate(0, math.radians(270), 0)
    vertices = np.matmul(vertices, R.T)
    vertices = vertices*0.5

    normals = compute_normal(vertices, faces)

    if args.geo_render:
        renderer.set_mesh(vertices, faces, normals, faces)
    else:
        renderer.set_mesh(vertices, faces, 0.5 * normals + 0.5, faces)

    cam.center = np.array([0, 0, 0])
    cam.eye = np.array([2.0 * math.sin(math.radians(0)), 0, 2.0 * math.cos(math.radians(0))]) + cam.center

    renderer.set_camera(cam)
    renderer.display()
    img = renderer.get_color(0)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
    
    img = add_frame_number(img, i)
    
    cv2.imwrite(os.path.join(obj_root, 'side_%04d.png' % i), img)

file_name.replace(" ", "_")
file_name = "_".join(file_name.split())
print(file_name)
cmd = 'ffmpeg -framerate 30 -i ' + obj_root + '/side_%04d.png -vcodec libx264 -y -pix_fmt yuv420p -refs 16 ' + os.path.join(obj_root, str(obj_fold) + '_90.mp4')
os.system(cmd)
#cmd = 'rm %s/rot_*.png' % obj_root
#os.system(cmd)


''' Render 0° side view'''
for i, obj_path in enumerate(obj_files):
    print(obj_path)
    obj_file = obj_path.split('/')[-1]
    obj_fold = obj_path.split('/')[-2]
    obj_root = obj_path.replace(obj_file,'')
    file_name = obj_file[:-4]

    if not os.path.exists(obj_path):
        continue    
    mesh = trimesh.load(obj_path)
    vertices = mesh.vertices
    faces = mesh.faces

    '''0° side view'''
    R = make_rotate(0, 0, 0)
    vertices = np.matmul(vertices, R.T)
    vertices = vertices*0.5

    normals = compute_normal(vertices, faces)
    
    if args.geo_render:
        renderer.set_mesh(vertices, faces, normals, faces)
    else:
        renderer.set_mesh(vertices, faces, 0.5*normals+0.5, faces) 

    cam.center = np.array([0, 0, 0])
    cam.eye = np.array([2.0*math.sin(math.radians(0)), 0, 2.0*math.cos(math.radians(0))]) + cam.center

    renderer.set_camera(cam)
    renderer.display()
    img = renderer.get_color(0)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
    
    img = add_frame_number(img, i)

    cv2.imwrite(os.path.join(obj_root, 'front_%04d.png' % i), img)

file_name.replace(" ", "_")
file_name = "_".join(file_name.split())
print(file_name)
cmd = 'ffmpeg -framerate 30 -i ' + obj_root + '/front_%04d.png -vcodec libx264 -y -pix_fmt yuv420p -refs 16 ' + os.path.join(obj_root, str(obj_fold) + '_0.mp4')
os.system(cmd)
#cmd = 'rm %s/rot_*.png' % obj_root
#os.system(cmd)