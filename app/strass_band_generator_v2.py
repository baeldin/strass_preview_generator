#!/usr/bin/env python
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import numpy as np
import copy
from PIL import Image
import time
import os
import subprocess
import pandas as pd


alphabet_strings = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']




class Model:
    def __init__(self, name, band_coordinates, band_height, band_columns, band_glow, band_raw):
        self.name = name
        self.band_coordinates = band_coordinates
        self.band_height = int(float(band_height) / 5.5 * 5) if band_raw else band_height
        self.band_columns = band_columns
        self.add_glow = band_glow
        self.is_raw = band_raw

models = [
    #Model("Damenboerse_ohneStrass_fixed", [ 49, 538],  66, 72),
    #      Artikel                 X    Y     H    W    glow  raw
    # Model("IN111520",             [211, 580], 121, 40,  True, False), # OK
    # Model("IN112150",             [125, 699],  63, 87,  True, False), # OK
    # Model("IN116490",             [171, 440], 103, 50,  True, False), # OK
    # Model("IN116585",             [176, 457],  70, 76,  True, False), # OK
    # Model("IN118102",             [384, 732],  50, 74,  True, False), # OK
    # Model("IN118103",             [280, 533],  44, 84,  True, False), # OK
    # Model("IN118104",             [337, 863],  42, 84,  True, False), # OK
    # Model("void",                 [ 50,  50], 130, 70, False,  True),
    # Model("void2",                [ 50,  50], 130, 80, False,  True),
    Model("raw",                  [ 50,  50], 130, 87,  True,  True),
    Model("raw_glow",             [ 50,  50], 130, 87, False,  True)
    ]

# stone_type_list = ["1088_DenimBlue266_enh"]
# stone_type_list = ["1088_001Crystal_R"]

stone_type_list = ["2078I_001LTCHZ_FV_PL", "1088_BlackDiamond215", "1088_001Crystal_R", "1088_DenimBlue266", "1088_DenimBlue266_enh", "1088_Greige284", "1088_Rose209"]
# stone_type_list = ["1088_BlackDiamond215", "1088_001Crystal_R", "1088_DenimBlue266", "1088_DenimBlue266_enh", "1088_Greige284", "1088_Rose209"]
# stone_type_list = ["2038_215_FO_15_FV_PL", "2078_001_FO_10_FV_MI", "2038_DenimBlue_266", "2038_284_FO_15_FV_PL", "2038_209_FO_15_FV_PL"]
band_bg_colors = ["002", "007", "045", "066"]
wallet_colors = ['002', '007']
# wallet_colors = ['002', '007', '045', '066']
# band_bg_colors = ["002"]
adjust_letter_spacing = True

def sRGB_to_linear(img):
    wasImage = False
    if isinstance(img, Image.Image):
        data = np.array(img).astype(float) / 255.
        wasImage = True
    else:
        data = img.astype(float) / 255.
    data = np.where(data <= 0.04045, data / 12.92, np.power((data + 0.055) / 1.055, 2.4))
    if wasImage:
        return Image.fromarray((data * 255.).astype(np.uint8))
    else:
        return (data * 255.).astype(np.uint8)


def linear_to_sRGB(img):
    wasImage = False
    if isinstance(img, Image.Image):
        data = np.array(img).astype(float) / 255.
        wasImage = True
    else:
        data = img.astype(float) / 255.
    data = np.where(data <= 0.0031308, data * 12.92, 1.055 * np.power(data, 1. / 2.4) - 0.055)
    if wasImage:
        return Image.fromarray((data * 255.).astype(np.uint8))
    else:
        return (data * 255.).astype(np.uint8)


def draw_debug_halo(img):
    nx, ny = img.size
    for yy in [0, ny-1]:
        for xx in range(nx):
            img.putpixel((xx, yy), (0, 255, 0, 255))
    for xx in [0, nx-1]:
        for yy in range(ny):
            img.putpixel((xx, yy), (0, 255, 0, 255))
    return img


def make_debug_square(N, col=(0, 255, 0, 255)):
    debug_square_arr = np.zeros((N, N, 4), dtype=np.uint8)
    for ic, c in enumerate(col):
        debug_square_arr[:, :, ic] = c
    return Image.fromarray(debug_square_arr)


def make_debug_cross(N, col=(0, 255, 0, 255)):
    debug_cross_arr = np.zeros((N, N, 4), dtype=np.uint8)
    t1 = int(N/2)
    t2 = int(N/2+1)
    for ic, c in enumerate(col):
        debug_cross_arr[t1:t2, :, ic] = c
        debug_cross_arr[:, t1:t2, ic] = c
    return Image.fromarray(debug_cross_arr)


def get_parent_directory():
    full_path = __file__
    return full_path.replace("app/strass_band_generator_v2.py", "")


def download_img(urlstring, local_img_path):
    print("Downloading ", urlstring, " to ", local_img_path)
    urllib.request.urlretrieve(urlstring, local_img_path)
    time.sleep(1)
    print(os.path.isfile(local_img_path))
    img = sRGB_to_linear(Image.open(local_img_path))
    return img


def get_img(img_path):
    parent_dir = get_parent_directory()
    local_img_path = parent_dir + img_path
    print("check for {:s}".format(local_img_path))
    OK = False
    if os.path.isfile(local_img_path):
        print("Found, opening {:s}".format(local_img_path))
        try:
            img = sRGB_to_linear(Image.open(local_img_path))
            OK = True
        except:
            raise
            print("Opening {:s} failed, corrupt file? Falling back to download option.".format(local_img_path))
            OK = False
    if not OK or not img:
        print("{:s} not found, downloading.".format(local_img_path))
        img_url = "https://raw.githubusercontent.com/baeldin/strass_preview_generator/main/" + img_path
        img = download_img(img_url, local_img_path)
    return img, local_img_path


def split_shine(img_shine, sizes = [30, 15, 7]):
    img_shine_ = copy.copy(img_shine)
    shine = []
    for ii in range(4):
        for jj in range(5):
            shine.append(img_shine.crop((100*ii, 100*jj, 100*(ii+1), 100*(jj+1))))
    shine_all = {}
    for size in sizes:
        shine_all[size] = []
        for idx, sh in enumerate(shine):
            shine_resized = sh.resize((size, size), Image.BICUBIC)
            shine_all[size].append(sRGB_to_linear(shine_resized))
    return shine_all, sizes


def get_alphabet_coordinates():
    # req = urllib.request.urlopen("https://raw.githubusercontent.com/baeldin/strass_preview_generator/main/Alphabet.asc")
    df = pd.read_csv("../Alphabet.asc", delimiter=";", header=None, skiprows=3)
    leter_indices = {
        'A': [  0,   9], 'B': [ 47,  59], 'C': [ 10,  20], 'D': [151, 162], 'E': [163, 172], 'F': [173, 180], 
        'G': [181, 193], 'H': [194, 204], 'I': [ 60,  64], 'J': [ 21,  27], 'K': [ 94, 103], 'L': [269, 275], 
        'M': [205, 219], 'N': [220, 233], 'O': [234, 245], 'P': [246, 255], 'Q': [256, 268], 'R': [ 82,  93], 
        'S': [ 28,  38], 'T': [ 39,  46], 'U': [104, 114], 'V': [115, 123], 'W': [ 65,  81], 'X': [135, 143], 
        'Y': [144, 150], 'Z': [124, 134]}
    letters_new = {}
    for key, item in leter_indices.items():
        letter_list = []
        for ii in range(item[0], item[1]+1):
            letter_list.append([df[3][ii], df[2][ii]])
            tmp_arr = np.array(letter_list)
            scale = 4. / (tmp_arr[:,0].max() - tmp_arr[:,0].min())
            tmp_arr[:,0] = (tmp_arr[:,0] - tmp_arr[:,0].min()) * scale
            tmp_arr[:,1] = (tmp_arr[:,1] - tmp_arr[:,1].min()) * scale
            letters_new[key] = tmp_arr
    return letters_new


def prepare_alphabet_coordinates(img_stone, letter_coordinates, side):
    stone_size = img_stone.size[0] # must be square!!
    alphabet = []
    coords_all = []

    dimx_max = 0
    for draw in alphabet_strings:
    # loop over stone coordinates in letter and add stones to image
        dimy = 5 * stone_size
        dimx = 1 + int(np.ceil(stone_size * (1 + letter_coordinates[draw][:,1].max()) - np.floor(letter_coordinates[draw][:,1].min())))
        dimx_max = dimx if dimx > dimx_max else dimx_max
    img_dimensions = (dimy, dimx_max + stone_size, 4)
    for draw in alphabet_strings:
    # loop over stone coordinates in letter and add stones to image
        dimy = 5 * stone_size
        dimx = 1 + int(np.ceil(stone_size * (1 + letter_coordinates[draw][:,1].max()) - np.floor(letter_coordinates[draw][:,1].min())))
        dimx_offset = dimx_max - dimx
        padding_columns = int(np.floor(dimx_offset/float(stone_size)))
        print("Offsetting " + draw + " by {:d} pixels, will insert {:d} solid columns".format(dimx_offset, padding_columns))
        coords = []
        for dot in range(letter_coordinates[draw].shape[0]):
            doty, dotx = letter_coordinates[draw][dot, :]
            yy = int(stone_size * doty)
            if side == 'left':
                xx = int(stone_size * (dotx - 0.5)) + stone_size + dimx_offset
            elif side == 'right':
                xx = int(stone_size * (dotx + 0.5))
            coords.append((yy, xx))
        for ii in range(padding_columns):
            for dot in range(letter_coordinates['I'].shape[0]):
                doty, dotx = letter_coordinates['I'][dot, :]
                yy = int(stone_size * doty)
                if side == 'left':
                    xx = int(stone_size * (ii + dotx))
                elif side == 'right':
                    xx = int(dimx_max + stone_size * (- ii + dotx) - 1)
                coords.append((yy, xx))
        coords_all.append(coords)
    return coords_all, img_dimensions


def prepare_alphabet(img_stone, letter_pixel_coordinates, img_dimensions, side):
    # print("making new img with dimensions ",*img_dimensions)
    # print(letter_pixel_coordinates)
    alphabet = []
    for coords in letter_pixel_coordinates:
        # print(coords)
        new_img = Image.fromarray(np.zeros(img_dimensions, dtype=np.uint8))
        for yy, xx in coords:
            new_img.paste(img_stone, (xx, yy), mask=img_stone)
        alphabet.append(new_img)
    return alphabet


def get_spaces(coords_all, side='left'):
    max_spaces = []
    for ii, coords in enumerate(coords_all):
        y = [y[0] for y in coords]
        x = [x[1] for x in coords]
        end = len(x)
        x_sorted = sorted(x)
        dx = [x1 - x0 for x0, x1 in zip(x_sorted[0:end-1], x_sorted[1:end])]
        if ii == 0 and side == 'right':
            rightmost_x = np.max(x)
            image_edge = rightmost_x + 25.
        if ii == 22:
            if side == 'left':
                max_spaces.append(0.02 * (np.min(dx) + 25))
            else:
                max_spaces.append(0.02 * (image_edge - np.max(x) - 25))
        else:
            max_spaces.append(0.02 * (np.max(dx) - 50))
    return max_spaces


def get_padding_column_number(coords_all, side='left'):
    padding_columns = []
    for ii, coords in enumerate(coords_all):
        if ii == 22:
            padding_columns.append(0) # W has none
        else:
            x = np.array([x[1] for x in coords])
            x_sorted = sorted(x)
            end = len(x)
            dx = [x1 - x0 for x0, x1 in zip(x_sorted[0:end-1], x_sorted[1:end])]
            if side == 'right':
                dx.reverse()
            padding_columns.append(int((1 + np.argmax(dx)) / 5))
            #print(bands.alphabet_strings[ii], np.max(dx), np.argmax(dx), (1 + np.argmax(dx)) / 5)
    return padding_columns


def weight_func(x, exponent=2.):
    """x is an array with values between 0 and 1"""
    x_left = np.sum(np.power(1 - x, exponent))
    x_right = np.sum(np.power(x, exponent))
    x_left, x_right = (x_left, x_right) / (x_left + x_right)
    return x_right - x_left


def letter_balance(coords_all, side='left'):
    padding_columns = get_padding_column_number(coords_all, side=side)
    letter_balance_values = []
    for coords, pdc in zip(coords_all, padding_columns):
        x = np.array([x[1] for x in coords])
        n_dots_letter = len(x) - 5 * pdc
        x = x[0:n_dots_letter]
        if x.min() == x.max(): # add exception for I, which has all stones in one column
            x_normalized = np.zeros(5) + 0.5
        else:
            x_normalized = (x - x.min()) / (x.max() - x.min())
        letter_balance_values.append(weight_func(x_normalized))
    return letter_balance_values

        
def adjust_spacing(coords_all, side='left', balance_adjustment_weight=0.5):
    ii = 0
    coords_all_new = []
    padding_columns = get_padding_column_number(coords_all, side=side)
    max_spaces = get_spaces(coords_all, side=side)
    letter_balance_values = letter_balance(coords_all, side=side)
    width_adjustment = []
    balance_adjustment = []
    for coords, pdc, offset, lbv in zip(coords_all, padding_columns, max_spaces, letter_balance_values):
        x = [x[1] for x in coords]
        y = [y[0] for y in coords]
        x_new = np.zeros(len(x))
        n_dots = len(x)
        n_dots_letter = n_dots - 5 * pdc
        move = (offset - 1.) * 0.3333333 * 50 if side == 'left' else - (offset - 1.) * 0.3333333 * 50
        move2 = balance_adjustment_weight * lbv * 50
        width_adjustment.append(move)
        balance_adjustment.append(move2)
        x_new[0:n_dots_letter] = np.array(x[0:n_dots_letter]) - move - move2
        x_new[n_dots_letter:n_dots] = x[n_dots_letter:n_dots]
        ii += 1
        coords_new = [(y_, x_) for y_, x_ in zip(y, list(x_new.astype(int)))]
        coords_all_new.append(coords_new)
    return coords_all_new, width_adjustment, balance_adjustment


def draw_alphabet(alphabet_images, name=""):
    fig, ax = plt.subplots(5, 6, figsize=(10,5), dpi=150)
    for ii, letter in enumerate(alphabet_images):
      # letter = np.where(letter > 255., 255., letter)
      ax.flatten()[ii].imshow(letter)
      ax.flatten()[ii].axis('off')

    for ii in range(26, 30):
      ax.flatten()[ii].axis('off')

    plt.savefig("alphabet"+name+".png")
    plt.close('all')


def make_my_band(img_stone, stone_size, dimx_max, letter_coordinates, total_columns = 72, letter_indices = [0, 13]):
    dimy = 5 * stone_size
    letter_width = 2 * dimx_max + 3 * stone_size
    total_border = int(total_columns - letter_width / stone_size)
    left_space = int(np.floor(total_border/2))
    right_space = int(np.ceil(total_border/2))
    def make_border(img_stone, N, dimy):
        coords = []
        border_img = Image.fromarray(np.zeros((dimy, int(N) * stone_size, 4), dtype=np.uint8))
        for ii in range(N):
            for dot in range(letter_coordinates['I'].shape[0]):
                doty, dotx = letter_coordinates['I'][dot, :]
                yy = int(stone_size * doty)
                xx = int(stone_size * (ii + dotx))
                coords.append((yy, xx))
                border_img.paste(img_stone, (xx, yy), mask=img_stone)
        return border_img, coords
    left_border, coords_border_left = make_border(img_stone, left_space, dimy)
    right_border, coords_border_right = make_border(img_stone, right_space, dimy)
    full_band, coords_full = make_border(img_stone, total_columns - 4, dimy) # - 4 fixes length, TODO: fix fix requirement??
    return left_border, right_border, full_band, coords_border_left, coords_border_right, coords_full


def reshape_img_to_height(img, height, debug_halo=False):
    ix, iy = img.size
    scale_factor = float(height) / float(iy)
    small_dims = (int(ix * scale_factor), int(iy * scale_factor))
    img_smaller = img.resize(small_dims, Image.BICUBIC)
    if debug_halo:
        img_smaller = draw_debug_halo(img_smaller)
    return(img_smaller)


def make_transparent_halo(img, place_xy, size=(1000, 1000)):
    new_img = Image.fromarray(np.zeros([size[1], size[0], 4], dtype=np.uint8))
    img.save("testA.png")
    new_img.paste(img, copy.copy(place_xy), mask=img.convert("RGBA"))
    return new_img


def adjust_band_corners(img_band):
    img_band_arr = np.array(img_band.convert("RGBA"))
    xend, yend = img_band.size
    corner_size = 5
    corner_alpha_tl = np.array([[  4,  99, 224, 253, 255],
                                [ 99, 245, 255, 255, 255],
                                [224, 255, 255, 255, 255],
                                [253, 255, 255, 255, 255],
                                [255, 255, 255, 255, 255]], dtype=np.uint8)
    img_band_arr[0:corner_size, 0:corner_size, 3] = corner_alpha_tl
    img_band_arr[0:corner_size, xend-corner_size:xend, 3] = np.flip(corner_alpha_tl, axis=1)
    img_band_arr[yend-corner_size:yend, 0:corner_size, 3] = np.flip(corner_alpha_tl, axis=0)
    img_band_arr[yend-corner_size:yend, xend-corner_size:xend, 3] = np.flip(corner_alpha_tl, axis=(0, 1))
    return Image.fromarray(img_band_arr)


def make_band_background(img_band_base, width, height, coordinates_xy, total_size):
    img_shaped_to_height = reshape_img_to_height(img_band_base, height)
    img_shaped_to_height = img_shaped_to_height.crop((0, 0, width, height)) #.convert("RGBA")
    img_shaped_to_height = adjust_band_corners(img_shaped_to_height)
    img_halo = make_transparent_halo(img_shaped_to_height, coordinates_xy, total_size)
    return img_halo


def place_glows(img, glows, offset_xy, old_img_dimensions, stone_coords, stone_size, glow_sizes, glow=True):
    new_stone_size = float(old_img_dimensions[1]) / 5.
    dot_scale_factor = float(new_stone_size) / float(stone_size)
    max_displacement = int(new_stone_size * 0.2)
    debug_square_r = make_debug_square(3, col=(255, 0, 0, 255))
    debug_square_b = make_debug_square(3, col=(0, 255, 0, 255))
    debug_square_g = make_debug_square(3, col=(0, 0, 255, 255))
    for yy, xx in stone_coords:
        doty = offset_xy[1] + old_img_dimensions[1] - (yy + stone_size) * dot_scale_factor
        # dotx = int((4*stone_size - dot_coords[1]) * dot_scale_factor + offset_xy[0])
        dotx = offset_xy[0] + xx * dot_scale_factor
        # img.paste(debug_square_r, (int(dotx), int(doty)))
        x = dotx + 0.5 * new_stone_size + np.random.randint(-max_displacement, max_displacement)
        y = doty + 0.5 * new_stone_size + np.random.randint(-max_displacement, max_displacement)
        # img.paste(debug_square_g, (int(x), int(y)))
        s = np.random.randint(0, 15)
        i = np.random.randint(0, 1000) if glow else 0
        if i > 993:
          siz = glow_sizes[2]
        elif i > 970:
          siz = glow_sizes[1]
        elif i > 900:
          siz = glow_sizes[0]
        else:
          continue
        x += -int(siz / 2)
        y += -int(siz / 2)
        img.paste(glows[siz][s], (int(x), int(y)), mask=glows[siz][s])
        # img.paste(debug_square_b, (int(x), int(y)))
    return img


def place_shadows(img, offset_xy, old_img_dimensions, stone_coords, stone_size, is_edge='False', is_raw=False):
    shadow_size = 2.
    new_stone_size = float(old_img_dimensions[1]) / 5.
    new_stone_size_int = int(float(old_img_dimensions[1]) / 5.)
    dot_scale_factor = float(new_stone_size) / float(stone_size)
    img_shadow_arr = np.zeros((3*new_stone_size_int, 3*new_stone_size_int, 4), dtype=np.uint8)
    img_shadows = sRGB_to_linear(Image.fromarray(np.zeros((img.size[1], img.size[0], 4), dtype=np.uint8)))
    img_shadow_debug = make_debug_cross(3*new_stone_size_int)
    xx, yy = np.meshgrid(np.linspace(-3, 3, num=3*new_stone_size_int), np.linspace(-3, 3, num=3*new_stone_size_int))
    img_shadow_arr[:,:,3] = (np.exp(-(xx**2+yy**2)/shadow_size) * 255).astype(np.uint8)
    img_shadow = sRGB_to_linear(Image.fromarray(img_shadow_arr))
    for yy, xx in stone_coords:
        doty = offset_xy[1] + old_img_dimensions[1] - (yy + stone_size) * dot_scale_factor
        dotx = offset_xy[0] + xx * dot_scale_factor
        x = dotx - new_stone_size
        y = doty - new_stone_size
        img_shadows.paste(img_shadow, (int(x), int(y)), mask=img_shadow)
        # img.paste(img_shadow_debug, (int(x), int(y)), mask=img_shadow_debug)
    img_shadows_arr = np.array(img_shadows)
    raw_padding = int(0.25*new_stone_size) if is_raw else 0
    img_shadows_arr[0:offset_xy[1] - raw_padding, :, :] = 0
    img_shadows_arr[offset_xy[1] + old_img_dimensions[1] + raw_padding:img_shadows.size[1], :, :] = 0
    if 'left' in is_edge:
        img_shadows_arr[:, 0:offset_xy[0] - int(0.25*new_stone_size), :] = 0
    else:
        img_shadows_arr[:, 0:offset_xy[0], :] = 0
    if 'right' in is_edge:
        img_shadows_arr[:, offset_xy[0] + old_img_dimensions[0] + int(0.25*new_stone_size):img_shadows_arr.shape[1]-1, :] = 0
    else:
        img_shadows_arr[:, offset_xy[0] + old_img_dimensions[0]:img_shadows_arr.shape[1]-1, :] = 0
    img_shadows = Image.fromarray(img_shadows_arr)
    img_shadows.save("test_img_shadows.png")
    img.save("test_img.png")
    img_shadows.paste(img, (0, 0), mask=img)
    img_shadows.save("test_img_shadows2.png")
    return img_shadows


def get_max_width(imgs):
    nx_max = 0
    for img in imgs:
        nx_max = img.size[0] if img.size[0] > nx_max else nx_max
    return nx_max


def save_image(img, img_name):
    print("Saving ", img_name)
    linear_to_sRGB(img).save(img_name)


debug_halo = False
def main():
    img_shine, _ = get_img("funkeln.png")
    #prepare shine dict
    glows, glow_sizes = split_shine(img_shine)
    for model in models:
        model_subdir = "img_out/{:s}/".format(model.name)
        if not os.path.isdir(model_subdir):
            os.system("mkdir -p "+model_subdir)
        for col in wallet_colors:
            model_with_color = model.name + "_{:s}".format(col)
            wallet_img_url = "models/{:s}.jpg".format(model_with_color)
            img_wallet, img_wallet_path = get_img(wallet_img_url)
            os.system("cp {:s} {:s}.".format(img_wallet_path, model_subdir))
        need_bg = True
        for stone_type in stone_type_list:
            output_subdir = "{:s}/{:s}/".format(model_subdir, stone_type)
            if not os.path.isdir(output_subdir):
                os.system("mkdir -p "+output_subdir)
            stone_img_url = "stones/{:s}.png".format(
                stone_type)
            img_stone, _ = get_img(stone_img_url) #.transpose(Image.FLIP_TOP_BOTTOM)
        
            stone_size = img_stone.size[0] # must be square

            letter_coordinates = get_alphabet_coordinates()
            letter_pixel_coordinates_left, img_dimensions_left = prepare_alphabet_coordinates(img_stone, letter_coordinates, 'left')
            letter_pixel_coordinates_right, img_dimensions_right = prepare_alphabet_coordinates(img_stone, letter_coordinates, 'right')
            if adjust_letter_spacing:
                letter_pixel_coordinates_left, _, _ = adjust_spacing(letter_pixel_coordinates_left, side='left', balance_adjustment_weight=0.4)
                letter_pixel_coordinates_right, _, _ = adjust_spacing(letter_pixel_coordinates_right, side='right', balance_adjustment_weight=0.4)
            alphabet_left = prepare_alphabet(img_stone, letter_pixel_coordinates_left, img_dimensions_left, 'left')
            alphabet_right = prepare_alphabet(img_stone, letter_pixel_coordinates_right, img_dimensions_right, 'right')

            dimx_max = np.max([get_max_width(alphabet_left), get_max_width(alphabet_right)])

            band_left, band_right, band_full, coords_border_left, coords_border_right, coords_full = make_my_band(img_stone, stone_size, dimx_max, 
                letter_coordinates, total_columns=model.band_columns)

            # do an initial resize to get dimensions:
            img1 = reshape_img_to_height(band_left.transpose(Image.FLIP_TOP_BOTTOM), model.band_height)
            img2 = reshape_img_to_height(alphabet_left[0].transpose(Image.FLIP_TOP_BOTTOM), model.band_height)
            img3 = reshape_img_to_height(alphabet_right[0].transpose(Image.FLIP_TOP_BOTTOM), model.band_height)
            img4 = reshape_img_to_height(band_right.transpose(Image.FLIP_TOP_BOTTOM), model.band_height)
            # get exact locations from img sizes:
            left_start = model.band_coordinates
            letter1_start = [left_start[0] + img1.size[0], left_start[1]]
            letter2_start = [letter1_start[0] + img2.size[0], left_start[1]]
            right_start = [letter2_start[0] + img3.size[0], left_start[1]]
            if need_bg:
                total_band_width =  img1.size[0] + img2.size[0] + img3.size[0] + img4.size[0] + int(0.5 * 0.2 * model.band_height)
                total_band_height = int((1 + 0.5 * 0.2) * model.band_height) if model.is_raw else model.band_height
                bg_band_coordinates = [
                    model.band_coordinates[0] - int(0.25 * 0.2 * model.band_height), 
                    model.band_coordinates[1] - int(0.25 * 0.2 * model.band_height)] if model.is_raw else [
                    model.band_coordinates[0] - int(0.25 * 0.2 * model.band_height), 
                    model.band_coordinates[1]]
                print("Making band with dimensions ", total_band_width, total_band_height)
                print("Placing it at ", bg_band_coordinates, " original band coordinates are ", model.band_coordinates)
                for bg_col in band_bg_colors:
                    url_band_bg = "models/Alcantara_{:s}.png"
                    img_band_bg, _ = get_img(url_band_bg.format(bg_col))
                    band_bg_current = make_band_background(img_band_bg, total_band_width, total_band_height, bg_band_coordinates, img_wallet.size)
                    save_image(band_bg_current, model_subdir + "Alcantara_{:s}.png".format(bg_col))
                need_bg = False

            full_scaled = reshape_img_to_height(band_full.transpose(Image.FLIP_TOP_BOTTOM), model.band_height, debug_halo=debug_halo)
            full_scaled_halo = make_transparent_halo(full_scaled, left_start, size=img_wallet.size)
            full_scaled_shadows = place_shadows(
                  full_scaled_halo, left_start, full_scaled.size, 
                  coords_full, stone_size, is_edge='left right', is_raw=model.is_raw)
            full_scaled_glow = place_glows(
                  full_scaled_shadows, glows, left_start, full_scaled.size, 
                  coords_full, stone_size, glow_sizes, glow=model.add_glow)
            left_scaled = reshape_img_to_height(band_left.transpose(Image.FLIP_TOP_BOTTOM), model.band_height, debug_halo=debug_halo)
            left_scaled_halo = make_transparent_halo(left_scaled, left_start, size=img_wallet.size)
            left_scaled_shadows = place_shadows(
                  left_scaled_halo, left_start, left_scaled.size, 
                  coords_border_left, stone_size, is_edge='left', is_raw=model.is_raw)
            left_scaled_glow = place_glows(
                  left_scaled_shadows, glows, left_start, left_scaled.size, 
                  coords_border_left, stone_size, glow_sizes, glow=model.add_glow)
            right_scaled = reshape_img_to_height(band_right.transpose(Image.FLIP_TOP_BOTTOM), model.band_height, debug_halo=debug_halo)
            right_scaled_halo = make_transparent_halo(right_scaled, right_start, size=img_wallet.size)
            right_scaled_shadows = place_shadows(
                  right_scaled_halo, right_start, right_scaled.size, 
                  coords_border_right, stone_size, is_edge='right', is_raw=model.is_raw)
            right_scaled_glow = place_glows(
                right_scaled_shadows, glows, right_start, right_scaled.size, 
                coords_border_right, stone_size, glow_sizes, glow=model.add_glow)
            save_image(full_scaled_glow, output_subdir + "full.png")
            save_image(left_scaled_glow, output_subdir + "left.png")
            save_image(right_scaled_glow, output_subdir + "right.png")
# def place_shadows(img, offset_xy, old_img_dimensions, stone_coords, stone_size, is_edge='False'):
            for ii in range(len(alphabet_left)):
                letter1_scaled = reshape_img_to_height(alphabet_left[ii].transpose(Image.FLIP_TOP_BOTTOM), model.band_height, debug_halo=debug_halo)
                letter1_scaled_halo = make_transparent_halo(letter1_scaled, letter1_start, size=img_wallet.size)
                img_shadows_halo = place_shadows(letter1_scaled_halo, letter1_start, letter1_scaled.size, 
                    letter_pixel_coordinates_left[ii], stone_size, is_raw=model.is_raw)
                img_glow = place_glows(
                    img_shadows_halo, glows, letter1_start, letter1_scaled.size, 
                    letter_pixel_coordinates_left[ii], stone_size, glow_sizes, glow=model.add_glow)
                save_image(img_glow, output_subdir + "l_{:02d}.png".format(ii))
                # cv2.imwrite("img_out/l_{:02d}.png".format(ii), letter1_scaled_halo_glow)
                letter2_scaled = reshape_img_to_height(alphabet_right[ii].transpose(Image.FLIP_TOP_BOTTOM), model.band_height, debug_halo=debug_halo)
                letter2_scaled_halo = make_transparent_halo(letter2_scaled, letter2_start, size=img_wallet.size)
                img_shadows_halo = place_shadows(letter2_scaled_halo, letter2_start, letter2_scaled.size, 
                    letter_pixel_coordinates_right[ii], stone_size, is_raw=model.is_raw)
                img_glow = place_glows(
                    img_shadows_halo, glows, letter2_start, letter2_scaled.size,
                    letter_pixel_coordinates_right[ii], stone_size, glow_sizes, glow=model.add_glow)
                save_image(img_glow, output_subdir + "r_{:02d}.png".format(ii))
    os.system('cp -R img_out ../../../sf_share/')

if __name__ == "__main__":
    main()

