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
import pandas as pd


alphabet_strings = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']




class Model:
    def __init__(self, name, band_coordinates, band_height, band_columns):
        self.name = name
        self.band_coordinates = band_coordinates
        self.band_height = band_height
        self.band_columns = band_columns

models = [
    #Model("Damenboerse_ohneStrass_fixed", [ 49, 538],  66, 72),
    Model("111520_066_99ini",             [211, 579], 122, 40),
    Model("112150_002_99ini",             [ 92, 696],  74, 79),
    Model("116490_007_99ini",             [169, 440], 106, 40),
    Model("116585_045_99ini",             [138, 442],  75, 77),
    Model("118102_007_99ini",             [308, 692],  47, 71),
    Model("118103_045_99ini",             [434, 581],  30, 84),
    Model("118104_066_99ini",             [298, 856],  42, 84)
    ]

# stone_type_list = ["1088_DenimBlue266_enh"]
# stone_type_list = ["1088_001Crystal_R", "1088_DenimBlue266"]
stone_type_list = ["1088_BlackDiamond215", "1088_001Crystal_R", "1088_DenimBlue266", "1088_DenimBlue266_enh", "1088_Greige284", "1088_Rose209"]


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
        return (data * 255.).astype(np.unit8)


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
        return (data * 255.).astype(np.unit8)


def draw_diagnostic_halo(img):
    nx, ny = img.size
    for yy in [0, ny-1]:
        for xx in range(nx):
            img.putpixel((xx, yy), (0, 255, 0, 255))
    for xx in [0, nx-1]:
        for yy in range(ny):
            img.putpixel((xx, yy), (0, 255, 0, 255))
    return img

def download_img(urlstring):
    print("Downloading ", urlstring)
    urllib.request.urlretrieve(urlstring, urlstring.split("/")[-1])
    img = Image.open(urlstring.split("/")[-1])
    return sRGB_to_linear(img)

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
    req = urllib.request.urlopen("https://raw.githubusercontent.com/baeldin/strass_preview_generator/main/Alphabet.asc")
    df = pd.read_csv("https://raw.githubusercontent.com/baeldin/strass_preview_generator/main/Alphabet.asc", delimiter=";", header=None, skiprows=3)
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


def prepare_alphabet(img_stone, letter_coordinates, side):
    stone_size = img_stone.size[0] # must be square!!
    alphabet = []
    coords_all = []

    dimx_max = 0
    for draw in alphabet_strings:
    # loop over stone coordinates in letter and add stones to image
        dimy = 5 * stone_size
        dimx = 1 + int(np.ceil(stone_size * (1 + letter_coordinates[draw][:,1].max()) - np.floor(letter_coordinates[draw][:,1].min())))
        dimx_max = dimx if dimx > dimx_max else dimx_max

    for draw in alphabet_strings:
    # loop over stone coordinates in letter and add stones to image
        dimy = 5 * stone_size
        dimx = 1 + int(np.ceil(stone_size * (1 + letter_coordinates[draw][:,1].max()) - np.floor(letter_coordinates[draw][:,1].min())))
        dimx_offset = dimx_max - dimx
        padding_columns = int(np.floor(dimx_offset/float(stone_size)))
        print("Offsettting " + draw + " by {:d} pixels, will insert {:d} solid columns".format(dimx_offset, padding_columns))
        new_img = Image.fromarray(np.zeros((dimy, dimx_max + stone_size, 4), dtype=np.uint8))
        coords = []
        for dot in range(letter_coordinates[draw].shape[0]):
            doty, dotx = letter_coordinates[draw][dot, :]
            if side == 'left':
                yy = int(stone_size * doty)
                xx = int(stone_size * (dotx - 0.5)) + stone_size + dimx_offset
            elif side == 'right':
                yy = int(stone_size * doty)
                xx = int(stone_size * (dotx + 0.5))
            coords.append((yy, xx))
            new_img.paste(img_stone, (xx, yy), mask=img_stone)
        for ii in range(padding_columns):
            for dot in range(letter_coordinates['I'].shape[0]):
                doty, dotx = letter_coordinates['I'][dot, :]
                if side == 'left':
                    yy = int(stone_size * doty)
                    xx = int(stone_size * (ii + dotx))
                elif side == 'right':
                    yy = int(stone_size * doty)
                    xx = int(dimx_max + stone_size * (- ii + dotx) - 1)
                coords.append((yy, xx))
                new_img.paste(img_stone, (xx, yy), mask=img_stone)
        alphabet.append(new_img)
        linear_to_sRGB(new_img).save("test.png")
        coords_all.append(coords)
    return alphabet, coords_all


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
    total_width = total_border + letter_width
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
    return left_border, right_border, coords_border_left, coords_border_right




def reshape_img_to_height(img, height, diagnostic_halo=False):
    ix, iy = img.size
    scale_factor = float(height) / float(iy)
    small_dims = (int(ix * scale_factor), int(iy * scale_factor))
    img_smaller = img.resize(small_dims, Image.BICUBIC)
    if diagnostic_halo:
        img_smaller = draw_diagnostic_halo(img_smaller)
    return(img_smaller)





def make_transparent_halo(img, place_xy, size=(1000, 1000)):
    new_img = Image.fromarray(np.zeros([*size, 4], dtype=np.uint8))
    ny, nx = img.size
    new_img.paste(img, copy.copy(place_xy), mask=img.convert("RGBA"))
    return new_img

def make_band_background(img_band_base, width, height, coordinates_xy, total_size):
    img_height = reshape_img_to_height(img_band_base, height)
    img_height = img_height.crop((0, 0, width, height)) #.convert("RGBA")
    img_halo = make_transparent_halo(img_height, coordinates_xy, total_size)
    return img_halo

def place_glows(img, glows, offset_xy, old_img_dimensions, stone_coords, stone_size, glow_sizes):
    new_stone_size = old_img_dimensions[1] / 5.
    dot_scale_factor = 1. / stone_size * new_stone_size
    max_displacement = int(new_stone_size * 0.35)
    for dot_coords in stone_coords:
        doty = int(dot_coords[0] * dot_scale_factor+ offset_xy[1])
        # dotx = int((4*stone_size - dot_coords[1]) * dot_scale_factor + offset_xy[0])
        dotx = int(dot_coords[1] * dot_scale_factor + offset_xy[0])
        x = dotx + np.random.randint(-max_displacement, max_displacement)
        y = doty + np.random.randint(-max_displacement, max_displacement)
        s = np.random.randint(0, 15)
        i = np.random.randint(0, 1000)
        if i > 993:
          siz = glow_sizes[2]
        elif i > 970:
          siz = glow_sizes[1]
        elif i > 900:
          siz = glow_sizes[0]
        else:
          continue
        x += 7 - int(siz / 2)
        y += 7 - int(siz / 2)
        img.paste(glows[siz][s], (x, y), mask=glows[siz][s])
    return img


def get_max_width(imgs):
    nx_max = 0
    for img in imgs:
        nx_max = img.size[0] if img.size[0] > nx_max else nx_max
    return nx_max


def save_image(img, img_name):
    print("Saving ", img_name)
    linear_to_sRGB(img).save(img_name)


diagnostic_halo = False
def main():
    img_shine = download_img("https://raw.githubusercontent.com/baeldin/strass_preview_generator/main/funkeln.png")
    img_band_bg = download_img("https://raw.githubusercontent.com/baeldin/strass_preview_generator/main/models/alcantara_002.png")
    #prepare shine dict
    glows, glow_sizes = split_shine(img_shine)
    for model in models:
        wallet_img_url = "https://raw.githubusercontent.com/baeldin/strass_preview_generator/main/models/{:s}.png".format(model.name)
        img_wallet = download_img(wallet_img_url)
        model_subdir = "img_out/{:s}/".format(model.name)
        if not os.path.isdir(model_subdir):
            os.system("mkdir -p "+model_subdir)
        os.system("mv {:s}.png img_out/{:s}/.".format(model.name, model.name))
        need_bg = True
        for stone_type in stone_type_list:
            output_subdir = "img_out/{:s}/{:s}/".format(model.name, stone_type)
            if not os.path.isdir(output_subdir):
                os.system("mkdir -p "+output_subdir)
            stone_img_url = "https://raw.githubusercontent.com/baeldin/strass_preview_generator/main/stones/{:s}.png".format(
                stone_type)
            img_stone = download_img(stone_img_url) #.transpose(Image.FLIP_TOP_BOTTOM)
        
            stone_size = img_stone.size[0] # must be square

            letter_coordinates = get_alphabet_coordinates()
            alphabet_right, coords_right = prepare_alphabet(img_stone, letter_coordinates, 'right')
            alphabet_left, coords_left = prepare_alphabet(img_stone, letter_coordinates, 'left')
            dimx_max = np.max([get_max_width(alphabet_left), get_max_width(alphabet_right)])

            band_left, band_right, coords_border_left, coords_border_right = make_my_band(img_stone, stone_size, dimx_max, 
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
                total_band_width = img1.size[0] + img2.size[0] + img3.size[0] + img4.size[0] + int(0.5 * 0.2 * model.band_height)
                bg_band_coordinates = [model.band_coordinates[0] - int(0.25 * 0.2 * model.band_height), model.band_coordinates[1]]
                band_bg_current = make_band_background(img_band_bg, total_band_width, model.band_height, bg_band_coordinates, img_wallet.size)
                save_image(band_bg_current, model_subdir + "band_bg_002.png")
                need_bg = False

            left_scaled = reshape_img_to_height(band_left.transpose(Image.FLIP_TOP_BOTTOM), model.band_height, diagnostic_halo=diagnostic_halo)
            left_scaled_halo = make_transparent_halo(left_scaled, left_start, size=img_wallet.size)
            left_scaled_glow = place_glows(
                  left_scaled_halo, glows, left_start, left_scaled.size, 
                  coords_border_left, stone_size, glow_sizes)
            right_scaled = reshape_img_to_height(band_right.transpose(Image.FLIP_TOP_BOTTOM), model.band_height, diagnostic_halo=diagnostic_halo)
            right_scaled_halo = make_transparent_halo(right_scaled, right_start, size=img_wallet.size)
            right_scaled_glow = place_glows(
                right_scaled_halo, glows, right_start, right_scaled.size, 
                coords_border_right, stone_size, glow_sizes)
            save_image(left_scaled_glow, output_subdir + "left.png")
            save_image(right_scaled_glow, output_subdir + "right.png")

            for ii in range(len(alphabet_left)):
                letter1_scaled = reshape_img_to_height(alphabet_left[ii].transpose(Image.FLIP_TOP_BOTTOM), model.band_height, diagnostic_halo=diagnostic_halo)
                letter1_scaled_halo = make_transparent_halo(letter1_scaled, letter1_start, size=img_wallet.size)
                img_glow = place_glows(
                    letter1_scaled_halo, glows, letter1_start, letter1_scaled.size, 
                    coords_left[ii], stone_size, glow_sizes)
                save_image(img_glow, output_subdir + "l_{:02d}.png".format(ii))
                # cv2.imwrite("img_out/l_{:02d}.png".format(ii), letter1_scaled_halo_glow)
                letter2_scaled = reshape_img_to_height(alphabet_right[ii].transpose(Image.FLIP_TOP_BOTTOM), model.band_height, diagnostic_halo=diagnostic_halo)
                letter2_scaled_halo = make_transparent_halo(letter2_scaled, letter2_start, size=img_wallet.size)
                img_glow = place_glows(
                    letter2_scaled_halo, glows, letter2_start, letter2_scaled.size,
                    coords_right[ii], stone_size, glow_sizes)
                save_image(img_glow, output_subdir + "r_{:02d}.png".format(ii))
    os.system('cp -R img_out ../../../sf_share/.')

if __name__ == "__main__":
    main()

