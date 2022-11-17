import numpy as np
import os
import csv
import pandas as pd

name = []
brand = []
hightop = []
color = []


def foldername_split(PATH):
    folder_list = os.listdir(PATH)

    for foldername in folder_list:
        folder = foldername.split('_')

        brand.append(folder[0])
        hightop.append(folder[1])
        color.append(folder[2])
        name.append(folder[3])

    return brand, hightop, color, name


def generate_csv(brand, hightop, color, name):
    csv_made = pd.DataFrame(name, columns=['name'])
    csv_made['brand'] = brand
    csv_made['color'] = color
    csv_made['hightop'] = hightop

    to_csv = csv_made.to_csv("meta_info.csv", index=False)

    return to_csv


if __name__ == '__main__':
    PATH = input("Dataset folder directory: ")

    brand, hightop, color, name = foldername_split(PATH)

    generate_csv(brand, hightop, color, name)
