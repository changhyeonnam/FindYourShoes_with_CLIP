import os
import pandas as pd

def foldername_split(PATH):
    folder_list = os.listdir(PATH)
    brand,hightop,sole,color,name = [],[],[],[],[]
    for foldername in folder_list:
        folder = foldername.split('_')
        if len(folder)<5:
            print(folder)
            continue
        brand.append(folder[0].strip())
        hightop.append(folder[1].strip())
        color.append(folder[2].strip())
        sole.append(folder[3].strip())
        name.append(folder[4].strip())

    return brand, hightop, color, sole, name


def generate_csv(brand, hightop, color, sole, name,meta_filename):
    csv_made = pd.DataFrame(name, columns=['name'])
    csv_made['brand'] = brand
    csv_made['color'] = color
    csv_made['hightop'] = hightop
    csv_made['sole'] = sole
    print(f'meta information dataframe information : {csv_made.info}')
    csv_made.to_csv(meta_filename, index=False)


if __name__ == '__main__':
    PATH = 'final_dataset'
    meta_filename = 'meta_info_final.csv'
    brand, hightop, color, name, sole = foldername_split(PATH)
    generate_csv(brand, hightop, color, name, sole,meta_filename)
