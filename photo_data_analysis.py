from barcode_tracker_photos import *
from os.path import isfile, isdir, join, splitext, ismount
import re
from datetime import datetime
from os import rename

server_path = "/mnt/Videos_GRETI/field_season_winter_2020"
data_dir_csv = "/home/michael/TITS/pinpoint_data_2020"


#these parent directories are filled with child directories, uploaded twice a day from raspis
parent_directories = [join(server_path, d) for d in os.listdir(server_path) if (isdir(join(server_path, d)) and ("Puzzle" not in d))]
parent_directories = sorted(parent_directories, key=str.lower, reverse=False)
#print(parent_directories)

for directory in parent_directories:
    if "processed" not in directory:
        rawtime = directory[-13:]
        print(rawtime)
        if "P10" in directory or "P11" in directory or "P12" in directory:
            aviary = directory[-27:-14]
        else:
            aviary = directory[-26:-14]
        print(aviary)
        #time = datetime.strptime(rawtime, "%Y-%m-%d_%H")
        data_filepath = create_csv(data_dir_csv, aviary, rawtime)
        
        #these child directories are filled with photos from 5 min intervals
        child_directories = [join(server_path, directory, child) for child in os.listdir(directory) if (isdir(join(server_path, directory, child)) and "processed" not in child)]
        child_directories = sorted(child_directories, key=str.lower, reverse=False)
        for five_min_folder in child_directories:
            if "processed" not in five_min_folder:
                print(five_min_folder)
                try:        
                    decode(five_min_folder,data_filepath)
                except Exception as e:
                    print("error processing {}. Error: {}".format(five_min_folder, e))
                else:
                    rename(five_min_folder,"{}{}".format(five_min_folder,"_processed"))

    rename(directory,"{}{}".format(directory,"_processed"))
