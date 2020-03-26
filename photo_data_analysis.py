from barcode_tracker_photos import *
from os.path import isfile, isdir, join, splitext, ismount
import re
from datetime import datetime
from os import rename
import TagList

target_pop="P10"

with open("processed_lists/processed_social_photos_{}.txt".format(target_pop), "a+") as processed_file:
    already_processed = [line.strip() for line in processed_file]
    print(len(already_processed))

def create_csv(data_filepath):
    with open(data_filepath, "a+") as savefile: # open data file in append mode
        # write column names to file
        header = "population,time,frame_num,id,id_prob,x,y,orientation\n"
        savefile.write(header)

if __name__=="__main__":

    tags = TagList.TagList()
    tags.load("master_list_outdoor.pkl")


    server_path = "/mnt/Videos_GRETI/field_season_winter_2020"
    data_dir_csv = "/home/michael/pinpoint_tits/social_data"

    #these parent directories are filled with child directories, uploaded twice a day from raspis
    parent_directories = [join(server_path, d) for d in os.listdir(server_path)
                            if (isdir(join(server_path, d)) and
                            "Social" in d and
                            re.search("P\d?\d", d).group(0)==target_pop) and
                            d not in already_processed]
    parent_directories = sorted(parent_directories, key=str.lower, reverse=False)

    for directory in parent_directories:
        rawtime = re.search("\d\d\d\d-\d\d-\d\d_\d\d", directory).group(0)
        population = re.search("P\d?\d", directory).group(0)
        data_filepath = join(data_dir_csv,"{}_{}_pinpoint.csv".format(population,rawtime))
        if not isfile(data_filepath):
            create_csv(data_filepath)
        #these child directories are filled with photos from 5 min intervals
        child_directories = [join(server_path, directory, child) for child in os.listdir(directory)
                                if (isdir(join(server_path, directory, child)) and
                                child not in already_processed)]
        child_directories = sorted(child_directories, key=str.lower, reverse=False)

    try:
        for five_min_folder in child_directories:
            if five_min_folder not in already_processed:
                print(five_min_folder)
                try:
                    decode(five_min_folder, data_filepath, tags,population)
                except Exception as e:
                    print("error processing {}. Error: {}".format(five_min_folder, e))
                else:
                    with open(data_filename, "a+") as processed_file:
                        processed_file.write("{}\n".format(five_min_folder))
    except:
        print("error processing {}".format(directory))
        pass

    else:
        print("finished processing {}".format(directory))
        with open(data_filename, "a+") as processed_file:
            processed_file.write("{}\n".format(directory))
