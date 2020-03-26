from barcode_tracker_live import *
from os.path import isfile, isdir, join, splitext, ismount
import re
from datetime import datetime
from os import rename, listdir
import TagList

target_pop="P10"
data_filename = "processed_lists/processed_puzzle_vids_{}.txt".format(target_pop)
with open(data_filename, "r") as processed_vids_file:
    already_processed = [line.strip() for line in processed_vids_file]
    print(len(already_processed))

def create_csv(data_filepath):
    with open(data_filepath, "a+") as savefile: # open data file in append mode
        # write column names to file
        header = "population,time,frame_num,id,id_prob,x,y,orientation\n"
        savefile.write(header)

if __name__=="__main__":
    # ## Import list of barcodes
    tags = TagList.TagList()
    tags.load("master_list_outdoor.pkl")

    #server_path = "/mnt/Videos_GRETI/field_season_winter_2020"
    server_path = "/mnt/Videos_GRETI/field_season_winter_2020"
    data_dir_csv = "/home/michael/pinpoint_tits/puzzle_data"

    #these parent directories are filled with child directories, uploaded twice a day from raspis
    parent_directories = [join(server_path, d) for d in listdir(server_path)
                            if (isdir(join(server_path, d)) and
                            ("Puzzle" in d) and
                            (re.search("P\d?\d", d).group(0)==target_pop) and
                            d not in already_processed)]
    parent_directories = sorted(parent_directories, key=str.lower, reverse=False)
    print("finished gathering parent directories.")

    for directory in parent_directories:
        population = re.search("P\d?\d", directory).group(0)
        rawtime = re.search("\d\d\d\d-\d\d-\d\d_\d\d", directory).group(0)
        data_filepath = join(data_dir_csv,"{}_{}_pinpoint.csv".format(population,rawtime))
        if not isfile(data_filepath):
            create_csv(data_filepath)
        child_files = [join(server_path, directory, child) for child in listdir(directory)
                                if (isfile(join(server_path, directory, child)) and
                                child not in already_processed and
                                "debug" not in child)]
        child_files = sorted(child_files, key=str.lower, reverse=False)

        try:
            for video in child_files:
                print(video)
                try:
                    decode(video,data_filepath,tags,target_pop)
                except Exception as e:
                    print("error processing {}. Error: {}".format(video, e))
                else:
                    with open(data_filename, "a+") as processed_vids_file:
                        processed_vids_file.write("{}\n".format(video))

        except:
            print("error processing {}".format(directory))
            pass
        else:
            print("finished processing {}".format(directory))
            with open(data_filename, "a+") as processed_file:
                processed_file.write("{}\n".format(directory))
