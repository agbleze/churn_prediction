
import argparse

#%%
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-- file_name", type=str, help="name of data file")
    parser.add_argument("-- data_store", type=str, help="location of the data file. Could be a folder"                    
                        )
    parser.add_argument("-- save_dir", type=str, help="directory to save model output")
    
    args = parser.parse_args()
    return args