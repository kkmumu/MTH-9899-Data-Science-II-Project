### main.py

import argparse
import sys

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-i", "--input", help = "input directory")
    parser.add_argument("-o", "--output", help = "output directory")
    parser.add_argument("-p", "--provide", help = "directory containing the learned models (only for mode 2)")
    parser.add_argument("-s", "--start", help = "start date in YYYYMMDD format", type = int)
    parser.add_argument("-e", "--end", help = "end date in YYYYMMDD format", type = int)
    parser.add_argument("-m","--mode", help = "mode to choose 1 or 2", type = int)
    
    args = parser.parse_args()

    return args

def create_features(start, end, input_dir, output_dir):
    """
    Create features between startdate and enddate and saved to the output directory specified.

    Parameters
    ----------
    start : int
        start date to create features (YYYYMMDD format)
    end : int
        end date to create features (YYYYMMDD format)
    input_dir : str
        directory to read the raw input data
    output_dir : str
        directory to save the output features files

    Returns
    -------
    None.

    """
    
def LinearRegression(start, end, input_dir, output_dir):
    """
    Make predictions between startdate and enddate by applying the learned model provided.

    Parameters
    ----------
    start : int
        start date to predict (YYYYMMDD format)
    end : int
        end date to predict (YYYYMMDD format)
    input_dir : str
        directory of the features previously created
    output_dir : str
        directory to save the output predictions

    Returns
    -------
    None.

    """



def main():
    print("this is the main function")
    
    args = parse_args()
    
    
    if args.mode == 1:
        print("Mode 1 entered.")
        create_features(args.start, args.end, args.input, args.output)
        
    elif args.mode == 2:
        print("Mode 2 entered.")
        if args.provide == "LinearRegression":
            LinearRegression(args.start, args.end, args.input, args.output)
        
    else:
        print("Invalid input!")
    
    

if __name__ == "__main__"
    main()

