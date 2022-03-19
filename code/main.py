### main.py

from prep import *


def create_features(args):
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
    start_date = args.start
    end_date = args.end
    input_dir = args.input
    output_dir = args.output
    
    
    
def LinearRegression(args):
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
    start_date = args.start
    end_date = args.end
    input_dir = args.input
    output_dir = args.output



def main():
    print("this is the main function")
    args = parse_args()
    
    
    if args.mode == 1:
    """"
    Mode 1 creates features between startdate and enddate and save to the output directory specified, a CSV file per date with features for each ID.
    """"
        print("Mode 1 entered.")
        create_features(args)
        
    elif args.mode == 2:
    """
    Mode 2 makes predictions between startdate and enddate by applying the learned model provided to features produced from a run in Mode 1
    """
        print("Mode 2 entered.")
        if args.provide == "LinearRegression":
            LinearRegression(args)
        
    else:
        print("Invalid input!")
    
    

if __name__ == "__main__"
    main()

