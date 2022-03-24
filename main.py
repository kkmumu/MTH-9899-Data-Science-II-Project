### main.py
import numpy as np

from prep import *
from features import *
from cross_validation import *
from regmodel import *
import light_gbm
import xg_boost
import extra_trees


def predict(args, input_dir, output_dir, start_date, end_date, model = "light_gbm"):
    df = pd.read_csv(input_dir + "features"+str(start_date)+"_"+str(end_date)+".csv", sep = ",")
    
    id_date = df.iloc[:,0:2]
    test = df[df[df["Date"] >= (start_date)]["Date"] <= (end_date)].drop(columns = ["Date", "Id"], axis = 1)
    #test = test.iloc[:, 1:]
    X_test = test.iloc[:, :-1]

    y_test = test['y']

    filepath = args.provide + model + ".sav"
    # print(filepath)
    
    # lightgbm is saved as lightgbm.sav in folder model
    # load the model from disk
    try:
        loaded_model = pickle.load(open(filepath, 'rb'))
        y_pred = loaded_model.predict(X_test)
    except Exception as e:
        print(str(e))
        print("There is no such model!")
        return


    id_date["Pred"] = y_pred
    id_date["Time"] = "17:30:00.000"
    id_date = id_date[["Date", "Time", "Id", "Pred"]]
    #id_date.to_csv("drift_plot.csv")
    for date in set(id_date.Date):
        id_date[id_date["Date"] == date].to_csv(output_dir + str(date)+".csv", index = False)



def main():

    print("Welcome!")
    
    args = parse_args()
    # parse params
    start_date = args.start
    end_date = args.end
    input_dir = args.input
    output_dir = args.output
    
    
    #######################################################  Mode 1 ##########################################################
    if args.mode == 1:
        """
        Mode 1 creates features between startdate and enddate and save to the output directory specified, a CSV file per date with features for each ID.
    
        Parameters
        ----------
        start : int64
            start date to predict (YYYYMMDD format)
        end : int64
            end date to predict (YYYYMMDD format)
        input_dir : str
            directory of the features previously created
        output_dir : str
            directory to save feature files
        """
        print("Mode 1 entered.")
        
        create_features(input_dir, output_dir, start_date, end_date)
        
        
        
        
    #######################################################  Mode 2 ##########################################################
    elif args.mode == 2:
        """
    Mode 2 makes predictions between startdate and enddate by applying the learned model provided to features produced from a run in Mode 1
    
    Parameters
    ----------
    start : int64
        start date to predict (YYYYMMDD format)
    end : int64
        end date to predict (YYYYMMDD format)
    input_dir : str
        directory of the features previously created and there is a subdirectory included called "model" storing model pickles
    output_dir : str
        directory to save the output predictions
        
        """
        print("Mode 2 entered.")
        
        predict(args, input_dir, output_dir, start_date, end_date)

            
        
    #######################################################  Others ##########################################################
    else:
    
        print("*******************Invalid input!******************")
    
    
    
    print("Main function executed.")
    
    

            
    
    

if __name__ == "__main__":
    main()



# save the model to disk
# filename = 'finalized_model.sav'
# pickle.dump(model, open(filename, 'wb'))
