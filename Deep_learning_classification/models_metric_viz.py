import os
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


def LoadTensorboards():
	df_tensorboards = dict()
	path = "metrics"
	path_to_= os.path.join(os.getcwd(),path)
	for csv in os.listdir(path_to_):
	    if csv.endswith("csv"):
	        df_tensorboards[str(csv)]=(pd.read_csv(os.path.join(path_to_,csv)))
	return df_tensorboards

def showmcuvemodel(title,df):
    key_split = list(title.split("_"))
    fig, axes = plt.subplots( figsize=(15.0, 15.0) , nrows=2, ncols=1, sharey=True)
    title_ = f"model = {key_split[0]}, image size = {key_split[1]}, learning rate = {key_split[2]}"
    
    axes[0].plot(df.epoch,df.val_loss,df.epoch,df.val_accuracy)
    axes[0].set_xlabel('epoch')
    axes[0].set_ylabel('train set metric (value loss and value accuracy)')
    
    fig.suptitle(title_, fontsize=16)
    
    axes[1].plot(df.epoch,df.loss,df.epoch,df.accuracy)
    axes[1].set_xlabel('epoch')
    axes[1].set_ylabel('test set metric (value loss and value accuracy)')
    newpath = os.path.join(os.getcwd(),"metrics/metricsviz")
    folderexist(newpath)
    plt.savefig(f"{newpath}/{title}.png")
   

def showmetrics(df,metric1,metric2):
    list_metric1 = []
    list_metric2= []
    index = list(df.keys())
    for k,v in df.items():
        list_metric1.append(v[metric1].max())
        list_metric2.append(v[metric2].min())

    df = pd.DataFrame({metric1: list_metric1,
                        metric2: list_metric2}, index=index)
    ax = df.plot.bar(rot=45, color={metric1: "green", metric2: "red"})
    
    newpath = os.path.join(os.getcwd(), 'metrics/metricsviz')
    folderexist(newpath)
    plt.savefig(os.path.join(os.getcwd(),f"{newpath}/{metric1},{metric2},{index}.png"))

def folderexist(newpath:str):
    try:
        # Create target Directory
        os.mkdir(newpath)
        print("Directory " , newpath ,  " Created ") 
    except FileExistsError:
        print("Directory " , newpath ,  " already exists")


if __name__=="__main__":
	df = LoadTensorboards()
	for k,v in df.items():
		showmcuvemodel(k,v)

	showmetrics(df,"accuracy","loss")
	showmetrics(df,"val_accuracy","val_loss")



    	


