Pokedex
=====================

This project is an implementation of a Pokedex using deep neural networks such as AlexNet and ResNet.

Requirements
============

Before running the code, you have several libraries to install. To make your life easier, we have made a virtual environment using pipenv.

Run the virtual environment :
-----------

Go the root of the project (i.e. in the *projet* folder) and follow the instruction bellow :

- verify that pip is installed :

    $ pip --version

if not, [install it](https://pip.pypa.io/en/stable/installing/).

When it is done, let's continue the installation :

- install pipenv :

    $ pip install pipenv

- once it is installed and well configured, run your environment (when being at the root folder) :

    $ pipenv install
    
Now, everything should be well installed and the code should run properly. However, considering the fact that we're using Keras,
the installation of TensorFlow could not work. If so, follow [TensorFlow official installation guide](https://www.tensorflow.org/install?hl=fr).

*Note : if the installation of all the required libraries failed, just open the pipfile file, check the name of the used 
libraries and their version and install them manually using pip.*

Train a neural network :
-----------

If you want to train a neural network, make sure you're in the virtual environment. Then, chose the config you want to run by 
modifying the "config_example.yaml" file. After doing that, follow the instruction bellow :

    $ cd pokedex
    $ python -m classification -cf path_to_your_config_file
    
Now, the training should run and it should have created a folder named *training* where everything should be saved.

Visualize the training :
-----------

You can either chose to use Keras's callback Tensorboard or our custom graphs. For the first one, place yourself 
in one of your training's folder (assuming you're still in the *pokedex* directory) :

    $ cd training
    $ cd used_dataset_name
    $ cd your_training_name
    $ tensorboard --logdir tensorboard_logs

Then just follow the instruction printed in the terminal.

If you want to use our custom graphs, copy and paste the .csv of your training into the folder *metrics*. Then, run the 
following command lines (assuming you're in the *pokedex* directory) :

    $ cd classification
    $ python models_metric_viz.py
    
It should have created a folder into the *metrics* folder, named *metricsviz* and containing graphs and charts of your training.

Output the confusion matrix :
-----------

Copy and paste the .h5 of your training into the folder *best_models*. Modify the variable "chosen_model"
accordingly to your model's name (don't forget to modify the "target_size" to match the size of the images used during the
training. Then, just do the following (assuming you're in the *pokedex* directory) :

    $ cd classification
    $ python confusion_matrix.py
    
It should have displayed you the confusion matrix and created a folder into *best_models_, named *eval* and
containing a new .csv file with interesting values such as the recall, the precision...

Inference (working Pokedex) :
-----------

Now, if you want to test your trained model, put your model into the *best_models* folder. Then, do the following 
(assuming you're in the *pokedex* directory):

    $ cd classification
    $ python predict.py
    
Again, you have to first check that the parameter "inputs" at the end of the *predict.py* file matches 
the size of the images used during the training. Then, open the link prompted in the terminal, drag a pokemon image 
into the dedicated area and hit *submit*. After a little while, it should output the predicted label.

You can also put a second/third/... model into the same folder as before (and having the same input images' size) to
compare the two/three/...

Split a dataset :
-----------

Place your dataset folder into the *pokedex* directory and make sure you have a folder named *data* at the same
level. Then, after modifying the variable "dataset_name" accordingly to your dataset name and run the following
(assuming you're in the *pokedex* directory):

    $ cd classification
    $ python created_dataset.py
    
It should have created your split dataset into the *data* folder. You can choose the splitting ratio by modifying the 
value of "ratio" at the end of the file.

Run the solution in Google Colab :
-----------

To run this solution in Colab you need to : 

- upload the directory in your Drive Homepage
- Open a new Colab Notebook
- editter le fichier de config et changer   root_folder_path: "/content/gdrive/MyDrive/Pokedex/pokedex"

- excute this instructions :
    
        !pip install tensorflow
        import tensorflow
        from google.colab import drive
        drive.mount('/content/gdrive/',force_remount=True)
        !python3 "/content/gdrive/MyDrive/Pokedex/pokedex/classification/the_script_you_want_to_run.py"
