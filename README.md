# Code for HAN experiment - Mashrur

This is a code adaptation of Deutsch, Jasbi and Shieber(2020). I have modified it to use it with the twitter corpus provided by Jacod and Uitdenbogerd (2019)

<br>

## Declaration :
<hr>
Majority of the code is taken from https://github.com/TovlyDeutsch/Linguistic-Features-for-Readability 

The model described in the associated research paper and the code has been tweaked to achieve our purposes.  

<br>

## Results and Evaluation
<hr>
The results should be contained in <code>out/Evaluation.txt</code>. The model was run with 5 k-folds. Learning Rate: 0.0001, Batch Size : 64. 20 Epochs, with patience of 10 and min_delta of 0.0001.

<br><br>

## How to run the experiment? 
<hr>

### HAN Model  
Ensure the data is in <code>myCorpus.csv</code> or provide alternate file path in <code>opt</code> dictionary in the <code>main.py</code>. All the configurations for running the model is contained in <code>main.py</code>. For changing the model itself, please modify: <code>HAN.py, word_att_model.py</code> and <code>sent_att_model.py</code>. 

### SVM  
The data along with the feautes is in <code>svm_data_with_HAN.csv</code> which contains the HAN output as a feautre to train the SVM on. 

To get HAN's output, <code>test_params = {"batch_size": 64,
                   "shuffle": False,
                   "drop_last": False}</code>

To train the SVM, I have used 5-fold Cross Validation.  
