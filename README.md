# ML_freqHop
## GitHub
Here is a qick [tutorial](https://product.hubspot.com/blog/git-and-github-tutorial-for-beginners) on how to use GitHub.
It's pretty comprehensive and should provide a quick starting point.

## Univariate LSTM
### File: univariate.py
The code comes form a 
[tutorial](https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/) 
from machinelearningmastery.com.
It produces a pure sinusoide and has the option to add Gaussian White Noise to the it.
There are options to train and test on the same types of data or on different sets (i.e. train on clean and test on noisy).

Depending on your environment, you may not need to have tensorflow in order to import from keras.

The main program begins after the functions.

## Multivariate LSTM
### Data
all_data.csv and all_data2.csv contains the data generated for 2 frequency hopping cycles.
(The hopper hops to each channel twice.)

3_cycles.csv contians the data generated for 3 frequency hopping cycles.
(The hopper hops to each channel thrice.)

### File: multivariate.py
The code comes form a
[tutorial](https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/)
from machinelearningmastery.com.
It predicts the status of a frequency hopper on Bluetooth (79 channels) using the one of the data files described above.

Again, importing tensorflow may not be necessary to run keras depending on your environment.

The main program begins with loading the data file, as shown below:
```
df = init('3_cycles.csv')
```
The number of neurons, epochs, and the batch size can be changed under the comment design network.
```
#design network
neurons = 80
batchSize = 5
e=50
```
