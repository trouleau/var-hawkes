# Learning Hawkes Processes from a Handful of Events

Learning the causal-interaction network of multivariate Hawkes processes is a useful task in many applications. Maximum-likelihood estimation is the most com- mon approach to solve the problem in the presence of long observation sequences. However, when only short sequences are available, the lack of data amplifies the risk of overfitting and regularization becomes critical. Due to the challenges of hyper-parameter tuning, state-of-the-art methods only parameterize regularizers by a single shared hyper-parameter, hence limiting the power of representation of the model. To solve both issues, we develop in this work an efficient algorithm based on variational expectation-maximization. Our approach is able to optimize over an extended set of hyper-parameters. It is also able to take into account the uncertainty in the model parameters by learning a posterior distribution over them. Experimental results on both synthetic and real datasets show that our approach significantly outperforms state-of-the-art methods under short observation sequences.

## Install

The `varhawkes` package must be be installed to run the examples. To install it, run 

    python setup.py -e varhawkes/

Other dependencies must be installed using the `requirement.txt` file.

## Run an example

An example script is provided in the `examples` folder. To run the example, run the script `script_run_example.py` as follows:

    python script_run_example.py -d . -p params.json -o output.json

The script will read the experiment parameters in `params.json` that holds the parameters to simulate a realization of approx. 2000 events from a multivariate Hawkes process with 50 dimensions, and run all the learning algorithms discussed in the paper. Relevant performance evaluation information is printed along the run of the script.
