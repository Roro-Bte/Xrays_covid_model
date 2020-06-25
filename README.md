# Xrays covid19 model
The complete report can be found [here](https://github.com/Roro-Bte/Xrays_covid_model/blob/master/Machine%20Learning%20Capstone%20Project.docx).


The problem is a classification for four classes.  The proposed model classifies between normal, and three types of pneumonia: bacteria, virus (not covid-19), and covid-19.
To get started, please check the Dependencies section. Please note that the code was created both in a local machine and AWS because of the data augmentation process, which copies a big amount of data. This data manipulation was decided to be done in the local computer and then uploaded with [cyberduck](https://cyberduck.io/download/).

## Data augmentation
The sections before training have to be done outside of AWS, on your local PC. 
Inside the notebook, the necessary folders are extracted and created for the training and testing data. Get 8 covid-19 images and move them to the covid-19 test folder. This is because no images came in that destination folder. The other 50 images stay, and both folders are augmented with the code in the notebook.  

## Uploading the training files
In AWS, create a new role that has full access to S3. AWS then creates a "Access key ID" and a secret key. Please store that information to enter it in Cyberduck.
Upload the newly created folders with the data to S3. 

## Uploading the testset
Upload a zip for the testset directly to the AWS notebook instance, and extract it. 

## Training
From here on the notebook is meant to be run from a notebook instance in AWS. There is only one Pytorch estimator defined, but run several times with different names for the different models. Four models were run in total for the final report, using the ml.p2.xlarge machine. Even though the ml.p2.8xlarge was tried, the economy of time was less than the higher price. 
After training, the output of the training and validation losses are imported and graphed. To save the graph, click on the little camera icon on the top right corner.  
The validation size used by default is 15% of the training data. 

## Testing
The code accesses a local test folder in the notebook instance instead of taking the information directly from S3, as opposed to the training estimator that acceses directly. That is why the testset must be uploaded directly to the notebook instance.
A Pytorchmodel object is deployed, and it must have the path (or name) of the uri that contains the model artifacts in S3. In this way, the model can access an old saved model for deployment. 




# Dependencies

## Configure and Manage Your Environment with Anaconda

Per the Anaconda [docs](http://conda.pydata.org/docs):

> Conda is an open source package management system and environment management system 
for installing multiple versions of software packages and their dependencies and 
switching easily between them. It works on Linux, OS X and Windows, and was created 
for Python programs but can package and distribute any software.

## Overview
Using Anaconda consists of the following:

1. Install [`miniconda`](http://conda.pydata.org/miniconda.html) on your computer, by selecting the latest Python version for your operating system. If you already have `conda` or `miniconda` installed, you should be able to skip this step and move on to step 2.
2. Create and activate * a new `conda` [environment](http://conda.pydata.org/docs/using/envs.html).

\* Each time you wish to work on this project, activate your `conda` environment!

---

## 1. Installation

**Download** the latest version of `miniconda` that matches your system.

|        | Linux | Mac | Windows | 
|--------|-------|-----|---------|
| 64-bit | [64-bit (bash installer)][lin64] | [64-bit (bash installer)][mac64] | [64-bit (exe installer)][win64]
| 32-bit | [32-bit (bash installer)][lin32] |  | [32-bit (exe installer)][win32]

[win64]: https://repo.continuum.io/miniconda/Miniconda3-latest-Windows-x86_64.exe
[win32]: https://repo.continuum.io/miniconda/Miniconda3-latest-Windows-x86.exe
[mac64]: https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
[lin64]: https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
[lin32]: https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86.sh

**Install** [miniconda](http://conda.pydata.org/miniconda.html) on your machine. Detailed instructions:

- **Linux:** http://conda.pydata.org/docs/install/quick.html#linux-miniconda-install
- **Mac:** http://conda.pydata.org/docs/install/quick.html#os-x-miniconda-install
- **Windows:** http://conda.pydata.org/docs/install/quick.html#windows-miniconda-install

## 2. Create and Activate the Environment

For Windows users, these following commands need to be executed from the **Anaconda prompt** as opposed to a Windows terminal window. For Mac, a normal terminal window will work. 

#### Git and version control
These instructions also assume you have `git` installed for working with Github from a terminal window, but if you do not, you can download that first with the command:
```
conda install git
```

If you'd like to learn more about version control and using `git` from the command line, take a look at [free course: Version Control with Git](https://www.udacity.com/course/version-control-with-git--ud123).

**Now, we're ready to create our local environment!**

1. Clone the repository, and navigate to the downloaded folder. This may take a minute or two to clone due to the included image data.
```
git clone https://github.com/Roro-Bte/Xrays_covid_model.git
cd Xrays_covid_model
```

2. Create (and activate) a new environment, named `SCEngineer` with Python 3.6. If prompted to proceed with the install `(Proceed [y]/n)` type y.

	- __Linux__ or __Mac__: 
	```
	conda create -n deep-learning python=3.6
	source activate deep-learning
	```
	- __Windows__: 
	```
	conda create --name deep-learning python=3.6
	activate deep-learning
	```
	
	At this point your command line should look something like: `(SCEngineer) <User>:Xrays_covid_model <user>$`. The `(SCEngineer)` indicates that your environment has been activated, and you can proceed with further package installations.

3. Install a few required pip packages, which are specified in the requirements text file.
```
pip install -r requirements.txt
```

Now, assuming your `SCEngineer` environment is still activated, you can navigate to the main repo and start looking at the notebooks:

```
cd
cd Xrays_covid_model
jupyter-notebook
```

To exit the environment when you have completed your work session, simply close the terminal window.


# Issues
-Finding a way to access S3 from the deployed model.
