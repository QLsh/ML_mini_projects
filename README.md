# ML_mini_projects
This is a repository for some of my past ML mini-projects from my master course module. :) 

## COVID-19_prediction
Since the pandemic started, scientists have been analyzing the circumstances using COVID related data. With some useful data analysis tools, the trend of the pandemic and even its impact in many other fields such as economy, industries and etc. Scientist can then use the tools to find a model and give reliable advice to governments with the support of figures to help them make better decisions.

In this project, the Data analysis will be mainly carried out as Regression analysis and further forecasting with this model. The data sets are provided by UK gov website https://coronavirus.data.gov.uk/ and goodle mobility https://www.google.com/covid19/mobility/. For the government data, a packge called uk_covid19 is created for convinience of studies and can be installed directly in our environment.

This project also requires certain proficiency in using Pandas.dataFrame and Sklearn.LinearRegression, as they are the most important skills tested in this assessment.

## Classification_of_waveforms


This projet is about designing a 1D Convolutional Neural Net (CNN) that classifies digitised signals from two type of scintillating materials used to record particle energy.

There are many signal processing tasks where it is important to separate the signals recorded in different categories. Due the complex features of the signals recorded, a computer vision solution is well suited for solving this problem.

We are dealing with two type of signals:

    The first type of material is an organic scintillator (PVT) with a fast time response of a few nanoseconds. These fasts signals are also called Electron Scintillation (ES) signals.

    The second type of material is an inorganic scintillator to detect neutrons (ZnS(Ag)), one of the oldest scintillator used. When a neutron is detected, the scintillation signal a long trail of fast pulses that slowly decreases in amplitude. These slow signals are called Nuclear Scintillation (NS) signals.

We are interested here to separate as well as possible each type of signal and therefore coming up with a model that is able to have a very high score at predicting each class of signal.

Numpy files for the training and testing datasets (xtrain.npy, ytrain.npy) and their labels are already prepared. Each digitised waveform is a 1000 samples long with recorded pulses starting at a fixed time around 250 samples. Some examples of how to look at each signals and their label is provided below. Both type of signals have their amplitude normalised to the smallest pulse detectable i.e. what we call the 1 Photo-electron pulse.


## Neural-ODEs
We introduce a new family of deep neural network models. Instead of specifying a discrete sequence of hidden layers, we parameterize the derivative of the hidden state using a neural network. The output of the network is computed using a black-box differential equation solver. These continuous-depth models have constant memory cost, adapt their evaluation strategy to each input, and can explicitly trade numerical precision for speed. We demonstrate these properties in continuous-depth residual networks and continuous-time latent variable models. We also construct continuous normalizing flows, a generative model that can train by maximum likelihood, without partitioning or ordering the data dimensions. For training, we show how to scalably backpropagate through any ODE solver, without access to its internal operations. This allows end-to-end training of ODEs within larger models.


## Simulation_of_e+ 𝑒−_Collison

When electrons $(e^-)$ and positrons $(e^+)$ collide together they sometimes produce two muons $(\mu^- , \mu^+)$ that come out of the collision back-to-back. The angle between the incoming $e^+$ and the outgoing $\mu^+$ is defined to be $\theta$. See the figure below:
![simple_drawing.png](attachment:simple_drawing.png)

If the energy of the $e^-e^+$ collision is a long way below the mass of the $Z$ boson then $\theta$ has a distribution of $1+\cos^2\theta$. However, as the centre of mass energy of the collision approaches that of the $Z$ boson an asymmetry appears and the distribution becomes Eq.1 as shown below:

$$\cal{A}(1+\cos^2\theta) + \cal{B} \cos \theta $$

which can be written $(1+\cos^2\theta) + \dfrac{\cal{B}}{\cal{A}}\cos \theta$. The ratio $\dfrac{\cal{B}}{\cal{A}}= \kappa$ varies greatly (up to a maximum of ~10%) around the collision energies near the centre of mass of the $Z$ boson and even changes sign. In the 1990s measuring these assymetries was an important scientific goal as they told us much about electroweak unification.

In the early 1990s the LEP collider at CERN was one of the first  $e^-e^+$ colliders to run near the $Z$ mass. 
The aim of the mini-project is to determine how well the experiments could measure $\kappa$ in three different scenarios:

1. When LEP started running each experiment would collect a few 10s(20 or 30) of these events/day and (say) would run for 100 days a year.

2. A couple of years later each experiment would collect a few 100s(200 or 300) of these events/day and (say) would run for 100 days a year.

3. A couple of years later again each experiment would collect a few 1000s(2000 or 3000) of these events/day and (say) would run for 100 days a year.

Consider (and simulate) three different values for $\kappa = \pm 0.07$ and $0$. You should fit your simulation to consider how well you can tell them apart and what precision can you make on the individual measurements. 




## SoLid_experiment


This project is a typical classification problem in HEP where the experimental data contains large amount of background events and very few signal events (less than ~1% of total).

The aim is to develop a simpler version of the analysis conducted with the first detector module SM1 to explore various multi-variate models. If you are interested, results from this analysis where published in article below to estimate the background expected for the first phase of the full scale experiment:

https://iopscience.iop.org/article/10.1088/1748-0221/13/05/P05005

The introductory material provided during the course should give you the necessary background to understand the dataset and the approach used to separate the signal and background. The paper is worth a read as it offers more in-depth description of the detector system and the measurements obtained

This exercise is done exclusively in Python and with the aim of introducing the basic classification tools and techniques that are use in most classification problems using some very prowerful library like scikit learn

The project is split in two parts :

    The first part is to learn about the dataset, visualise the data and make sense of the metrics used to optimise the task. It focuses on developing a simple cut and count analysis based on 1 dimensional selection cut
    The second part is about developing the multi-variate analysis using machine learning algorithms like Support Vector Machines or Neural Nets to achieve superiro performance.
