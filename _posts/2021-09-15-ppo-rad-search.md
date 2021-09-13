---
layout: post
title: On-policy Reinforcement Learning for Radiation Source Search
cover-img: /assets/img/path.jpg
thumbnail-img: /assets/img/rad-post/rad_robot.png
share-img: /assets/img/path.jpg
tags: [test]
comments: true
---

Detection, localization, and identification are based upon the measured gamma-ray spectrum from a radiation detector. 
Radioactive sources decay at a certain rate which, with the amount of material, gives an activity, often measured in disintegrations per second or Becquerels [bq]. 
Most decays leave the resulting nucleus in an excited state, which may lose energy by emitting energy-specific gamma rays. 
With decay branching, not all decays might emit one gamma ray, so to remove ambiguity, instead of decays per second we look at the gamma rays emitted per second [gps]. 
Localization methods in the current work rely upon the intensity in counts per second [cps] of the gamma photon radiation measured by a single, mobile scintillation detector that is searching for the source and is composed of materials such as sodium iodide (NaI). 
It is common to approximate each detector measurement as being drawn from a Poisson distribution because the success probability of each count is small and constant {% cite knolldetect %}. 
The inverse square relationship, $$\frac{1}{r^{2}}$$, is a useful approximation to describe the measured intensity of the radiation as a function of the distance between the detector and and source, r. 
<br><br>
In the case of a single mobile detector, there are numerous challenges to overcome. The nonlinear relationship paired with the probabilistic nature of gamma-ray emission/measurement and background radiation from the environment can lead to ambiguity in the estimation of a source's location. 
Common terrestrial materials such as soil and granite contain naturally occurring radioactive materials that can contribute to a spatially varying background rate.
Far distances, shielding with materials such as lead, and the presence of obstructions, i.e., a non-convex environment, can significantly attenuate or block the signal from a radioactive source. 
Further challenges arise with multiple or weak sources. 
Given the high variation in these variables, the development of a generalizable algorithm with minimal priors becomes quite difficult. 
Additionally, algorithms for localization and search need to be computationally efficient due to energy and time constraints. 
 

<div class='figure'>
    <img src="/assets/img/rad-post/rad_robot.png"
         style="width: 50%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 1.</span> A mobile robot performing active nuclear source search in a non-convex environment. The source and detector are considered isotropic. 
    </div>
</div>

A history is a sequence of observations up to timestep $$n$$, that is defined as $$H_{n}=(o_{0},o_{1},...,o_{n-1},o_{n})$$. A successful policy needs to consider $$H_{n}$$ to inform its decisions since a single observation does not necessarily uniquely identify the current state. This can be implemented directly by concatenation of all previous observations with the current observation input or through the use of the hidden state, $$h_{n}$$, of a *recurrent neural network* (RNN). The sufficient statistic $$M(H_{n})$$ is a function of the past history and serves as the basis for the agent's decision making {% cite wierstra2010recurrent %}. In this work, $$h_{n} = M_{\rho}(H_{n})$$, where $$\rho$$ denotes a RNN parameterization. This allows the control policy to be conditioned on $$h_{n}$$ as $$\pi_{\theta}(a_{n+1} \mid h_{n}) = p(a_{n+1},M_{\rho}(H_{n});\theta)$$, where $p$ is some probability distribution, $$\theta$$ is some neural network parameterization, and $$a_{n+1}$$ is the next action. The distribution $$p$$ was selected to be multinomial as the set of actions was discrete. A Gaussian distribution can be used in the case of a continuous action space.
{% bibliography --cited %}	