# spin-models

This repository is dedicated for my first research into spin models. Simpliest spin models, such as Ising or voter, are used extensively to study opinion dynamics in sociophysics. For my bachelor thesis research I used spin models to create and identify spatial structures, such as domains, clusters. Spatiality is a new look into dynamics proposed by statistical physics models since temporal change has already been widely studied. This also makes more sense from sociophysics point of view - empirical and experimental data is more often available in different scales and/or geographical distributions rather than time-scale.

One might ask - why look for new ways to define domains and/or clusters when they can typically be easily approached with correlation or similar to such functions? Those functions are of little use when we want to compare these spatial units with the ones reproduced by empirical data. I compared results from said models (and their interpretations) with United Kingdom 2011 census data. This data can be found on nomis website, available at many different scales (for comparison I used 9) and demographic indicators (used 6 indicators with specific sub-category of each). My random spatial grouping model and diversity index (proposed by me and my thesis supervisor) show that spatial distributions are not of random nature and for some even thend to be polarised.

Uploaded files with .py extensions contain functions to be used for implementing modeling algorithms. They are as follow:
**voter.py** - functions to implement noisy voter model, where noise is used with probability parameter p.
**ising.py** - functions to implement Ising model by Metropolis interpretation with according acceptance ratios.
**kawasaki.py** - functions to implement Ising model by Kawasaki interpretation. For modelling speed a global interaction mechanism is used, which changes the dynamics of equilibriation. Final result is the same.
**wolff.py** - functions to implement Ising model by Wolff interpretation, the best one to simulate in critical temperature point.
**scale.py** - functions to implement diversity index calculations (random spatial grouping model is one instance of diversity index, namely _I = 0_).
**scaling-new.py** - new set of functions to implement diversity index calculations (_i_ at this case). Idea of this stems from grouping clusters in Ising model by Wolff interpretation.

The .pdf document contains text of my final thesis.

**Notebooks** with extension .ipynb contain some of jupyter notebooks where said functions are used for simulations.

New uploads (check commit history) include my submissions to scientific conferences - OpenReadings 2022 and FizTech 2022. **They very briefly and accurately describe methods and scope of this work.**

Part of the code was written by my thesis supervisor Aleksejus Kononovicius. Please see http://web.vu.lt/tfai/a.kononovicius/ for more of his work and research in sociophysics.
