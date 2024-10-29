<div align="center">


<h1>In Vivo Mapping of the Chemical Exchange Relayed Nuclear Overhauser Effect using Deep Magnetic Resonance Fingerprinting (rNOE-MRF) </h1>



</div>
【<a href='https://github.com/InbalPower100' target='_blank'>Inbal Power</a> |
<a href='https://mri-ai.github.io/' target='_blank'>Michal Rivlin</a>
|
<a href='https://mri-ai.github.io/' target='_blank'>Hagar Shmuely</a> |
<a href='https://mzaiss.cest-sources.org/index.php/en/' target='_blank'>Moritz Zaiss</a> | 
<a href='https://ronaz6.wixsite.com/gil-navon' target='_blank'>Gil Navon</a>

<a href='https://github.com/operlman' target='_blank'>Or Perlman</a>】
<div>
<a href='https://mri-ai.github.io/' target='_blank'>Momentum Lab, Tel Aviv University</a>
</div>
</div>

[![DOI](https://zenodo.org/badge/874612530.svg)](https://doi.org/10.5281/zenodo.14006943)


## 📚 Overview

This repository enables rNOE quantification using deep-learning-based reconstruction  following magnetic resonance fingerprinting (MRF) acquisition. See additional details in the associated paper: Power et al., iScience 2024, https://doi.org/10.1016/j.isci.2024.111209.

## ⚙️ Setup 
1. Clone the repository

2. Install the requirements

* pip enviroment:
```bash
pip install -r requirements.txt
```

* conda enviroment:
```bash
conda env create -f enviroment.yaml
```

## 🪄 Usage

To run the main script, use the following command:

python main.py --name_of_scenario [SCENARIO] --paper_example [PAPER_EXAMPLE] --path_to_acquired_data [DATA_PATH] --name_of_quant_maps [QUANT_MAPS_NAME]

Arguments

    --name_of_scenario: Scenario to run. Options are:
        0: Liver Glycogen Phantoms
        1: BSA Phantoms
        2: Mice
        3: Human
    --paper_example: Is it new data (1) or the paper example data (0)
    --path_to_acquired_data: Path to the acquired data in .mat format, with data in 'data' key
    --name_of_quant_maps: Name of the quantitative parameter maps
    Additional arguments are available for specific scenarios (see main.py for details).


## 📑 References
If you use this code for research or software development please reference the following publication:
``` 
I. Power, M. Rivlin, H. Shmuely, M. Zaiss, G. Navon, O. Perlman, ”In Vivo Mapping of the Chemical Exchange Relayed Nuclear Overhauser Effect using Deep Magnetic Resonance Fingerprinting (rNOE-MRF),” iScience, 111209, 2024. https://doi.org/10.1016/j.isci.2024.111209.
```
