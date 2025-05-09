[![INFORMS Journal on Computing Logo](https://INFORMSJoC.github.io/logos/INFORMS_Journal_on_Computing_Header.jpg)](https://pubsonline.informs.org/journal/ijoc)

# Combating Fake News on Social Media: An Early Detection Approach using Multimodal Adversarial Transfer Learning
This archive is distributed in association with the [INFORMS Journal on Computing](https://pubsonline.informs.org/journal/ijoc) under the [MIT License](LICENSE.txt).

The software and data in this repository are a snapshot of the software and data that were used in the research reported on in the paper [Combating Fake News on Social Media: An Early Detection Approach using Multimodal Adversarial Transfer Learning](https://10.1287/ijoc.2023.0514) by Cong Wang and Chuchun Zhang and Runyu Chen.

# Cite
To cite the contents of this repository, please cite both the paper and this repo, using their respective DOIs.

https://10.1287/ijoc.2023.0514

https://10.1287/ijoc.2023.0514.cd

Below is the BibTex for citing this snapshot of the repository.
```
@misc{Wang2025,
  author =        {Cong Wang and Chuchun Zhang and Runyu Chen},
  publisher =     {INFORMS Journal on Computing},
  title =         {{Combating Fake News on Social Media: An Early Detection Approach using Multimodal Adversarial Transfer Learning}},
  year =          {2025},
  doi =           {10.1287/ijoc.2023.0514.cd},
  url =           {https://github.com/INFORMSJoC/2023.0514},
  note =          {Available for download at https://github.com/INFORMSJoC/2023.0514},
}  
```


# Installation
First you need to install Python dependencies by running:
```bash
pip install -r requirements.txt
```
# Run
To execute our model, run:
```bash
python src/main.py
```

# Folder Structure
   * `src` Contains our model implementation and training process code

   * `data` Contains provided source and target data
     
   * `scripts` 
    - Includes variants of our model:
    - Model without transfer learning module
    - Models using single modalities (image-only/text-only)
    - Other experimental variations


   * `results` 
    - Contains:
    - Ablation study results
    - Code for generating result visualizations


