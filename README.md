# Diffusion-UIO

This repo is the official implementation of the following paper:

**[Under Review] Diffusion-Driven Hybrid Unknown Input Observer for Vehicle Dynamics Estimation**
<br> [Cheng Tian](https://scholar.google.com/citations?user=OIlgz_gAAAAJ&hl=en), [Anh-Tu Nguyen](https://scholar.google.com/citations?user=eE6A1aIAAAAJ&hl=fr), [Edward Chung](https://scholar.google.com/citations?user=UFrzhnMAAAAJ&hl=en), [Hailong Huang](https://scholar.google.com/citations?user=ulsViyoAAAAJ&hl=en)
 
<br> [AIMS Research Group, The Hong Kong Polytechnic University](https://sites.google.com/view/hailong-huang/home)
<br> [INSA Hauts-de-France, Université Polytechnique Hauts-de-France](https://sites.google.com/view/anh-tu-nguyen)

## Experimental Results
The proposed framework is validated with the test data collected from a real-world test track.

![](table_4.png)

Methods:
- ```TS-UIO```: The TS fuzzy UIO [[Link]](https://ieeexplore.ieee.org/document/9314225)
- ```E2E-UIO```: The designed LPV UIO with an end-to-end approximator (modified from [[Link]](https://ieeexplore.ieee.org/document/10054430))
- ```Diff-UIO```: The proposed diffusion-driven hybrid UIO

## Get Started

### 1.Codes
Full code will be released soon.

### 2.Downloads
Model weights will be released soon.
      
## Useful link

If you are interested in other SOTA state estimation techniques of Vehicle-Road-Pedestrian, you can refer to our survey paper:

  ```
@ARTICLE{tian_recentestimationtechniques_2025,
  author={Tian, Cheng and Huang, Chao and Wang, Yan and Chung, Edward and Nguyen, Anh-Tu and Wong, Pak Kin and Ni, Wei and Jamalipour, Abbas and Li, Kai and Huang, Hailong},
  journal={IEEE Transactions on Intelligent Transportation Systems}, 
  title={Recent Estimation Techniques of Vehicle-Road-Pedestrian States for Traffic Safety: Comprehensive Review and Future Perspectives}, 
  year={2025},
  volume={26},
  number={3},
  pages={2897-2920}
}
```
Full-text link: [ResearchGate](https://www.researchgate.net/publication/387093260_Recent_Estimation_Techniques_of_Vehicle-Road-Pedestrian_States_for_Traffic_Safety_Comprehensive_Review_and_Future_Perspectives) [IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/10814926) 

## TODO
- [x] Upload the initial Readme
- [x] Release the codes
- [ ] Update code usage tutorial

## Acknowledgement

We acknowledge the PROCORE-France/Hong Kong Joint Research Scheme for supporting this study.


