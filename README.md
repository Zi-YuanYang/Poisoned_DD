 # <p align=center>  Poisoned Distillation: Injecting Backdoors into Distilled Datasets without Raw Data Access
 <div align="center">
[![arXiv](https://img.shields.io/badge/Poisoned_DD-arXiv-red.svg)](https://arxiv.org/abs/2502.04229)
</div>

---
>**Poisoned Distillation: Injecting Backdoors into Distilled Datasets without Raw Data Access**<br>  [Ziyuan Yang](https://zi-yuanyang.github.io/), [Ming Yan](https://scholar.google.com/citations?hl=zh-CN&user=nRknP8UAAAAJ), [Yi Zhang](https://deep-imaging-group.github.io/)<sup>* </sup>, [Joey Tianyi Zhou](https://joeyzhouty.github.io/)<sup>* </sup> <br>
(* Corresponding author)<br>

> **Abstract:** *Dataset distillation (DD) condenses large datasets into smaller synthetic ones to enhance training efficiency and reducing bandwidth. DD enables models to achieve comparable performance to those trained on the raw full dataset, making it popular for data sharing. Existing work shows that injecting backdoors during the distillation process can threaten downstream models. However, these studies assume attackers can have access to the raw dataset and interfere with the entire distillation process, which is unrealistic. In contrast, this work is the first to address a more realistic and concerning threat: attackers may intercept the dataset distribution process, inject backdoors into the distilled datasets, and redistribute them to users. While distilled datasets were previously considered resistant to backdoor attacks, we demonstrate that they remain vulnerable to such attacks. Furthermore, we show that attackers do not even require access to any raw data to inject the backdoors successfully within one minute. Specifically, our approach reconstructs conceptual archetypes for each class from the model trained on the distilled dataset. Backdoors are then injected into these archetypes to update the distilled dataset. Moreover, we ensure the updated dataset not only retains the backdoor but also preserves the original optimization trajectory, thus maintaining the knowledge of the raw dataset. To achieve this, a hybrid loss is designed to integrate backdoor information along the benign optimization trajectory, ensuring that previously learned information is not forgotten. Extensive experiments demonstrate that distilled datasets are highly vulnerable to our attack, with risks pervasive across various raw datasets, distillation methods, and downstream training strategies. *

#### Citation
If our work is valuable to you, please cite our work:

```
@InProceedings{Yang_2026_DDPoison,
    author    = {Yang, Ziyuan and Yan, Ming and Zhang, Yi and Zhou, Joey Tianyi},
    title     = {Poisoned Distillation: Injecting Backdoors into Distilled Datasets Without Raw Data Access},
    booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
    year      = {2026}
}
```
