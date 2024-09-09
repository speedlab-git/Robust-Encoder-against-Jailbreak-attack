# Securing Vision-Language Models with a Robust Encoder Against Jailbreak and Adversarial Attacks

<!-- ![system architecture](./utils/arch.png) -->

<p align="justify">Large Vision-Language Models (LVLMs), trained on multimodal big datasets, have significantly advanced AI by excelling in vision-language tasks. However, these models remain vulnerable to adversarial attacks, particularly jailbreak attacks, which bypass safety protocols and cause the model to generate misleading or harmful responses. This vulnerability stems from both the inherent susceptibilities of LLMs and the expanded attack surface introduced by the visual modality. We propose Sim-CLIP+, a novel defense mechanism that adversarially fine-tunes the CLIP vision encoder by leveraging a Siamese architecture. This approach maximizes cosine similarity between perturbed and clean samples, facilitating resilience against adversarial manipulations.
Sim-CLIP+ offers a plug-and-play solution, allowing seamless integration into existing LVLM architectures as a robust vision encoder. Unlike previous defenses, our method requires no structural modifications to the LVLM and incurs minimal computational overhead. Sim-CLIP+ demonstrates effectiveness against both gradient-based adversarial attacks and various jailbreak techniques.
We evaluate Sim-CLIP+ against three distinct jailbreak attack strategies and perform clean evaluations using standard downstream datasets, including COCO for image captioning and OKVQA for visual question answering. Extensive experiments demonstrate that Sim-CLIP+ maintains high clean accuracy while substantially improving robustness against both gradient-based adversarial attacks and jailbreak techniques.</p>

## Contents

- [Overview](#overview)
- [Installation](#installation)
- [Fine-tuning Dataset](#dataset)
- [Adversarial Training](#adversarial-training)
- [Models](#models)
- [Jailbreak Attacks](#jbattacks)
    -[VisualAdv](#visual)
    -[ImgJP](#visual)
    -[Hades](#hades)
- [Evaluation](#evaluation)

## Overview



<p align="center">
  <img src="./assets/syswork.png" width="950" alt="accessibility text">
</p>


<p align="justify">Adversarial attacks, particularly jailbreak attacks, pose a significant threat to Large Vision-Language Models (LVLMs) by bypassing safety protocols and prompting harmful or misleading outputs. These vulnerabilities stem from the expanded attack surface introduced by the visual modality and the inherent weaknesses of LLMs. To address this, we propose Sim-CLIP+, a novel defense mechanism that enhances the robustness of the CLIP vision encoder within LVLMs. As illustrated in Figure (a), Sim-CLIP+ utilizes a Siamese architecture to fine-tune the vision encoder by maximizing the cosine similarity between clean and adversarially perturbed images. This process helps the model learn invariant features, strengthening its resistance to adversarial manipulations. Additionally, Figure (b) demonstrates how a robust Sim-CLIP+ encoder effectively blocks jailbreak attempts during LVLM inference, preventing harmful outputs. Sim-CLIP+ is a plug-and-play solution that integrates seamlessly into existing LVLM architectures, requiring no structural modifications and adding minimal computational overhead, making it a powerful tool against adversarial and jailbreak attacks.</p>





## Installation

1. Clone this repository and navigate to the SimCLIP folder:

```
git clone https://github.com/speedlab-git/Robust-Encoder-against-Jailbreak-attack.git
cd Robust-Encoder-against-Jailbreak-attack
```

2. We recommend you to use [Anaconda](https://www.anaconda.com/products/distribution) to maintain installed packages and the environment. We use **Python 3.11** for our training and evaluation. Install required packages using the following commands:

```
conda create -n robustEnocder python=3.11 -y
conda activate robustEnocder
pip install -r requirements.txt
```


### Adversarial training dataset

We adversarially pre-train CLIP on the ImageNet dataset. Please download the ImageNet dataset from [here](https://www.image-net.org/download.php) or use the following command:
If you are using windows, please use `linux subsystem (WSL)`

```
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar
```

After downloading the ImageNet dataset, extract the training and validation data using the provided script in `bash` folder:

```
./bash/imagenet/extract_ILSVRC.sh
```


## Adversarial training

In this repository, we provide scripts for running adversarial training with `FARE` and `TeCoA` alongside our proposed method, `Sim-CLIP`. We have provided bash scripts for easier execution of these training methods. Each script is tailored to run the respective training method with the necessary configurations. 

### 1. Sim-CLIP+<sup>4</sup>

```
python -m train.adversarial_training_simclip --clip_model_name ViT-L-14 --pretrained openai --dataset imagenet --imagenet_root /c/CodesSpring24/Data/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC --template std --output_normalize False --steps 10000 --warmup 1400 --batch_size 64 --loss l2 --opt adamw --lr 1e-3 --wd 1e-5 --attack pgd --attack_loss l2 --norm linf --eps 4 --iterations_adv 10 --stepsize_adv 1 --wandb True --output_dir "output directory" --experiment_name SimCLIP4 --log_freq 10
```

or execute the bash script(you can specify the training parameters inside). Make sure you are in the `SimCLIP` folder

```
./bash/training/simclip_train.sh
```


### **Note:**

- Set `--imagenet_root` with the path of your downloaded ImageNet dataset. Set `eps 2` to obtain Sim-CLIP<sup>2</sup>, FARE<sup>2</sup> and TeCoA<sup>2</sup> models
- We recommend a dual GPU setup with a total of 32 GB VRAM. If you are facing any issues with the GPU running out of memory, please reduce the `batch size`
- Modify the `output_dir` parameter to specify the directory to save the model checkpoints

## Models

| Model Name           | Type   | Proposed By                                                  | Download Link                                                                                               |
| -------------------- | ------ | ------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------- |
| CLIP                 | Clean  | [OpenAI](https://arxiv.org/pdf/2103.00020)                   | [Load CLIP model](https://huggingface.co/docs/transformers/en/model_doc/clip#transformers.CLIPModel)        |
| Sim-CLIP<sup>4</sup> | Robust | Our Method                                                   | [Download Sim-CLIP<sup>4</sup>](https://huggingface.co/hossainzarif19/SimCLIP/blob/main/simclip4.pt)        |
| Sim-CLIP<sup>2</sup> | Robust | Our Method                                                   | [Download Sim-CLIP<sup>2</sup>](https://huggingface.co/hossainzarif19/SimCLIP/blob/main/simclip2.pt)        |
| FARE<sup>4</sup>     | Robust | [Schlarmann et al. (2024)](https://arxiv.org/pdf/2402.12336) | [Download FARE<sup>4</sup>](https://huggingface.co/collections/chs20/robust-clip-65d913e552eca001fdc41978)  |
| FARE<sup>2</sup>     | Robust | [Schlarmann et al. (2024)](https://arxiv.org/pdf/2402.12336) | [Download FARE<sup>2</sup>](https://huggingface.co/collections/chs20/robust-clip-65d913e552eca001fdc41978)  |

## Usage

To use these models, you can load them using the provided code. For example, to load the Sim-CLIP<sup>4</sup> model, you can use the following code snippet:

```
import torch
import open_clip
model, _, image_processor = open_clip.create_model_and_transforms(
            'ViT-L-14', pretrained='openai', device='gpu'
        )

checkpoint = torch.load('simclip4.pt path', map_location=torch.device('gpu'))
model.vision_encoder.load_state_dict(checkpoint)
```

## Evaluation

### Zero-shot Classification

Acquire the classification dataset by visiting the Huggingface CLIP_benchmark repository at [Huggingface CLIP_benchmark](https://huggingface.co/clip-benchmark). Configure the models for evaluation in `CLIP_benchmark/benchmark/models.txt` and specify the datasets in `CLIP_benchmark/benchmark/datasets.txt`. Then execute

```

cd CLIP_benchmark
./bash/run_benchmark_adv.sh

```

### Down-stream tasks evaluation (Untargeted Attacks)

Before proceeding with Down-stream tasks evaluations, download validation annotations set from [Huggingface openflamingo repository](https://huggingface.co/datasets/openflamingo/eval_benchmark/tree/main)

### Captioning Tasks

- OpenFlamingo

  To evaluate the OpenFlamingo 9B model, first download the model from [here](https://huggingface.co/openflamingo/OpenFlamingo-9B-vitl-mpt7b/tree/main). Then, supply the downloaded annotation set and flamingo checkpoint paths in `/bash/of_eval_9B_coco.sh` . Set the `--vision_encoder_pretrained` parameter to `openai` or provide the path to a fine-tuned CLIP model checkpoint (e.g., Sim-CLIP). Finally, run the evaluation script.

```

./bash/of_eval_9B_coco.sh

```

```
./bash/of_eval_9B_Flickr.sh

```

- LLAVA

  The LLaVA model checkpoint will be automatically downloaded from repository. Update the dataset path with the location of your downloaded dataset and then execute the following command:

```

./bash/llava_eval_coco.sh

```

### Visual Question Answering Tasks

- For VQA, provide the path of OKVQA dataset in the script and then execute the following commands:

For LLAVA run

```

./bash/llava_eval_okvqa.sh

```

For Flamingo run

```

./bash/of_eval_9B_okvqa.sh

```

## Targeted attacks

To perform targeted attacks with the LLAVA model on the COCO or Flickr30k dataset, please run these steps:


```
./bash/eval_targeted.sh
```

**Note**: Default target strings can be updated in `run_evaluation.py`

For targeted attacks on custom images, update `vlm_eval/run_evaluation_qualitative.py` with your images and captions, then execute:

```
python -m vlm_eval.run_evaluation_qualitative --precision float32 --attack apgd --eps 2 --steps 10000 --vlm_model_name llava --vision_encoder_pretrained openai --verbose
```
**Note**: 
To increase the strength of the attack, modify the `--attack` parameter with higher steps in the bash script. A higher attack step size results in a stronger attack.







## Results

<p align="justify">
The tables provided illustrate the robust performance of various vision-language models (VLMs) using different versions of the CLIP model across two main tasks: image captioning and visual question answering (VQA). For the image captioning tasks on COCO and Flickr30k datasets, we evaluate the models using the CIDEr score, which quantifies the consensus between a candidate caption and reference captions. Higher CIDEr scores indicate a better capture of semantic meaning and relevance in the generated captions. For VQA tasks on VizWiz and OKVQA datasets, we report accuracy to measure the model's ability to correctly interpret and answer questions based on visual content.
From the results, Sim-CLIP models, particularly Sim-CLIP<sup>4</sup>, consistently outperform other robust models like FARE and TECOA across most datasets in both clean and adversarial settings. This suggests that Sim-CLIP's adversarial fine-tuning approach not only enhances robustness against adversarial attacks but also preserves or even improves the model's ability to understand and generate semantically meaningful responses in complex multimodal tasks.
In adversarial scenarios, where models are evaluated at an epsilon value of 4/255, Sim-CLIP models demonstrate superior resilience, maintaining higher performance metrics compared to other models. This robustness is crucial for practical applications where models might encounter manipulated or adversarially altered inputs.
</p>


- ### Clean Evaluation



<table>
    <thead>
        <tr>
            <th>VLM</th>
            <th>Vision Encoder</th>
            <th>COCO</th>
            <th>Flickr30k</th>
            <th>VizWiz</th>
            <th>OKVQA</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan="7">Open Flamingo</td>
            <td>Sim-CLIP-2</td>
            <td>85.6</td>
            <td>56.3</td>
            <td>21.8</td>
            <td>35.1</td>
        </tr>
        <tr>
            <td>FARE-2</td>
            <td>84.3</td>
            <td>53.1</td>
            <td>22.1</td>
            <td>34.5</td>
        </tr>
        <tr>
            <td>TECOA-2</td>
            <td>74.5</td>
            <td>48.2</td>
            <td>22.3</td>
            <td>33.6</td>
        </tr>
        <tr>
            <td>Sim-CLIP-4</td>
            <td>81.6</td>
            <td>54.5</td>
            <td>20.0</td>
            <td>32.0</td>
        </tr>
        <tr>
            <td>FARE-4</td>
            <td>81.4</td>
            <td>51.8</td>
            <td>16.4</td>
            <td>31.8</td>
        </tr>
        <tr>
            <td>TECOA-4</td>
            <td>71.0</td>
            <td>45.6</td>
            <td>19.3</td>
            <td>31.0</td>
        </tr>
        <tr>
            <td>CLIP</td>
            <td>60.5</td>
            <td>41.7</td>
            <td>18.0</td>
            <td>28.2</td>
        </tr>
        <tr>
            <td rowspan="7">LLaVA 1.5</td>
            <td>Sim-CLIP-2</td>
            <td>109.1</td>
            <td>66.3</td>
            <td>45.8</td>
            <td>54.5</td>
        </tr>
        <tr>
            <td>FARE-2</td>
            <td>108.1</td>
            <td>65.1</td>
            <td>43.8</td>
            <td>54.1</td>
        </tr>
        <tr>
            <td>TECOA-2</td>
            <td>85.3</td>
            <td>60.3</td>
            <td>41.6</td>
            <td>55.3</td>
        </tr>
        <tr>
            <td>Sim-CLIP-4</td>
            <td>103.6</td>
            <td>65.0</td>
            <td>41.3</td>
            <td>54.6</td>
        </tr>
        <tr>
            <td>FARE-4</td>
            <td>105.1</td>
            <td>64.0</td>
            <td>41.7</td>
            <td>54.5</td>
        </tr>
        <tr>
            <td>TECOA-4</td>
            <td>79.1</td>
            <td>57.7</td>
            <td>39.4</td>
            <td>50.0</td>
        </tr>
        <tr>
            <td>CLIP</td>
            <td>75.2</td>
            <td>50.3</td>
            <td>37.5</td>
            <td>48.1</td>
        </tr>
    </tbody>
</table>

- ### Adversarial evaluation at $\epsilon$ = 4/255 radii

<table>
    <thead>
        <tr>
            <th>VLM</th>
            <th>Vision Encoder</th>
            <th>COCO</th>
            <th>Flickr30k</th>
            <th>VizWiz</th>
            <th>OKVQA</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan="7">Open Flamingo</td>
            <td>Sim-CLIP-2</td>
            <td>58.4</td>
            <td>35.1</td>
            <td>13.6</td>
            <td>19.7</td>
        </tr>
        <tr>
            <td>FARE-2</td>
            <td>53.5</td>
            <td>34.3</td>
            <td>12.3</td>
            <td>17.1</td>
        </tr>
        <tr>
            <td>TECOA-2</td>
            <td>40.3</td>
            <td>27.4</td>
            <td>10.6</td>
            <td>15.3</td>
        </tr>
        <tr>
            <td>Sim-CLIP-4</td>
            <td>60.5</td>
            <td>39.2</td>
            <td>15.7</td>
            <td>22.0</td>
        </tr>
        <tr>
            <td>FARE-4</td>
            <td>56.1</td>
            <td>37.6</td>
            <td>13.7</td>
            <td>19.2</td>
        </tr>
        <tr>
            <td>TECOA-4</td>
            <td>50.3</td>
            <td>32.9</td>
            <td>14.7</td>
            <td>20.5</td>
        </tr>
        <tr>
            <td>CLIP</td>
            <td>5.6</td>
            <td>3.8</td>
            <td>1.8</td>
            <td>0</td>
        </tr>
        <tr>
            <td rowspan="7">LLaVA 1.5</td>
            <td>Sim-CLIP-2</td>
            <td>69.4</td>
            <td>42.3</td>
            <td>33.4</td>
            <td>33.9</td>
        </tr>
        <tr>
            <td>FARE-2</td>
            <td>68.3</td>
            <td>41.6</td>
            <td>30.1</td>
            <td>31.8</td>
        </tr>
        <tr>
            <td>TECOA-2</td>
            <td>61.1</td>
            <td>36.1</td>
            <td>25.3</td>
            <td>24.1</td>
        </tr>
        <tr>
            <td>Sim-CLIP-4</td>
            <td>72.9</td>
            <td>46.3</td>
            <td>35.2</td>
            <td>36.2</td>
        </tr>
        <tr>
            <td>FARE-4</td>
            <td>71.7</td>
            <td>43.6</td>
            <td>33.7</td>
            <td>34.0</td>
        </tr>
        <tr>
            <td>TECOA-4</td>
            <td>65.5</td>
            <td>39.4</td>
            <td>28.4</td>
            <td>27.2</td>
        </tr>
        <tr>
            <td>CLIP</td>
            <td>13.5</td>
            <td>10.0</td>
            <td>3.2</td>
            <td>0.0</td>
        </tr>
    </tbody>
</table>


## Acknowledgements

We extend our gratitude to the developers and contributors of the following repositories for their invaluable resources and tools that have significantly aided in the development and evaluation of our project:

- [OpenFlamingo](https://github.com/mlfoundations/open_flamingo)
- [LLaVA](https://github.com/haotian-liu/LLaVA)
- [CLIP Benchmark](https://github.com/LAION-AI/CLIP_benchmark)
- [AutoAttack](https://github.com/fra31/auto-attack)
- [RobustVLM](https://github.com/chs20/RobustVLM)


