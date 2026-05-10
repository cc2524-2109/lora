# README
# 1. Introduction
The purpose of this repo is a project in an attempt to reimplement the LoRA: Low-Rank Adaptation of Large Language Models paper. This paper introduces the idea of using a LoRA when training a model, which reduces the trainable parameters by 1000x, making training significantly cheaper and faster without reducing the accuracy of the model.
# 2. Chosen Result
With this repo, we specifically aimed to reproduce Table 3 of the paper, which is a detailed benchmark and comparison of GPT2 Medium and Large with a variety of methods, including full fine-tuning and LoRA. It is vital in showing the aforementioned effectiveness/validity of LoRA.

<img width="917" height="429" alt="Table 3 of the paper" src="https://github.com/user-attachments/assets/dadaa6b4-fc08-4398-8291-2c0cd45f39b6" />

# 3. GitHub Contents
This Github repo contains our code for processing the dataset, injecting LoRA, training our model, and evaluating our results. Simply run main.ipynb after editing the local directory path to properly load and save files.
# 4. Re-implementation Details
Our re-implementation consists of creating a GPT2 wrapper class that decomposes the pretrained weight matrices into Q, K, and V, and injects low-rank decomposition matrices (A and B) into each target weight matrix (Q and V). Then, the LoRA-injected model (GPT2-medium and GPT-large) is trained on the E2E NLG dataset. 
# 5. Reproduction Steps 
Download repo and upload to Google Drive. Adjust base directory in cell 3 of main.ipynb to reflect folder location of repo in Google Drive. All required dependencies/libraries are downloaded by running all cells in main.ipynb in Google Colab. Minimum of T4 GPU needed to run main.ipynb. 
# 6. Results/Insights
Expect slightly lower scores than the paper, while still validating LoRA's claim of competitive performance with only 0.10% of trainable parameters. 

<img width="641" height="545" alt="Screenshot 2026-05-10 144242" src="https://github.com/user-attachments/assets/335aa079-90b9-4a90-80ac-13453aa53942" />

# 7. Conclusion
Our results confirm that the paper’s core claim still holds. Despite the metric gap, our results reproduce the qualitative finding that adapting only Wq and Wv with r = 4 yields just 0.35M trainable parameters for GPT-2 Medium — roughly 1000× fewer than full fine-tuning — while targeting comparable NLG quality.

# 8. References
Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021). LoRA: Low-rank adaptation of large language models. arXiv. https://arxiv.org/abs/2106.09685

Wang, A., Singh, A., Michael, J., Hill, F., Levy, O., & Bowman, S. R. (n.d.). GLUE: General Language Understanding Evaluation benchmark. https://gluebenchmark.com/

Novikova, J., Dušek, O., & Rieser, V. (2017). The E2E dataset: New challenges for end-to-end generation. In Proceedings of the 18th Annual Meeting of the Special Interest Group on Discourse and Dialogue (pp. 201–206). Association for Computational Linguistics. https://arxiv.org/abs/1706.09254
# 9. Acknowledgements
This project was completed as part of CS4/5782 at Cornell University. We thank the course staff for their guidance and feedback throughout the implementation process. 
