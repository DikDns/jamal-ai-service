# Research Article Outline: JAMAL AI

**Title**: Enhancing Brainstorming Efficiency with Siamese Neural Networks for Semantic Idea Grouping

**Methodology**: CRISP-DM (Cross-Industry Standard Process for Data Mining) (Chapman et al., 2000; Wirth & Hipp, 2000)

---

## 0. Abstract

- **Context**: The rise of remote collaboration tools (e.g., FigJam, Miro) leads to "sticky note overload" during brainstorming.
- **Problem**: Manually grouping hundreds of ideas by topic is time-consuming and cognitive-intensive.
- **Solution**: A machine learning service ("JAMAL AI") that automatically groups semantically similar ideas.
- **Method**: Usage of Siamese Neural Networks with Contrastive Loss to learn a semantic similarity metric from the STS Benchmark dataset.
- **Result**: A deployed API service capable of clustering brainstorming ideas with high accuracy, reducing manual effort.

---

## 1. Business Understanding

### 1.1 Determine Business Objectives

- **Background**: "Sticky note overload" in remote brainstorming sessions (e.g., FigJam, Miro) often overwhelms designers (Gonçalves et al., 2020).
- **Goal**: Automate affinity diagramming to save facilitator time and cognitive load, enabling automated graphical summarization of creative traces (Gero & Jiang, 2025).
- **Success Criteria**:
  - **Efficiency**: Reduce grouping time from minutes to seconds.
  - **Quality**: High semantic correlation (model output matches human intuition).

### 1.2 Assess Situation

- **Resources**:
  - _Hardware_: Standard GPU for training (Kaggle T4), CPU for inference (FastAPI).
  - _Data_: STS Benchmark (proxy for semantic similarity), Custom Brainstorming Dataset (validation).
- **Risks**: Model might struggle with domain-specific jargon or extremely short text (1-2 words).

### 1.3 Determine Data Mining Goals

- **Objective**: Learn a metric space where semantically similar ideas are close (Euclidean distance < Threshold) and dissimilar ones are far (Elhamifar et al., 2022).
- **Project Plan**: Train Siamese Network -> Evaluate on STS -> Validate on "Jamal Test" -> Deploy as API.

---

## 2. Data Understanding

### 2.1 Collect Initial Data

- **Source**: Hugging Face Datasets (`mteb/stsbenchmark-sts`).
- **Reasoning**: Gold-standard dataset for semantic similarity with 0-5 human-annotated scores (Rezaei et al., 2024).

### 2.2 Describe Data

- **Structure**: Sentence A, Sentence B, Similarity Score.
- **Volume**: ~5,700 Training pairs, ~1,500 Test pairs.

### 2.3 Explore Data

- **Distribution Analysis**: Visualizing score distribution to decide on binary thresholding.
- **Sample Inspection**: Identifying "paraphrases" vs "related topics" vs "distractors".

### 2.4 Verify Data Quality

- **Checks**: No missing values in identifying columns; scores are normalized.

---

## 3. Data Preparation

### 3.1 Select Data

- **Filtering**: Using all valid string pairs.

### 3.2 Clean Data

- **Normalization**: Lowercasing, removing special characters (if applicable).

### 3.3 Construct Data (Feature Engineering)

- **Tokenization**: Converting text to integer sequences (Vocab: 15,000) (Kim et al., 2024).
- **Padding**: Uniform sequence length (50 tokens) for LSTM input.

### 3.4 Format Data

- **Label Transformation**: Converting continuous STS scores (0-5) to binary labels (1/0) for Contrastive Loss stability (Threshold $\ge$ 2.5) (Biancofiore et al., 2023).

---

## 4. Modeling

### 4.1 Select Modeling Technique

- **Algorithm**: Siamese Neural Network (Twin Networks) (Chen et al., 2018; Yang et al., 2018).
- **Rationale**: Proven architecture for "one-shot" or few-shot similarity learning tasks. Deep Learning approach (LSTM) captures sequential context better than simple TF-IDF (Kudo & Yamamoto, 2023).

### 4.2 Generate Test Design

- **Loss Function**: Contrastive Loss (Distance Minimization for Similarity) (Hadsell et al., 2006).
- **Validation**: Split data into Train/Validation/Test sets.

### 4.3 Build Model

- **Architecture**:
  - Embedding (128d) -> BiLSTM (128d) -> Dense (64d) -> L2 Norm (Zhao et al., 2018; Dhanasekaran et al., 2022).
- **Training**: Adam Optimizer, 15 Epochs, Batch Size 64.

### 4.4 Assess Model

- **Technical Metrics**: Loss curves, Accuracy, Precision, Recall, F1-Score.

---

## 5. Evaluation

### 5.1 Evaluate Results against Business Success

- **Metric Verification**: Does the 80%+ F1-score on STS translate to good sticky note grouping?
- **Domain Validation ("The JAMAL Test")**:
  - Testing on real-world IT brainstorming scenarios (e.g., "Login bugs").
  - Result: Model successfully groups paraphrased ideas, meeting the "Quality" objective.

### 5.2 Review Process

- **Analysis**: High correlation with human scores (Pearson/Spearman) confirms the model captures semantic meaning, not just keywords.

### 5.3 Determine Next Steps

- **Decision**: Proceed to deployment given the satisfactory performance on domain-specific test cases.

---

## 6. Deployment

### 6.1 Plan Deployment

- **Architecture**: Microservice pattern.
- **Tech Stack**:
  - **Model format**: TensorFlow SavedModel (portable).
  - **API**: FastAPI (Python) for inference endpoint.
  - **Container**: Docker (reproducible environment).

### 6.2 Plan Monitoring and Maintenance

- **Monitoring**: Track API latency and grouping confidence scores.
- **Maintenance**: Periodic retraining if new domain jargon emerges (e.g., new tech terms).

### 6.3 Produce Final Report

- **Deliverables**: Source code, Dockerfile, and this Research Article.
- **Presentation**: Demo using "Jamal AI" on a sample whiteboard.

---

## 7. Conclusion & Future Work

- **Summary**: Successfully built a semantic grouping engine using Siamese Networks.
- **Future Improvements**:
  - Active Learning pipeline to improve model with user corrections.
  - Transition to Transformer (BERT/MiniLM) for state-of-the-art performance (Zhang et al., 2023; Sato et al., 2024).

---

## 8. References

Bao, W., Bao, W., Du, J., Yang, Y., & Zhao, X. (2018). Attentive Siamese LSTM network for semantic textual similarity measure. In _2018 International Conference on Asian Language Processing (IALP)_ (pp. 312–317). IEEE. https://doi.org/10.1109/IALP.2018.8629212

Biancofiore, V., De La Rosa, K., Hovy, D., & Sarti, G. (2023). A distribution-based threshold for determining sentence similarity. _arXiv_. https://doi.org/10.48550/arXiv.2311.16675

Chapman, P., Clinton, J., Kerber, R., Khabaza, T., Reinartz, T., Shearer, C., & Wirth, R. (2000). _CRISP-DM 1.0: Step-by-step data mining guide_ (Technical report). SPSS Inc.

Chen, Q., Huang, H., Tian, S., & Qu, Y. (2018). A sentence similarity estimation method based on improved Siamese network. _Intelligent Information Management, 10_(5), 115–127. https://doi.org/10.4236/iim.2018.105010

Dhanasekaran, K., Nandhini, K., Subramanian, K., & Sundararajan, V. (2022). A hybrid approach of weighted fine-tuned BERT extraction with deep Siamese Bi–LSTM model for semantic text similarity identification. _Applied Soft Computing, 114_, 108082. https://doi.org/10.1016/j.asoc.2021.108082

Elhamifar, E., Li, X., & Xiong, C. (2022). Active metric learning and classification using similarity queries. In _Proceedings of the AAAI Conference on Artificial Intelligence_ (Vol. 36, No. 7, pp. 7426–7434).

Gero, J. S., & Jiang, H. (2025). Fuzzy linkography: Automatic graphical summarization of creative activity traces. In _Proceedings of the International Conference on Computational Creativity_.

Gonçalves, J., Cardoso, C., & Badke-Schaub, P. (2020). How interaction designers use tools to manage ideas. _International Journal of Design, 14_(3), 17–33.

Hadsell, R., Chopra, S., & LeCun, Y. (2006). Dimensionality reduction by learning an invariant mapping. In _2006 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR'06)_ (Vol. 2, pp. 1735–1742). IEEE. https://doi.org/10.1109/CVPR.2006.100

Kim, G., Park, J., & Lee, J. (2024). An empirical study of tokenization strategies for various Korean NLP tasks. _IEEE Access, 12_, 157889–157902. https://doi.org/10.1109/ACCESS.2024.3485058

Kudo, R., & Yamamoto, K. (2023). Semantic search of Japanese sentences using distributed representations. In _2023 International Conference on Technologies and Applications of Artificial Intelligence (TAAI)_ (pp. 1–6). IEEE. https://doi.org/10.1109/TAAI58568.2023.10409840

Rezaei, S., et al. (2024). Comparative study and evaluation of machine learning models for semantic textual similarity. _IEEE Transactions on Artificial Intelligence, 5_(x), xx–xx.

Sato, S., Yokoi, S., Takase, S., & Okazaki, N. (2024). Are ELECTRA's sentence embeddings beyond repair? The case of semantic textual similarity. In _Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)_ (pp. 479–489). Association for Computational Linguistics. https://arxiv.org/abs/2402.13130

Wirth, R., & Hipp, J. (2000). CRISP-DM: Towards a standard process model for data mining. In _Proceedings of the 4th International Conference on the Practical Applications of Knowledge Discovery and Data Mining_ (pp. 29–39).

Yang, P., Liu, J., & Huang, X. (2018). Siamese networks for semantic pattern similarity. In _Proceedings of the 14th International Conference on Semantic Systems_ (pp. 187–193). https://doi.org/10.1007/978-3-319-98192-5_35

Zhang, L., Wang, Y., & Li, H. (2023). Fine-tuned training method for semantic text similarity measurement using SBERT, Bi-LSTM and attention network (SBiLA). In _2023 12th International Conference on Awareness Science and Technology (iCAST)_ (pp. 1–6). IEEE. https://doi.org/10.1109/iCAST59294.2023.10578412
