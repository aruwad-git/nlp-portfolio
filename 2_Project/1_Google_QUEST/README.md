# 1. Overview.

## Highlights.
- **Open competition** on Kaggle, held by Google.
- **Rank**: 127 / 1,572 (Spearman's Correlation = 0.3813).
- Custom model with bypass, for non-linguistic features.
- Stage-wise Fine-Tuning, Alternating training between Full and Head-Only.
- One head for all labels, to capture the relative info between labels.
- Blog: [(tbd)]()
- Note) Project and results are post-competition.
  
## What is 'Google QUEST Q&A Labeling'?

- **Description:**  
  - This project aims to predict multiple attributes of question-answer pairs, such as clarity, relevance, and helpfulness, by analyzing their textual content. 
  - This task is challenging due to the need for multi-label predictions that capture nuanced semantic and contextual information. 
  - The dataset reflects real-world scenarios with varied and unstructured inputs, making it highly relevant to applications in search engines, chatbots, and Q&A platforms.
  - Successfully solving this problem demonstrates the ability to handle complex NLP challenges.

- **Task Type:**
  - Text Classification.
  - Multi-label Classification.
  - Supervised Learning.

- **Input:**
  - QA datasets corrected by CrowdSource team at Google Research.
  - Gathered from nearly 70 different websites.
  - Main part consists of **Question Title**, **Question Body**, and **Answer Text**.
  - Full list is summarized below.
  - Example:
    - Question Title: "Best way to learn Python for data science?"
    - Question Body: "I want to learn Python for data science. Should I start with courses or books?"
    - Answer Text: "Start with courses on Coursera or Udemy; they're hands-on. Books like 'Python for Data Analysis' are also helpful."

- **Output:**
  - Target values of **30 labels** for each QA pair.
  - Full list is summarized below.
  - Each label has range of [0, 1].
  - Example:
    - "question_asker_intent_understanding": 0.92
    - And other 29 labels.

- **Evaluation:**
  - Mean column-wise Spearman's correlation coefficient.
  - $$\text{Score}(y_{\text{pred}}, y_{\text{test}}) = \frac{1}{m} \sum_{j=1}^{m} \rho_j$$
    - $$y_{\text{pred}}$$: Predicted values for all samples across $$m$$ labels.
    - $$y_{\text{test}}$$: Ground truth values for all samples across $$m$$ labels.
    - $$m$$: Number of labels (columns).
    - $$\rho_j$$: Spearman's rank correlation coefficient for the $$j$$-th label.
  - $$\rho_j = 1 - \frac{6 \sum_{i=1}^n (r_{ij} - \hat{r}_{ij})^2}{n (n^2 - 1)}$$
    - $$n$$: Number of samples.
    - $$r_{ij} = \text{Rank}(y_{\text{test}, ij})$$
    - $$\hat{r}_{ij} = \text{Rank}(y_{\text{pred}, ij})$$

---

## About Project.

| **Name**               | **Description**                                                                                     |
|:------------------------|:----------------------------------------------------------------------------------------------------|
| **Name**               | Google QUEST Q&A Labeling.                                                                          |
| **Period**             | 2025.1.1 ~ 1.2. (2 days).                                                                       |
| **Category**           | `Text Classification`<br>`Multi-label Classification`<br>`Question-Answering`                       |
| **Tags**               | `NLP`, `LLM`, `Text Classification`<br>`QA`, `Kaggle`, `Competition`                        |
| **Blog**               | [Velog]().                                                                                         |
| **GitHub Repository**  | [GitHub]().                                                                                        |



## About Competition.

| **Name**                   | **Description**                                                                                     |
|:----------------------------|:----------------------------------------------------------------------------------------------------|
| **Name**                   | Google QUEST Q&A Labeling.      |
| **Host**                   | Google.                                                                                            |
| **Period**                 | 2019.11.23 ~ 2020.2.11 (80 days).                                                                  |
| **Prizes**                 | $25,000.                                                                                           |
| **Participation**          | `10,582 Entrants`<br>`1,904 Participants`<br>`1,571 Teams`<br>`27,817 Submissions`                  |
| **Notebook Requirements**  | `CPU Notebooks ≤ 9 hours run-time`<br>`GPU Notebooks ≤ 2 hours run-time`<br>`Internet must be turned off` |
| **Citation**               | Danicky, Praveen Paritosh, Walter Reade, Addison Howard, and Mark McDonald. Google QUEST Q&A Labeling.<br>Kaggle, https://kaggle.com/competitions/google-quest-challenge, 2019. |


## Feature List.

| **Category**       | **Features**             |
|:--------------------|:-------------------------|
| **Metadata**        | `qa_id`                 |
|                     | `question_user_name`    |
|                     | `question_user_page`    |
|                     | `answer_user_name`      |
|                     | `answer_user_page`      |
|                     | `url`                   |
|                     | `category`              |
|                     | `host`                  |
| **Question**        | `question_title`        |
|                     | `question_body`         |
| **Answer**          | `answer`                |

## Label List.

| **Category**       | **Output Labels**                                                                                   |
|:--------------------|:----------------------------------------------------------------------------------------------------|
| **Question Labels** | `question_asker_intent_understanding`<br>`question_body_critical`<br>`question_conversational`<br>`question_expect_short_answer`<br>`question_fact_seeking`<br>`question_has_commonly_accepted_answer`<br>`question_interestingness_others`<br>`question_interestingness_self`<br>`question_multi_intent`<br>`question_not_really_a_question`<br>`question_opinion_seeking`<br>`question_type_choice`<br>`question_type_compare`<br>`question_type_consequence`<br>`question_type_definition`<br>`question_type_entity`<br>`question_type_instructions`<br>`question_type_procedure`<br>`question_type_reason_explanation`<br>`question_type_spelling`<br>`question_well_written` |
| **Answer Labels**   | `answer_helpful`<br>`answer_level_of_information`<br>`answer_plausible`<br>`answer_relevance`<br>`answer_satisfaction`<br>`answer_type_instructions`<br>`answer_type_procedure`<br>`answer_type_reason_explanation`<br>`answer_well_written` |

