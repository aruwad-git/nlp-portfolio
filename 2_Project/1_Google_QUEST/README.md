# 1. Overview.

## Highlights.
- **Open competition** on Kaggle, held by Google.
- **Rank**: Spearman's Correlation = 0.3813, 127 / 1,572 (top 8%; post-competition).
- Skip Connection: Designed metadata features (question category, host website) as a skip connection, bypassing the transformer and directly connecting to the head to preserve information and enhance performance.
- Stage-wise Fine-tuning: Repeated combinations of full training and head-only training, configuring different learning rates and schedulers for each stage to enhance robustness against overfitting and improve performance.


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


- Code: [here](https://github.com/aruwad-git/nlp-portfolio/tree/main/2_Project/1_Google_QUEST)
# 1. Introduction.

## Overview.
- 이 글은 Google이 Kaggle에서 주최한 **["Google QUEST Q&A Labeling Competition"](https://www.kaggle.com/competitions/google-quest-challenge)**에 도전한 결과를 정리한다.
- Q&A 입력에 대해 다양한 Label을 예측하는, 전형적인 **Multi-label Text Classification** 문제다.
- 통상적인 Pretrained LLM with Fine-Tuning 기반 다양한 기법 사용, **127/1,572 (top 8%)** 달성! 
  - Custom model with skip connections.
  - Stage-wise Fine-Tuning, Alternating training between Full and Head-Only, etc.
  - One, Complex Head 등.
- 대회 종료 후 수행, 순위는 [Leaderboard](https://www.kaggle.com/competitions/google-quest-challenge/leaderboard?)를 기반으로 계산.
- 프로젝트는 실제 문제를 통해 LLM Fine-Tuning Workflow에 대한 명확한 이해를 보여주기 위해 수행 $\rightarrow$ 실제 대회를 위한 Ensemble, Optuna, QLoRA 등은 생략. 
- **Tech Stacks**: 그냥저냥 통상적인 NLP library들. PyTorch, Transformer, BERT, bitsandbytes, scikit-learn 등.

---

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

> ### 음 그냥 평범한 Text Classification 문제 아님?  

- **아니다.** 이 문제가 어려운 몇 가지 이유가 있다.
- 무려 **30개의 라벨**에 대한, 각 score를 예측해야 한다. 사실상 **Regression** 문제다.
- 뿐만 아니라, 라벨 중 상당수는 **주관적**이며, **주어진 Context로부터 추론하기 어렵다.**
  - 예컨대, `'answer_plausible'` 라벨은 평가자의 주관적인 평가를 의미하며 객관적으로 측정할 수 없다.
  - 또한 모델은 오직 질문-답변 데이터만 보고 추론해야 하므로(즉 질문자의 후기 비슷한 것도 없음), 주어진 Context 밖에서 이를 설명할 수 있는 무언가를 학습해야 한다.
  - 즉, 모델은 사람의 주관적인 감정을 표상하고 추론할 수 있는 **일반화된 규칙**을 학습해야 한다! (예컨대 규칙 : '출처'를 제공하는 것은 사람을 '만족'시킨다, 추론 : 이 답변의 이 부분은 '출처'를 제공하므로, 이 답변은 사람을 '만족'시킬 것이다)
  - NLP 좀 치는 분들은 눈치채셨겠지만, 이는 **Attention 기반 Transformer들이 잘하지 못하는 영역**이다.
- 마지막으로, **질문자**와 **평가자**가 다르다! 그러니까 아무 상관 없는 제 3자가 질문자의 의도, 만족도 등을 추론해야 한다는 것이다.
- 이에 대한 배려라도 하는 건지, Evaluation이 정확한 값이 아닌 **rank에 대한 정확도 (Spearsman)**로 수행되었다.

> ### 그래서 어떻게 할건데?

- 일단 Competition은 내가 한창 하던 예전이나 지금이나 대부분 ~~닥치고~~ **Ensemble**이다. 다들 비슷한 Pretrained LLM을 사용하는 NLP에선 더욱 그렇다. [Leaderboard](https://www.kaggle.com/competitions/google-quest-challenge/leaderboard)에서 실제 우승 팀들의 후기를 보면 거의 다 사용했다.
- 또한 Competition에 진짜 진심이라면 ~~닥치고~~ **Feature Engineering** 노가다를 하는게 국룰이다. Feature scaling 이런거 얘기하는게 아니다. 그 대회의 **Task-specific**한 feature들이다 (예컨대 Stackoverflow의 답변들을 긁어와서 **직접** 느껴보고 Postprocessing을 한다던지 ~~직접 앙상블에 들어가 인간 voter가 되어보자!~~). 나도 몇몇 꽤 좋은 것들을 발견했지만, 끝난 대회에 노가다를 하긴 좀 그래서 관뒀다.
- 전통적인 Hyperparameter Tuning만으론 어려울 것 같고, Context 외의 Feature들을 어떤 구조로 모델링하냐가 핵심일 것 같다.
- 당연한 귀결이지만, Body (Transformer) 안에서 못하면 Head (Dense)에서 하면 된다. 몇몇 [상위권 참가자](https://www.kaggle.com/competitions/google-quest-challenge/discussion/129927)들은 Dense를 몇 개 쌓고 특정 feature만 취하는 등의 Custom Head를 사용했고, 그냥 Default Head로 밀어붙인 사람은 거의 없었다.
- 나는 Non-linguistic feature를 Transformer 밖으로 bypass하는 방식(관심 있으시면 [이 논문](https://arxiv.org/abs/1606.07792?utm_source=chatgpt.com) 참조), Complex Head 및 Iterative training 등을 사용하였다. (자세한 건 아래 ㄱㄱ)
- Head는 라벨 별 구성이 아닌, 통합하여 하나의 Dense로 입력받았다. 해당 문제는 각 라벨의 값이 아닌 라벨 간 순위를 맞추는 것이므로, 라벨 간 상대적인 수치 정보를 잃는 것은 너무 큰 손해라고 생각했다.
- 30개 라벨의 순서라는, 굉장히 Task-specific한 타겟을 학습해야 하므로, Adapter 방식의 PEFT (e.g. LoRA)가 적합할 것 같다. 이번 프로젝트의 범위를 벗어나 생략했다. <small>~~아무튼 적합할거다~~</small>
- 그 밖의 Tokenization이나 Pretraining Model 선택(당연히 BERT-like encoder-only 모델들) 등은 다들 거기서 거기인 듯 하다. Data augmentation 등을 시도한 글들은 많았으나 대부분 그닥인 것 같다.

---

# 2. Preprocessing.

## 2.1. Data Load.

```python
# Load.
from sklearn.model_selection import train_test_split
data                = pd.read_csv('./dataset/train.csv')
train_set, test_set = train_test_split(data, test_size=0.2)

label_col_idx = train_set.columns.get_loc('question_asker_intent_understanding')  # Label Columns start from it.
x_train       = train_set.iloc[:, :label_col_idx]
y_train       = train_set.iloc[:, label_col_idx:]
x_test        = test_set.iloc[:, :label_col_idx]
y_test        = test_set.iloc[:, label_col_idx:]

# Null Check.
is_null = data.isnull().values.any()
print(f'Any null values? {is_null}')

# Copy train_set for EDA. (Only if train_set < 2 GB)
train_set_size = train_set.memory_usage(deep=True).sum() / (1024 ** 3)  # In GB.

if train_set_size < 2:               
    train_cp   = train_set.copy()
    x_train_cp = x_train.copy() 
    y_train_cp = y_train.copy()
else:
    print("Train set copy failed! It's more than 2 GB!")
```

- Competition 글인 만큼 다들 아는 내용들은 넘어가고 몇 가지만 간단히 짚고 가자.
- 받자마자 **Null Check**를 하는 습관을 들이자.
- 왠만하면 Original dataset은 **copy**해서 쓰자. <small>~~나중에 재수없어서 캐시 날려먹고 샤드 찾고 하다보면 아~~</small>
- 본 프로젝트에선 `pd.DataFrame`로 바로 받았지만, 🤗를 쓰는 경우는 왠만하면 `datasets.load_dataset()`으로 받자. 
Arrow와 Memory mapping 등을 사용하여 가뜩이나 부족한 RAM도 아끼고, type conversion에 interface만 바꾸는 등, 암튼 효율적이다. [참고](https://huggingface.co/docs/datasets/about_arrow?utm_source=chatgpt.com)

> #### Caution) Null Check for Future Input.
현업에서는 Train set에 **Null이 없어도 반드시!** handling 해줘야한다. Test set이나 Real data에서 뭐가 들어올지 모르기 때문이다. 
사실 Adversary input 등 훨씬 자세하게 처리해줘야 하나, 어지간한 출시 직전 단계가 아니고서야 ~~고귀하신 DS들은~~ 귀찮아서 잘 안하는게 국룰이지만 ㅎㅎ 그래도 Null handling만큼은 해주자! 

## 2.2. Feature Engineering.

### Table: Feature List.

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


- `qa_id` : Sample의 index이다. Submit 할 때 빼곤 1도 쓸모없다.
- `question_title`, `question_body`, and `answer` : 일반적인 text이다. 특이한 점은, 일종의 category 혹은 summary의 역할을 하는 **`question_title`**을 제공한다는 점이다. 이는 당연히 별도의 Transformer로 학습하고 merge를 하던 ensemble을 하던 해야겠지만, 시간이 없어서 못했다 (진짜로 ㅠ). 실제 참여했다면 당연히 했을 것이다.

### Asker-specific Pattern?

- `question_user_name`, `question_user_page` : 질문자 이름 및 프로필 페이지다. 
  - 프로필은 대충 [이런](https://serverfault.com/users/111705/beingalex) 형태이다.
  - URL 타고 가서 질문자에 대한 메타데이터라도 긁어오라고 준걸까? ~~(당연히 아무도 안함 ㅋㅋ)~~
  - 다들 자연스럽게 '질문자 별로 특징이 있다면?'이라는 생각이 들었을 것이다. 심지어 의외로 동일 질문자가 꽤나 반복해서 나온다?
  
  ```python
  uname = 'question_user_name'
x_train_cp[uname].value_counts().hist(bins=8)
  ```
![업로드중..](blob:https://velog.io/5884cdf4-a1ad-4341-b5d9-20b79646f748)

  - 나는 앞서 언급했듯이 이 문제는 context 밖의 feature가 굉장히 소중하다고 판단했다. 따라서 많이 고민했지만, 과감히 버리기로 했다.
    - 1) Test set에도 비슷한 패턴이 나온다는 보장이 없다. 엄청난 overfitting을 야기하거나, 아무 쓸모도 없을 가능성이 크다. 
    - 2) 결정적으로, 질문자와 평가자가 다르다. 설마 평가자가 질문자 이름을 보고 고유한 평가 패턴을 보였을까?
    - 3) 지극히 현실적인 이유지만, 이미 종료된 대회에 이런 Feature 노가다를 하긴 좀 그렇다 ㅠ.


### Delete Redundant Features.

```python
# Delete redundant features.
redundant_features = ['qa_id', 'question_user_name', 'question_user_page',
                      'answer_user_name', 'answer_user_page', 'url']
x_train_cp = x_train_cp.drop(columns=redundant_features) 
```

- `question_user_page`, `answer_user_name` 등 다른 feature들 역시 같은 이유로 제거하였다.


### Categorical Encoding.

```python
# Encode `category` and `host`.
# Find Top N hosts.
n_hosts   = 10
top_hosts = x_train_cp['host'].value_counts().nlargest(n_hosts).index

# Convert others into 'Others'.
x_train_cp['host'] = x_train_cp['host'].apply(lambda x: x if x in top_hosts else 'Others')

# One-hot Encoding.
from sklearn.preprocessing import OneHotEncoder

cols_to_enc = ['category', 'host']
one_enc     = OneHotEncoder(handle_unknown='ignore')          # Zero vector for unknown category.
x_enc       = one_enc.fit_transform(x_train_cp[cols_to_enc])

```

```
Length: 5
category
TECHNOLOGY       1960
STACKOVERFLOW    1022
CULTURE           769
SCIENCE           560
LIFE_ARTS         552
Name: count, dtype: int64

Length: 63
host
stackoverflow.com                1022
english.stackexchange.com         185
superuser.com                     176
electronics.stackexchange.com     173
serverfault.com                   165
Name: count, dtype: int64
```
- `category` : 적당한 정도의 class를 갖는 평범한 카테고리인 것 같다. Feature dimension도 별로 안 크니 적당히 One-hot encoding 해주자.
- `host` : 호스트 사이트는 꽤 중요한 feature 것 같으나, 63개는 너무 많다. 10개만 골라서 마찬가지로 One-hot encoding 해주자.


### Merge Text Columns.

```python
# Merge sentences into one column.
cols_txt = ['question_title', 'question_body', 'answer']
x_train_cp['txt_merged'] = x_train_cp[cols_txt].apply(lambda row: ' '.join(row), axis=1)
x_train_cp = x_train_cp.drop(columns=cols_txt)
```

- 3개 text column을 그냥 합쳤다. 앞서 언급했지만 적어도 `question_title`은, 가능하면 3개 모두 별도로 구성하는 것이 좋을 것 같다. 
(가물가물 하지만 1등이 각자 훈련해서 stacking 한 걸로 기억함)

### Preprocessing Pipeline.

```python
def preprocess(x_train, n_hosts=10):
    # 2.2.1. Drop redundant features.
    redundant_features   = ['qa_id', 'question_user_name', 'question_user_page', 
                            'answer_user_page', 'answer_user_name', 'url']
    x_train = x_train.drop(columns=redundant_features)

    # 2.2.2. Encode categorical features.
    # Converts other categories into 'Others'.
    top_hosts = x_train['host'].value_counts().nlargest(n_hosts).index
    x_train['host'] = x_train['host'].apply(lambda x: x if x in top_hosts else 'Others')

    # Encode `category` and `host`.
    categorical_features = ['category', 'host']
    one_enc     = OneHotEncoder(handle_unknown='ignore')    # Zero vector for unknown category.
    x_enc       = one_enc.fit_transform(x_train[categorical_features])

    # Convert back to DataFrame.
    enc_columns = one_enc.get_feature_names_out(categorical_features)
    x_enc_df    = pd.DataFrame(x_enc.toarray(), columns=enc_columns, index=x_train.index)
    x_train     = x_train.drop(columns=categorical_features)   # Drop original 'category' and 'host' columns.
    x_train     = pd.concat([x_train, x_enc_df], axis=1)       # Concatenate the encoded columns back.

    # 2.2.3. Merge txt columns.
    cols_txt = ['question_title', 'question_body', 'answer']
    x_train['txt_merged'] = x_train[cols_txt].apply(lambda row: ' '.join(row), axis=1)
    x_train  = x_train.drop(columns=cols_txt)    # Drop original txt cols.
    
    # Return.
    return x_train
```

```
category_CULTURE                                                                    0.0
category_LIFE_ARTS                                                                  0.0
category_SCIENCE                                                                    0.0
category_STACKOVERFLOW                                                              0.0
category_TECHNOLOGY                                                                 1.0
host_Others                                                                         1.0
host_askubuntu.com                                                                  0.0
host_electronics.stackexchange.com                                                  0.0
host_english.stackexchange.com                                                      0.0
host_math.stackexchange.com                                                         0.0
host_physics.stackexchange.com                                                      0.0
host_rpg.stackexchange.com                                                          0.0
host_serverfault.com                                                                0.0
host_stackoverflow.com                                                              0.0
host_superuser.com                                                                  0.0
host_tex.stackexchange.com                                                          0.0
txt_merged                            Ensuring successful merges in Subversion Subve...
```

---

# 3. Tokenization.

## 3.1. Pretrained LLM.

```python
checkpoints = {'distilbert' : 'distilbert-base-uncased',
               'bert' : 'bert-base-uncased',
               'roberta' : 'roberta-base'}
```
- Pretrained LLM을 쓸 때는 LLM과 동일한 Tokenizer를 사용해야 하므로, 지금 후보를 정해준다. ~~(설마 모르시는 분 없죠?)~~

## 3.2. How Many Tokens to Preserve?

- RNN and variations와 다르게, transformer는 고정된 길이의 입력을 받는다. 대부분의 Typical LLM은 512다. 즉 최대 512개까지 토큰을 받을 수 있다.
- 이를 넘는 Sample은 자르던가 (**truncation**) 나눠야한다 (**stride**).
- `tokenizer`의 **`max_length`**는 매우 중요한 hyperparameter다. 줄이면 Input dimension이 크게 줄어들지만, 당연히 정보 손실 역시 커진다.

```python
x_train_tokenized['input_ids']    = list(tokenized['input_ids'])
x_train_tokenized['token_length'] = x_train_tokenized['input_ids'].apply(len)
x_train_tokenized['token_length'].hist(bins=200).set_xlim(0, 2000)
```
![업로드중..](blob:https://velog.io/fe485e31-96b7-47f4-8d2a-a184231ca350)

```python
# How many tokens are covered, i.e. not truncated, with the given length?
n_train       = len(x_train_tokenized['token_length'])
token_lengths = [128, 256, 512]

for token_length in token_lengths:
    rank         = (x_train_tokenized['token_length'] <= token_length).sum()
    quantile     = (rank / n_train) * 100
    print(f"max_length={token_length} covers {quantile:.2f}% of samples!")
```

```
max_length=128 covers 7.38% of samples!
max_length=256 covers 35.10% of samples!
max_length=512 covers 73.06% of samples!
```

- 아쉽게도 많은 Sample들이 512를 가뿐히 넘고 있다. 불과 73%밖에 커버되지 않는다 (즉 Sample 중 27%는 끝 혹은 상당 부분이 잘린다).
- ~~리뷰 좀 작작 쓰라고! 라고 생각할 수도 있지만~~ 이는 아주 자연스러운 결과다. 3개 text를 합쳤기 때문이다. 이미 몇 번 언급했지만, 시간과 자원이 허락한다면 다양한 방식으로 나눠서 Ensemble을 해보자!

## 3.3. Tokenize.

```python
def tokenize(df):  
    # Define Tokenizer.                   
    tokenized = tokenizer(
        list(df['txt_merged']),
        padding          = True,
        truncation       = True,
        max_length       = max_length,
#       stride           = 0,           # Can be kept if you want overlapping tokens.
        return_tensors   = "np"  
    )

    # And tokenize.
    df['input_ids']      = list(tokenized['input_ids'])
    df['attention_mask'] = list(tokenized['attention_mask'])
    
    df = df.drop(columns=['txt_merged'])      # Drop original text column.
    
    return df

```

- `stride` : 일반적으로 Truncation이 많은 경우 `stride`는 좋은 선택이다. 하지만 필자는 여러가지 시도해보고 뺐다. Title, question, answer이 잡탕으로 섞여있고 각자의 길이가 굉장히 다양해서, Fixed length로 반복하는 `stride`는 결국 마이너스가 더 컸다. ~~(나눕시다 ㅎㅎ)~~

---

# 4. Model.

# 4.1. Custom Model.

```python
class CustomModel(nn.Module):
    def __init__(self, checkpoint, num_labels, additional_feature_dim):
        super(CustomModel, self).__init__()
        
        # Load pretrained transformer.
        self.transformer = AutoModel.from_pretrained(checkpoint)

        # Expose the transformer's config.
        self.config = self.transformer.config
        
        # Combine transformer outputs with additional features
        transformer_hidden_size = self.transformer.config.hidden_size
        self.fc1 = nn.Linear(transformer_hidden_size + additional_feature_dim, num_labels)
        
#       self.fc2 = nn.Linear(256, num_labels)   # For complex Head.
#       self.dropout = nn.Dropout(0.1)          # For dropout.
        
    def forward(self, input_ids, attention_mask, additional_features):
        # Transformer output.
        transformer_output = self.transformer(
            input_ids      = input_ids,
            attention_mask = attention_mask
        )
        
        # Use [CLS] token for concatenation.
        cls_output     = transformer_output.last_hidden_state[:, 0, :]
        combined_input = torch.cat([cls_output, additional_features], dim=1)
        
        # Pass through fully connected layers
#       x = self.dropout(torch.relu(self.fc1(combined_input)))
        output = self.fc1(combined_input)
        
        return output
```

- 초반부에 언급했지만, 나는 Direct context from dialogue 외의 Feature는 Transformer가 잘 detect 하지 못할 거라고 생각한다. ~~(도대체 어떤 corpus를 학습해야 '제 3자는 이 답변에 기뻐할거야! 좋아할거야!' 이런걸 학습한단 말인가?)~~
- 따라서 1) 일단 QA features를 Transformer에 Feedforward 시키고, 2) 나온 Hidden state를 `host` 및 `category`와 Concatenate 하여 Head에 Feedforward 시키며, 3) 조금 더 Complex architecture for Head를 시도해보았다.
- 지금은 없지만, 사실 `self.fc1`과 `self.fc2` 사이에는 굉장히 많은 시도들이 있었고, 몇몇 RNN 기반 시도들은 상당히 성공적이었다. 하지만 불쌍한 나의 3070이 터지기 직전이어서, 어쩔 수 없이 다 지워버렸다.
- 사실 요즘 트렌드는 막대한 자본력을 바탕으로 만들어진 LLM을 고이 모시는 쪽이다 (Optimized full-training). 하지만 이 대회처럼 적당한 규모로, 일반인 위주의, Task는 단순하지만 굉장히 Domain-specific한 경우, 적당한 규모의 LLM + Complex Head가 시간/자원이 제한된 대회에서 먹히는 경우가 꽤 있다.

## 4.2. HF Wrapper.

```python
# Define HF Wrapper for Custom Model.
class HuggingFaceModelWrapper(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model         # Custom model.
        self.config     = base_model.config  # Expose the base model's config.

    def forward(self, input_ids, attention_mask, additional_features, labels=None):
        # Forward pass through the base model.
        output = self.base_model(input_ids           = input_ids, 
                                 attention_mask      = attention_mask, 
                                 additional_features = additional_features)
        
        # If labels are provided, calculate loss.
        logits = output
        loss   = None
        if labels is not None:
            loss_fn = nn.BCEWithLogitsLoss()
            loss    = loss_fn(logits, labels)
        
        return {"loss": loss, "logits": logits}

    def prepare_inputs_for_generation(self, *args, **kwargs):
        # Delegate to the base model.
        return self.base_model.prepare_inputs_for_generation(*args, **kwargs)
```
- Custom PyTorch 모델과 🤗 library들(Trainer 등)을 연동하기 위한 wrapper다.
- 급하게 작성하느라 몇가지 실수가 보인다(`nn.Module`을 상속했다던지). 역시나 이후 PEFT 등 몇가지 문제가 있었지만 그럭저럭 다른 문제는 발견되지 않았다.
<small>~~(응 어차피 안쓸라했음)~~</small>

---

# 5. Fine-Tuning.

## 5.1. Metric.

```python
from scipy.stats import spearmanr

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions    = np.argmax(logits, axis=1) if logits.ndim == 3 else logits

    # Calculate Spearman's correlation for each label.
    spearman_corrs = []
    for i in range(labels.shape[1]):
        corr, _ = spearmanr(predictions[:, i], labels[:, i])
        spearman_corrs.append(corr)

    # Return the mean of Spearman's correlation.
    mean_spearman = np.nanmean(spearman_corrs)  # Handle NaNs if any.
    return {"spearman": mean_spearman}
```

- 본 대회에서는 **Spearman's Correlation**을 사용한다. scipy로 구현해주자. ~~(이야 개오랜만 ㅠㅠ)~~
- 따라서 한 라벨의 **정확한 값**을 맞추는 것이 아닌(그래봤자 확률이지만), **Rank**를 맞추는 것이 중요하다.
- 따라서 나는 Head를 라벨 별로 나누지 않고 하나에 다 받는 것으로 구성했다.
- 이는 **Tradeoff**가 있다. Task 별로 구성하면 개별 문제 자체는 훨씬 쉬워지기 때문이다. 특히 이 대회처럼 데이터가 적은 경우, 이 방식이 더 좋을 수 있다.
- 고민을 많이 했었는데, [2등](https://www.kaggle.com/competitions/google-quest-challenge/discussion/129978)이 나눈 걸 보면 그다지 좋은 선택은 아니었을지도..?

## 5.2. Training Hyperparameter.

```python
# Initialize the model.
checkpoint = checkpoints['bert']     # Other candidates : 'distilbert', 'roberta'.

num_labels              = 30
additional_features_dim = len(x_train_tokenized.columns) - 2    # Except 'input_ids' and 'attention_mask'.

model = HuggingFaceModelWrapper(
    base_model=CustomModel(checkpoint, num_labels, additional_features_dim))

# Define Callbacks.
from transformers import EarlyStoppingCallback
early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience  = 3,     # Stop after this consecutive non-improving eval steps
    early_stopping_threshold = 1e-5   # Minimum improvement threshold
)

# Define TrainingArguments.
from transformers import AutoModel, TrainingArguments, Trainer

# Hyperparameters.
batch_size    = 8
gra_steps     = 1
eval_steps    = 100
warmup_steps  = 0
logging_steps = 100

training_args = TrainingArguments(
    output_dir="./results",                   # Directory for saving model checkpoints
    overwrite_output_dir=True,                # Training from scratch, not from last training
    optim="adamw_bnb_8bit",                   # 8-bits Quantization of Optimizer.
    eval_strategy="steps",                    # Evaluate every few steps
    eval_steps=eval_steps,                    # Evaluation interval
    logging_dir="./logs",                     # Directory for TensorBoard logs
    logging_steps=logging_steps,              # Logging interval
    per_device_train_batch_size=batch_size,   # Batch size for training
    per_device_eval_batch_size=batch_size,    # Batch size for evaluation
    gradient_accumulation_steps=gra_steps,    # Steps for gradient accumulation
    lr_scheduler_type="linear",               # Learning scheduling: "linear"
    warmup_steps=warmup_steps,                # Warmup steps: 150. 
    weight_decay=2e-2,                        # Weight decay
    save_strategy="steps",                    # Save model checkpoints periodically
    save_steps=500,                           # Save every 500 steps
    save_total_limit=3,                       # Keep the last 3 checkpoints
    fp16=True,                                # Enable mixed precision (if supported)
    load_best_model_at_end=True,              # Load the best model after training
    metric_for_best_model="eval_spearman",    # Use Spearman for metric for this competition.
    greater_is_better=True,                   # Greater is better for Spearman.
    seed=42
)

```
- 통상적인 Train loop with 🤗. 중급자 이상 독자를 생각해서 딱히 할 말이.. 그래도 굳이 써보면 :
  - `EarlyStoppingCallback` : 이거 안쓰시는 분 없죠..? 일단 Large epoch 하고 Early stopping parameters 조정하는게 국룰.
  c.f. 초반에 너무 낮게 잡지 말자. ~~(어차피 하루종일 learning curve 보면서 튜닝할거잖슴 ㅋㅋ)~~
  - `batch_size=8` and `gra_steps=1` : 3개 txt를 하나로 합치는 바람에(~~또 너야~~) + Collate function도 제대로 안 정하고 하는 바람에, `batch_size`를 늘리면 훈련이 엄청 불안정해지는 기적이...
  - `optim='adamw_bnb_8bit'` : 확실히 `bnb` 안쓰고 자체 지원해주니까 너무 편했다. 4-bits는 언제 지원해주시나요...
  - `weight_decay=2e-2` : Default로 `1e-2`로 두고 overfitting시 조절하는게 국룰.
  - `metric_for_best_model="eval_spearman"` : Custom metric 지정해주면 `early_stopping_threshold`, `greater_is_better` 등도 같이 신경써주자! ~~다들 한번쯤 이거 깜빡하고 며칠치 training 날려먹어봤잖슴 ㅎㅎ~~
- `warmup_steps=0`.
  - 이 놈은 좀 언급을 하고 싶다.
  - 통상 `n_steps`의 10% 전후로 주어 초기 불안정성을 낮춘다.
  - 그런데 어찌된 일인지 작은 warmup만으로도 높은 확률로 saddle point로 직행하는 것 같다. 아무리 Tensorboard를 뒤져봐도 원인을 못찾았다.
  - 이제와서 드는 생각인데 설마 Non-IID data는 아니겠지..? <small>~~일단 shuffle 안하긴 했음 ㅎㅎ~~</small>
  - 일단 지금은 패스 ㅠ.
## 5.3. Training.

```python
%%time

# Define Trainer and train.

# Full-Training.
training_args.num_train_epochs = 10
training_args.learning_rate    = 4e-5
trainer_full = Trainer(
    model            = model,                          
    args             = training_args,              # TrainingArguments.
    train_dataset    = train_dataset,              # Training dataset.
    eval_dataset     = test_dataset,               # Validation dataset.
    processing_class = tokenizer,                  # Tokenizer.
    compute_metrics  = compute_metrics,            # Spearsman.
    callbacks        = [early_stopping_callback]   # EarlyStopping callback.
)
trainer_full.train()

# Freeze Body.
for param in model.base_model.transformer.parameters():
    param.requires_grad = False

# Head-only Training.
training_args.num_train_epochs = 20
training_args.learning_rate    = 3e-5
trainer_head = Trainer(
    model=model,                          
    args=training_args,                   # TrainingArguments
    train_dataset=train_dataset,          # Training dataset
    eval_dataset=test_dataset,            # Validation dataset
    processing_class=tokenizer,           # Tokenizer
    compute_metrics=compute_metrics,      # Evaluation metric function
    callbacks=[early_stopping_callback]   # EarlyStopping Callback.
)
trainer_head.train()

# Notification for finish.
# import winsound
# winsound.PlaySound("Alarm03", winsound.SND_ALIAS)
```

- **Alternating Training**.
  - Head에 많은 관심을 기울인만큼, Full-training과 Head-only training을 나누어 훈련했다.
  - 나는 Domain-specific 특성, 그리고 엄청 heavy한 LLM이 아니라 Full을 좀 많이 주고(3~5), Head-only와 iteration하며 다양한 방식으로 시도한 결과, 통상적인 방법 (Full의 끝부분만 조금씩 녹이며 Fine-tuning)보다 꽤 성능이 향상되었다. 
  - 당시 실제 대회도 아니고 블로그로 정리할 줄도 몰랐어서, 정확한 데이터를 복원하지 못한 점 심심한 사과의 말씀을...
  - 코드엔 `training_args.num_train_epochs = 10`이지만 대부분 Early Stopping이나 수동으로 끊었다. ~~(그렇다 종일 쳐다보고 있었다)~~
- **Stage-Wise Learning Rate**.
  - Learning rate도 다르게 주었다.
  - 일반적인 예상과는 달리, Head-only에서 작은 Learning rate를 주는 것이 성능이 더 좋았다. Full-training에서 이미 같이 학습했을테니, 일반적인 scheduling에 부합한다. 이상한 현상은 아니다.
  
## 5.4. Final Parameters.

### BERT.

| Parameter                   | Value                   |
|-----------------------------|-------------------------|
| Model                       | bert-base-uncased       |
| Max Length                  | 512                     |
| Learning Rate (Full)        | 4e-5                    |
| Learning Rate (Head)        | 3e-5                    |
| Weight Decay                | 2e-2                    |
| Warmup Steps                | 0                       |
| Batch Size                  | 8                       |
| Gradient Accumulation Steps | 1                       |
| Evaluation Steps            | 100                     |
| Early Stopping Patience     | 3                       |
| Early Stopping Threshold    | 1e-5                    |
| Full Fine-Tuning Epochs     | 10      |
| Head Fine-Tuning Epochs     | 20                      |
| Spearman Correlation        | 0.3813 (127 / 1572)     |
| CPU Time                    | 10min 32s              |
| Wall Time                   | 9min 26s               |
| Train Runtime               | 125.9416 seconds        |
| Train Samples Per Second    | 772.262                |
| Train Steps Per Second      | 96.553                 |
| Training Loss               | 0.324978                |
| Global Steps                | 1200                   |

- 분명 0.39를 넘었던 때도 있었는데, 아무리 checkpoint를 뒤져봐도 못찾았다 ㅠ.
- 사실 이건 한 Full-Head cycle에 대한 기록이고, Alternating Training에 대한 기록을 찾지 못했다. 여러모로 아쉬웠다.

---

# 6. Conclusion.

- 오랜만에 다시 대회에 도전해보았고, 127/1,572 (top 8%)라는 괜찮은 결과를 달성했다!
- BERT에 Custom Head, Stage-wise Fine-Tuning 등 여러가지 기법 및 튜닝을 시도했다.
- 갑작스럽게 준비하여 Optuna, PEFT, bitsandbytes, ETensorboard 등 실제 Fine-Tuning에 사용되는 Tech stacks를 소개하지 못해 아쉬웠다.
- 뭐 class-weighted training 등 여러가지를 언급해서 유식한 척할순 있겠지만, 그냥 ensemble이 갑인 전형적인 케이스였다. 
특히 label 별로 weak classifier를 구성해 adaboost를 돌리면 잘 먹힐 것 같았는데, 제대로 해보지 못해 역시 아쉬웠다.
- 블로그 작성만 6시간 38분이 걸렸다. 막상 쓴 거 보면 허술한데 이거 생각보다 시간이 엄청 걸린다. 욕심을 좀 내려놔야겠다.



