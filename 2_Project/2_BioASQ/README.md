
- Code: [here](https://github.com/aruwad-git/nlp-portfolio/tree/main/2_Project/2_BioASQ)
# 1. Introduction.

## 1.1. About.
- 이 글은 **Medical NLP**에서 **Top-tier Competition**으로 인정받는 **BioASQ Competition**에 도전한 과정 및 결과를 정리한다.
- 기간 : 1/2 ~ 1/10, 2025.
- **Keywords**: RAG, medical NLP, LLM fine-tuning, data augmentation, asynchronized HTTP request.
- **Tech Stacks**:
  - **For RAG**: 🤗, Bitsandbytes, PEFT, Optuna, LangChain, FAISS 등.
  - **For HTTP Request**: aiohttp, asyncio, ThreadPoolExecutor, BeautifulSoup.
  - Others: sklearn, numpy, matplotlib, TensorBoard 등. 
- **Highlights**:
  - Highly domain-specific한 medical NLP task에서 **높은 성능의 RAG 구축**.
    - Macro F1 of **0.9541** (top 1~20%). 
  *\* Performance is for post-competition and 'yes/no' subtask.*
    - RAG Pipeline: 🤗, `langchain`, `langsmith`.
  - **대용량 HTTP request 처리**로 효율적인 Vector DB 구축.
    - Multithread: `ThreadPoolExecutor`.
    - Asyncronized request: `asyncio`.
    - 13,025 URL requests를 single-threaded 대비 약 x17배 빠르게 처리 **(26 hrs $\rightarrow$ 90 mins)**.
  - **Fine-Tuning of Domain-specific LLM** for medical applications.
    - BERT-like LLM for medical QA: BioBERT, PubMedBERT, BioELECTRA 등.
    - 4-bits Quantization: `bitsandbytes`.
    - Adapter-based PEFT, LoRA: `peft`.
    - 효율적인 hyperparameter tuning: `optuna`.
  - **Data Augmentation** 및 **Class-weighted Training**으로 매우 불균형한 class distribution에 효과적으로 대응.
    - 주어진 URLs로부터 HTML retrieval: `bs4.BeautifulSoup`.
    - Similarity search를 통한 document retrival: `faiss`.
    - Training cost 및 performance를 고려한 최적의 sampling 방법 탐색.
    - Class distribution을 74%에서 58%까지 낮추고, Custom trainer for class-weighted Training으로 F1 score 크게 개선.

## 1.2. BioASQ Competition.

[![BioASQ Competition](https://velog.velcdn.com/images/aruwad/post/86016309-3b90-4e8f-92b4-d775f6cda116/image.JPG)](https://www.bioasq.org/)

- [**BioASQ Competition**](https://www.bioasq.org/)은 Medical NLP 분야에서 가장 권위있는 국제 대회 중 하나다.
- CMU, Stanford 등의 Top-tier participants, 2,000편 이상의 publications 등 academic society 및 industry에서 활발히 진행되고 있다.
- **기간** : 2013년 이래로 매년 개최되고 있으며, 올해로 13년차다.
- **주최 기관** : National Centre for Scientific Research (NCSR) in Greece, University of Huston, USA 등.
- **Tasks**: 매년 다양한 종류의 subtasks를 제시하며, Exact answer만 포함하는 기존 datasets과 달리 거의 모든 NLP task를 포함한다.
  - **Task a**: 수천만 이상의 PubMed 논문의 검색 및 분류를 위한 MeSH 추출.
  - **Task b**: LLM 기반 다양한 NLP tasks를 수행: Biomedical QA, Information retrieval, Summarization 등.
  - **Task Synergy**: Academic institutions, Industry 등이 연계하여 PubMed 기반 QA 시스템 구축. 
  - **Task MultiClinSum**: Lengthy clinical case reports를 다국어로 summarize.
  - **Task BioNNE-L**, **Task ELCardioCC**, **Task GutBrainIE** 등.
  
## 1.3. Task Description.
- BioASQ-12b의 subtask 중 하나인 'Exact answer classification'을 대상으로 하였다.
- Question 및 관련 정보들에 기반해 'yes' or 'no'를 답변하는 binary text classification 문제다.
- 관련 정보는 문장 단위의 짤막한 정보인 Snippets 및 관련 PubMed 논문의 URL들인 Documents로 구성된다.
- Example I/O:
  - Question: "Is TIM-3 a target for cancer immunotherapy in NSCLC?"
  - Snippets: 
    - "Our results imply that implementing combined treatment on CIK cells before transfusion via antibodies targeting PD-L1, LAG-3, TIM-3, and CEACAM-1 might improve the efficiency of CIK therapy for NSCLC patients."
    - "Furthermore, TIM-3 and CEACAM1 were strongly expressed simultaneously during long-term CIK culture and showed a significant and mutually positive correlation."
  - Documents:
    - http://www.ncbi.nlm.nih.gov/pubmed/27699239
    - http://www.ncbi.nlm.nih.gov/pubmed/29440769
  - Answer: 1 (for 'yes').
- Metrics: accuracy, f1-yes, f1-no, macro f1.
- Evaluation: 
  - 실제 ranking은 여러 phase (e.g. in 2024, phase A, phase A+, and phase B) 및 test splits (1~4)에서, 여러 subtasks들이 동시에 고려된다.
  - 본 프로젝트는 2024년의 [Leaderboard](http://participants-area.bioasq.org/results/12b/phaseB/)을 참고하여 대략적으로 평가하였다.
  
---
  
# 2. Data Preparation.
## 2.1. Data Retrieval.

```python
import json
from datasets import Dataset
import pandas as pd

def load_datasets_all():
...

# Load.
df_list = []
for path in path_list:
    with open(path, 'r', encoding='utf-8') as f:
        # Load json file.
        data = json.load(f)   

        # Read rows.
        rows = []
        for question in data['questions']:
            if question['type'] == 'yesno':   # Load only samples with type = 'yesno'.
                row = {
                    "question": question['body'],
                    "snippets": "\n".join([s['text'] for s in question['snippets']]),
                    "documents": "\n".join(question['documents']),
                    "answer_exact": question.get('exact_answer', ''),
                    "answer_ideal": question.get('ideal_answer', '')
                }
                rows.append(row)

        # Construct df.
        df = pd.DataFrame(rows)
        df['answer_ideal'] = df['answer_ideal'].apply(lambda x: x[0])
        df_list.append(df)
  
 ...

train_df.head()
```
![](https://velog.velcdn.com/images/aruwad/post/ff27c39f-8364-4247-823b-df9056e41cd0/image.JPG)


- Dataset는 [여기서](http://participants-area.bioasq.org/datasets/) 간단한 절차 후 쉽게 다운받을 수 있다.
- `json` 을 사용하여 간단히 parsing하고 `pd.DataFrame`을 구축하였다.
- 각 features 및 labels은 아래와 같다:

| Column         | Data Type | Description                                                       |
|----------------|-----------|-------------------------------------------------------------------|
| `question`     | Text      | One sentence for the question.                                   |
| `snippets`     | Text      | One to many sentences providing short information.               |
| `documents`    | URL       | One to many URLs linking to research papers from PubMed.         |
| `labels`       | Integer   | 0 for 'no', 1 for 'yes'.                                         |
| `answer_ideal` | Text      | Label for the ideal answer task.                                 |




## 2.2. Data Size.

![](https://velog.velcdn.com/images/aruwad/post/339ad31e-932f-462c-9eba-f6d11c080e0e/image.JPG)

```
n_samples   = 1,356
n_snippets  = 16,567
n_docs      = 13,025
```

- Sample은 총 1,356개로, large LLM을 fine-tuning 하기엔 터무니없이 부족하다.
- Mid~large 규모의 pretrained LLM 기반 RAG를 기본 방향으로 잡야야겠다.
- Full-training은 당연히 바람직하지 않다. Head-only, adapter-based PEFT, zero-shot learning 등을 고려해야 한다.
- 관련 정보들(snippets와 documents)이 꽤 많으므로, data augmentation을 고려해보자.

## 2.3. Length of Features.

```python
# Columns of features.
cols     = ['question', 'snippets', 'documents']
len_cols = ['len_question', 'len_snippets', 'len_documents']

# Calculate each length.
train_df[len_cols] = train_df[cols].map(len)

# Display statistics of each length.
train_df[len_cols].describe()
train_df[len_cols].hist()
```
![](https://velog.velcdn.com/images/aruwad/post/c2c5ba97-6ed8-4512-aa62-4e8a6edc3ac0/image.JPG)

![](https://velog.velcdn.com/images/aruwad/post/2a213634-e608-400a-bdea-ade3bd3d7c41/image.JPG)

- Question의 길이는 대부분 reasonable하다. 절대 truncated 되면 안되는 부분이지만, 통상의 `max_length=512`에선 문제 없을 것 같다.
- Snippets는 상당히 불안하다. 문장의 개수도 1개부터 수십 개로 다양하고, 각 문장의 길이도 천차만별이다. 무려 19,150의 길이를 갖는 outlier도 있다. `tokenizer(stride=128)`과 같은 효율적인 split을 고민해야 한다.
- Documents의 경우, 본문이 아닌 URL이므로 길이는 retrieval 하기 나름이다. 하나의 url은 43의 고정 길이를 가지므로, 대략 5개 전후의 url이 있다.

> #### Caution) Outliers in Documents for URL Request..
> Documents의 25%는 대략 10~15개의 URL을 가지고 있으며, 가장 많은 경우 무려 100개에 가깝다. 이는 URL request에서 상당한 load를 야기할 것이므로, 주의해서 처리해주어야 한다.

## 2.4. Question and Snippets.
```
- Question: Is JTV519 (K201) a potential drug for the prevention of arrhythmias?

- Snippets: 
We compared the suppressive effect of K201 (JTV519), a multiple-channel blocker and cardiac ryanodine receptor-calcium release channel (RyR2) stabilizer, with that of diltiazem, a Ca(2+ )channel blocker, in 2 studies of isoproterenol-induced (n = 30) and ischemic-reperfusion-induced VAs (n = 38) in rats.

(and many others...)

```

- 질문 및 관련 정보는 상당히 전문적인 의료 지식으로, general pretrained LLM은 잘 답변하지 못할 것이다. 따라서 medical NLP에 특화된 LLM이 필요하며, 최소한 classification head라도 fine-tuning은 피할 수 없을 것 같다.
- 몇몇 수동 대조 결과, Snippets는 Documents의 논문으로부터 추출된 정보로 보인다. 그러나 논문들에서 Snippets으로 추출되지 않은 유용한 정보가 꽤 있는 것으로 보인다. Documents로부터의 추가적인 정보가 도움이 될 것 같다.

## 2.5. Class Distribution.

```python
label_proportions = train_df['labels'].value_counts(normalize=True)
label_proportions.plot(kind='bar', legend=False)
```

```
labels
1    0.739171
0    0.260829
Name: count, dtype: float64
```

![](https://velog.velcdn.com/images/aruwad/post/93f01bdd-13cd-4b7b-bcff-c6df46d1aaf7/image.JPG)


- Train set의 label은 매우 uneven하다. 'yes'가 74% 가량을 차지하고 있다.
- Train set이 매우 작다는 사실과 더불어, 이는 상당한 overfitting을 야기할 것이다. 이에 대한 대책이 시급하다.

---

# 3. Non-RAG.

## 3.1. Data Preparation.

```python
cols_query     = ['question', 'labels']
train_ds_query = train_ds.select_columns(cols_query)
valid_ds_query = valid_ds.select_columns(cols_query)

train_ds_query
```

```
Dataset({
    features: ['question', 'labels'],
    num_rows: 1085
})
```

- 아이디어를 얻기 위해 `'question'`과 `'labels'`만을 사용해서 몇몇 모델을 테스트해보자.

## 3.2. No Fine-Tuning.

```python
import evaluate
from transformers import pipeline

# Model.
classifier      = pipeline("text-classification", model="distilbert-base-uncased")

# Metrics.
metric_accuracy = evaluate.load("accuracy")

# Predict.
predictions      = classifier(train_ds_query['question'])
y_pred           = [int(pred['label'].split('_')[-1]) for pred in predictions]     # convert to label.

# Compute accuracy.
accuracy = metric_accuracy.compute(predictions = y_pred, 
                                   references  = train_ds_query['labels'])

# Print.
print(f"Accuracy: {accuracy['accuracy']:.2f}")
```

```
Accuracy: 0.44
```

- 🤗의 `pipeline`과 `evaluate`을 사용하여 간단히 돌려보자.
- LLM은 오늘도 어김없이 등장하는 **DistilBERT** 군으로 시작하자. Classification head를 따로 훈련할 필요도 없다. <small>~~(거의 동네북)~~</small>
- 당연히 정확도는 좋지 않다. `accuracy = 0.44`로 동전 던지기보다 못하다.
- 그렇다면 이 도메인의 전문가 중 하나인 **BioBERT**는 어떨까? ~~<small>(자랑스런 한국인의 모델)</small>~~

```python
checkpoint = "dmis-lab/biobert-base-cased-v1.2"
```

![](https://velog.velcdn.com/images/aruwad/post/a7749933-eda9-4bca-9522-1f9bc8e21cc5/image.JPG)

- 정확도가 0.74라 높아보일 수 있겠지만 ~~<small>(앞에 안 읽으셨구나)</small>~~, f1을 보면 눈치 빠른 **BioBERT**가 1 epoch만에 얍삽한 방법을 파악했음을 알 수 있다. 대충 OMR에 한 줄로 긋는 것과 다를 것이 없다. 공부가 필요하다.
- 상당한 데이터를 학습한 **BioBERT**의 경우에도 추측 결과가 매우 극단적인 것으로 보아(거의 yes), 질문들이 **BioBERT**의 학습 시조차도 보지 못했던 data이고, 따라서 이 task가 상당히 domain-specific하다는 것을 알 수 있다.

## 3.3. Basic Training.
### 3.3.1. Model and Tokenizer.

![](https://velog.velcdn.com/images/aruwad/post/0bd59c2a-13de-4ea5-a2d1-a0512b82f373/image.JPG)

- 통상적인 🤗의 `AutoTokenizer` 및 `AutoModel`을 사용하였다.
- `tokenize_query(x)`
  - `max_length=512` : 아직 question 밖에 없긴 하지만, snippets 및 documents 를 추가하면 대부분 LLM들의 한계인 512를 대부분 넘길 것이므로, 512를 사용하였다.
  - `padding`은 여기서 하지 않는다. 할 수만 있다면 `DataCollator`를 통해 dynamically padding 하는게 대부분 더 좋다.
- `train_ds_query.map()`: 기초적인 부분이지만 혹시 모르는 분들을 위해서 ~~+취업을 위해서~~ 언급하면, `tokenize_query(train_ds_query)`로 직접 호출하는 건 좋지 않다. `.map()`을 사용하면 ~~🤗가 열심히 광고하는~~ Arrow의 장점, batch process 등을 사용하여 눈에 띄게 빨라진다.
- `remove_columns=['question']`: 항상 다 쓴 column은 휴지통에 잘 넣어주는 습관을 들이자. 나중에 shape 안 맞아서 개고생 할 수 있다.
  
> #### 🙋‍♀️ Dynamic Padding이 뭔가요?
> - 대부분의 transformer 기반 LLM들은 한 batch 내의 sample들이 동일한 길이를 가질 것을 요구한다. 따라서 짧은 sample들은 padding이 필요하다.
> - 그런데 `tokenizer(padding='max_length')`와 같이 `tokenizer` 내부에서 지정하면, 모든 512 미만의 sample들이 전부 512까지 padding 된다. 이는 연산속도, 메모리, 성능 등에 상당히 비효율적이다.
> - 🤗의 `DataCollator`를 사용하면 딱 batch가 구성된 이후에 그 batch에 대해 specific하게, 동적으로 padding과 같은 여러가지 처리를 할 수 있다. 이를 **Dynamic Padding**이라고 한다.
> - 예를 들어 이번 batch에서 가장 긴 놈이 24개의 토큰을 가지고 있으면, 얘보다 짧은 애는 24개까지만 padding 된다.
> - 🤗는 이 외에도 `DataCollatorForTokenClassification`, `DataCollatorForSeq2Seq` 등 여러가지를 지원하니 꼭 [확인해보자](https://huggingface.co/docs/transformers/main_classes/data_collator).

### 3.3.2. Train.

![](https://velog.velcdn.com/images/aruwad/post/f13665c1-f30c-4acd-998d-0525a81097cb/image.JPG)

- 🤗의 `TrainingArguments` 및 `Trainer`와 통상적인 hyperparameters로 통상적인 training을 구축하였다. ~~<small>뭐 죄다 통상적이래</small>~~
- `DataCollatorWithPadding`: 아까 봤지만 token length가 매우 다양하므로 dynamic padding을 해주었다.
- `bioasq_libs.compute_metrics`: Custom function으로 BioASQ 대회를 위한 metrics 계산을 해주었다 (accuracy, f1-yes, f1-no, macro f1).
- line 40 : Head-only training.
- 당연한 결과이지만, 학습이 잘 되지 않는다. 나같아도 저런 QA 보여줘봤자 1도 모를듯.
- 정확한 것은 아니지만, question들이 대략 IID를 만족한다고 대략 짐작할 수 있다. 그렇지 않으면 최소한 뭔가라도 배웠을테니까.
- 어차피 질문은 난생 처음 보는 해괴한 것들이므로, 사실상 class distribution을 보고 찍은 거나 다름없다. 그렇다면 class distribution을 제대로 줘보자.

### 3.3.3. Class-weighted Training.

![](https://velog.velcdn.com/images/aruwad/post/1aa41ae6-4db2-4f4d-afe9-861045fcfcbd/image.JPG)

- Python의 `Counter`를 사용하여 class weights를 구했다. 'no'의 weight이 약 3배가 되었다.
- 참고로 여기서 사용한 class weights는 $w_c = \frac{1}{n_c}$로, 단순히 샘플 수의 역수이다.

> #### 🙋‍♀️ 왜 Class weights를 normalize 하지 않고 그냥 쓰나요?
> 여기서 구한 Class weights는 바로 아래에서 training 시 loss의 값을 보정하는데 사용된다. 따라서 class frequency를 직접 loss에 반영할 수 있는 raw values가 일반적으로 더 낫다. 
> 특히 나는 더욱 세밀하게 class-weighted training 과정을 컨트롤하기 위해 hyperparameter를 추가하고 튜닝하였는데, 이럴 경우 더더욱 raw values가 낫다.

![](https://velog.velcdn.com/images/aruwad/post/7b4c87d3-f4a0-466a-bdbc-7d009d0c9508/image.JPG)

- Class-weighted training을 위해 Trainer를 상속받아 custom trainer를 만들었다. `compute_loss`를 overriding하면 기존의 `CrossEntropy`를 계산할 때 class weights를 고려한다. 즉, 이 경우 'no' sample의 경우 'yes' 보다 대략 3배의 패널티를 가한다.

> #### 🙋‍♀️ `global class_weights`는 뭔가요? 글로벌 변수라니 보기 불편하네요.
> 아시다시피 loss는 batch 단위로 계산된다. 따라서 여기서 `def compute_loss` 안에서 계산하면, class_weights는 **그 배치의 class_weights**가 되어버린다.
> 이는 당연히 의도하지 않은 결과이고, 결과적으로 class imbalance를 해결하지도 못할 뿐만 아니라, training 결과를 sample 순서에 상당히 민감하게 만든다.
>
> 이런 실수는 숙련자도 놓치기 쉬운데, 딱히 에러가 나는 것도, training 시 명확히 드러나는 것도 아니라 **나중에 찾기 진짜 어려울 수 있다**. 명심하자.

![](https://velog.velcdn.com/images/aruwad/post/b90b4eb5-07ac-4cba-8554-5c80b942940e/image.JPG)

- 예상대로 성능이 좋아졌다. 'no'를 틀리는 것에 3배 패널티를 주었고, **DistilBERT** 군은 마지못해 몇몇 'yes'를 'no'로 바꿨을 것이다. 그러나 역시 class distribution만을 보고 랜덤 추측한 것과 거의 다르지 않다.
- 0.70 등 숫자에 현혹되지 말자. 모두 'yes'로 예측하는 naive classifier는 정확도가 0.74이고 macro f1이 0.4254이다. 이를 넘지 못하는 모든 모델은 쓰레기다.

---

# 4. Training Optimization.
- 진짜 LLM으로 진짜 training을 하기 전에 training 과정의 optimization이 필요하다. ~~<small>부자라서 4090 10개 있으면 안해도 될지도 ㅎㅎ</small>~~
- **Quantization**: 일반적으로 floating point 변수 및 연산에 사용하는 16-bits/32-bits를 4-bits/8-bits로 압축해준다. 상당한 memory saving이 가능하여 요즘은 대규모 LLM 사용에 필수다.
- **PEFT**: Parametric-Efficient Fine-Tuning의 약자로, 이름처럼 training 할 parameter의 개수를 크게 줄여준다. 예컨대 10억개 파라미터 중 100만개만 사용해 training이 가능할 정도로 파격적인 방법이라, 역시 요즘은 필수다.
본 프로젝트에선 대표적인 adapter 방식 중 LoRA를 사용한다.
- **Optuna**: 본격적인 hyperparamter tuning을 위해 사용한다. 튜닝할 변수의 이름과 범위를 지정해주면, 알아서 '똑똑하게' 찾아준다. NLP 외에도 다양한 ML 분야에 국룰로 자리잡고 있다.

> #### 🙋‍♀️ QLoRA는 뭔가요?
> - 원래 Quantization과 PEFT는 전혀 별개의 영역이었다. 그러나 요즘은 대부분 둘을 (특히 LoRA와) 묶어 **QLoRA**로 부르며 널리 사용되고 있다. 왜일까?
> - 보통 quantization을 한 모델은 대부분의 library로 직접 training이 불가능하다. 대부분의 library는 최소 16-bits를 가정하고 구현되었기 때문이다.
> - 그러나 **LoRA**와 같은 adapter-based 방식은 기존 모델의 layer에 adapter라는 새로운 layer 자체를 추가하는 것이므로, quantized model도 훈련할 수 있게 해준다.
> - 한 마디로 **QLoRA**는 님 컴퓨터에서도 llama-10B 같은 분들을 training 할 수 있게 해주는 가장 편리한 방법이다!

## 4.1. QLoRA.

```python
# 4-bits Quantization.
from transformers import BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

skip_modules = ["classifier", "pre_classifier"]    # This modules are not quantized. Need to check with different pretrained LLMs.

quantization_config = BitsAndBytesConfig(
    load_in_4bit            = True,                # Quantize into 4-bits.                  
    llm_int4_threshold      = 6.0,                 # Layers whose norm of weights <= 6.0 are not quantized. 
    bnb_4bit_compute_dtype  = torch.float16,       
    low_cpu_mem_usage       = True,
    llm_int8_skip_modules   = skip_modules         # Need to check with different pretrained LLMs.
)

model_query = prepare_model_for_kbit_training(model_query)

# LoRA.
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r              = 8,                     # Rank of the low-rank decomposition.
    lora_alpha     = 32,                    # Scaling factor.
    lora_dropout   = 0.1,                   # Dropout.
    target_modules = "all-linear",          # Model-agnostic.
    task_type      = "SEQ_CLS",             # Task type = sequential clf.
    init_lora_weights="olora")
    
model_query = get_peft_model(model_query, lora_config)

# Print how many parameters we can ignore thx to LoRA :)
model_query.print_trainable_parameters()
```

```
trainable params: 739,586 || all params: 67,694,596 || trainable%: 1.0925
```

- 이제는 🤗 식구가 된 `bnb`와 `peft`로 QLoRA를 구현하였다.
- `skip_modules`: 가장 소중한 부분인 Classification head는 quantization에서 제외하였다.
- `llm_int4_threshold=6.0`: Layer의 weights의 norm이 6.0을 넘지 않으면 quantized 되지 않는다. Norm이 낮을수록 quantization의 왜곡이 심해지기 때문이다. 쉽게 말하면 100에서 1 빼면 1%지만 10에서 1 빼면 10%라는 것으로 이해하면 된다.
- `lora_dropout`: 새로 추가된 LoRA adapter들의 dropout이다. 당연한 얘기지만 중요한 hyperparameter다.
- `target_modules='all_linear'`: 국룰처럼 사용하는 세팅이다. 사실 난 head까지 건드는게 마음에 안들어 잘 안쓰지만, 다양한 checkpoint에 문제 없이 호환되어 자주 쓰인다.
- LoRA 덕분에 총 6700만개의 parameter 중 74만개만 훈련하면 된다. 아주 훌륭하다. ~~(근데 왜 1%지..?)~~

> #### 💁‍♂️ 4-bits라면서 왠 `'llm_int8_skip_modules'` 인가요? 오타아님?
> 아니다. 안타깝게도 🤗 `transformers ver. 4.40`부터 `bnb quantization_config`와 `peft`가 충돌이 나고 있다. `transformers`에 의해 `Linear4bit`로 바뀐 weights가 `modules_to_save`에 추가되고, 이를 `peft`가 training하기 위해 gradients 계산을 활성화시키려 하기 때문에 `TypeError`가 난다.
> 이는 적어도 작년 5월부터 있던 문제인데 아직도 안 고쳐졌고, workaround로서 `'llm_int8_skip_modules'`를 추가하는 것이 알려져있으니 [참고하자](https://github.com/huggingface/peft/issues/1720).

![](https://velog.velcdn.com/images/aruwad/post/d266fb0b-b182-4da8-a8d2-1f9ef71ff603/image.JPG)

![](https://velog.velcdn.com/images/aruwad/post/06c128c6-2f0c-4b39-b305-782673d2b52a/image.JPG)


- QLoRA를 적용하기 전(위)과 후(아래)의 training time을 비교하였다. 둘 다 DistilBERT를 1 epoch만 훈련하였다.
- 장난하냐고 할 수도 있겠지만, 모델이 커지고 toy data가 아닌 real data로 훈련하면 엄청난 차이가 벌어진다. 
(필자의 3070의 경우 당장 DistilBERT로 IMDb만 해보더라도 평균 x5배 이상 차이가 났다.)

## 4.2. Optuna.

```python
import optuna

# Objective ftn.
def objective(trial, model):
    # Hyperparameters to tune.
    hyperparams = {
        'learning_rate': trial.suggest_float('learning_rate', 3e-5, 5e-5, log=True)
    }
    
    ...
    
# Create Optuna study and optimize.
study = optuna.create_study(direction='maximize')
study.optimize(lambda trial: objective(trial, model_query), n_trials=2)

# Show best trial.
study.best_trial

```

- Hyperparameter tuning을 위한 objective 함수다.
- Optuna의 objective는 효율적인 탐색을 위해 하나의 독립적인 실행단위로 취급되므로, 가능하다면 밖에 있는 variable 및 functions를 가져오지 않는 게 좋다. ~~<small>작은 프로젝트에서는 그냥 training에 필요한 것들을 다 때려박으면 된다.</small>~~
- `trial.suggest`: 최적화 할 값들을 이런 식으로 제안해주면 알아서 찾아준다.
- `n_trials=2`: 이 횟수만큼 hyperparameter combination을 고르고 시도한다.
- `study.optimize()`: 이제 돌려놓고 편하게 놀고 오면 된다 ㅎㅎ.

# 5. Document Retrieval.

## 5.1. Data Preparation.

```python
cols_rag     = ['question', 'labels', 'snippets', 'documents']
train_ds_rag = train_ds.select_columns(cols_rag)
valid_ds_rag = valid_ds.select_columns(cols_rag)
```

- 드디어 본격적인 task 시작이다! `snippets`와 `documents`를 가져와 training set을 완성해보자.

## 5.2. Prompt.

```python
prompt_exact = """\
Question: {question}
Snippets:
{snippets}
Retrieved Chunks:
{retrieved_chunks}
Answer:
"""
```

- Retrieval에 사용할 간단한 promprt이다.
- 물론 prompt를 잘 깎아 query에 입혀야하지만, 다양한 prompt를 뒷받침할 시간과 computational resource가 도저히 없었기 때문에 그냥 retrieval에서 간단하게 처리했다.
- 사실 다양한 방식의 prompted engineering을 시도했고 몇몇은 꽤 성능을 향상시켰지만 커널이 죽어서 죄다 날려버렸다 ^^;

## 5.3. Retrieval from docs: raw docs -> DB -> retrieved docs.

```python
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

def retrieve_from_docs(query, documents, top_k_retrieval, chunk_size, chunk_overlap):
    
    # Text splitter.
    text_splitter = CharacterTextSplitter(separator      = ". ", 
                                          chunk_size     = chunk_size, 
                                          chunk_overlap  = chunk_overlap)
    
    # Embedding model.
    model_kwargs    = {'device': device}
    model_name      = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_model = HuggingFaceEmbeddings(model_name    = model_name,
                                            model_kwargs  = model_kwargs)
    
    # For each document, split, embed, and retrieve the most relevant chunk.
    retrievals = []
    chunks     = []
    for doc in documents:
        chunks.extend(text_splitter.split_text(doc))                     # Split the document into chunks
        
    vector_store = FAISS.from_texts(chunks, embedding_model)             # FAISS vector store.

    result = vector_store.similarity_search(query, k=top_k_retrieval)    # Similarity search to retrieve top-k relevant documents.
    retrievals.extend([res.page_content for res in result])              # Store relevant chunks.
    
    return retrievals
```

- 모든 URLs로부터 가져온 documents를 한번에 처리하여 vector DB에 저장하고, top-k relevent docs를 가져오는 함수이다.
- 원래는 각 URL, 즉 각 논문으로부터 top-k개를 가져오려 했으나, 100시간이 넘는 예상 시간과 PubMed가 request를 거부했다는 메세지를 보며 바로 마음을 바꿨다 ^^;
- `CharacterTextSplitter`: `langchain`의 character 기반 splitter다. Question/Snippets/Documents가 명확한 predefined word로 구분되고, 또 구분해야 하기 때문에, 통상적인 `RecursiveCharacterTextSplitter` 대신 사용했다.
- `HuggingFaceEmbeddings`: 통상적인 NLP tasks에 널리 쓰이는 `sentence-transformers/all-MiniLM-L6-v2`를 사용했다.
- Vector db 및 search에는 통상적인 `FAISS`를 사용했다.

## 5.4. Retrieval from URL: URL -> raw docs.

- URLs로부터 raw docs (논문)을 가져오는 부분이다.
- 논문의 본문에는 무료로 접근할 수 없으므로, abstract를 가져왔다.
- 우습게 봤다가 하루종일 씨름했던 부분이다. 그치만 덕분에 실무에서 중요한 부분을 많이 배웠다.

### 5.4.1. Single Sample.

```python
from langchain.document_loaders import WebBaseLoader

def preprocessing_rag(sample):
    results = []
    doc_urls  = sample['documents'].split('\n')    # URL of each document.
    documents = []

    for url in doc_urls:
        loader           = WebBaseLoader(url)
        document         = loader.load()[0]
        abstract_pattern_begin = 'AbstractPubMedPMID\n\n\n        Abstract\n        \n      \n\n\n      \n      '
        abstract_idx_begin     = document.page_content.find(abstract_pattern_begin) + len(abstract_pattern_begin)
        abstract_idx_end       = document.page_content.find('\n', abstract_idx_begin)

        abstract         = document.page_content[abstract_idx_begin:abstract_idx_end]
        documents.append(abstract)

    retrieved_chunks = retrieve_from_each_doc(query     = sample["question"],
                                              documents = documents)

    # Join retrieved chunks into a single string.
    retrieved_chunks = "\n".join(retrieved_chunks)

    # Use the prompt template to create the input text.
    input_text = prompt.format(
        question          = sample["question"],
        snippets          = sample["snippets"],
        retrieved_chunks  = retrieved_chunks
    )

    # Use the encoded label directly.
    labels = sample["labels"]
    
    return {'input_text': input_text, 
            'labels': labels}
```

- 하나의 sample의 URLs를 각각 읽고 한땀한땀 가져오는 것부터 시작했다. 즉 하나의 process가 (당연히) 비동기적으로 처리한다.
- `abstract_pattern_begin`: 그렇다, 하드코딩 맞다. 애초에 multithread & async 구현하려면 다 갈아엎어야 한다는 걸 알기에 일단 대충 작성했다. ~~<small>대표적인 보여주기식 코딩 ㅎㅎ</small>~~
- 너~~무나 오랜 시간이 걸려서 그냥 바로 다음으로 넘어갔다.

### 5.4.2. Single Batch.

```python
total_urls_in_batch = 0
for idx in range(len(batch["question"])):
    question  = batch['question'][idx]
    snippets  = batch['snippets'][idx]
    documents = batch['documents'][idx]
    label     = batch['labels'][idx]

    # Split documents into URLs.
    doc_urls = documents.split("\n")
    doc_contents = []

    total_urls_in_batch += len(doc_urls)
```

- 다음으로 구현을 batch로 확장했다. 본질은 request의 병렬화이므로 아직 큰 차이는 없겠지만, 그래도 좀 빨라졌다. ~~<small>(응 26시간\~)</small>~~

### 5.4.3. Multithread using `ThreadPoolExecutor`.

```python
def fetch_url(url):
    """Fetch content from a URL with timeout handling."""
    try:
        loader   = WebBaseLoader(url)
        document = loader.load()[0]
     ...

from concurrent.futures import ThreadPoolExecutor, as_completed
def process_batch_parallel(batch):
    """Call fetch_url with multithreaded processes!"""
    inputs = []
    labels = []

    for idx in range(len(batch["question"])):
        question = batch['question'][idx]
        snippets = batch['snippets'][idx]
        documents = batch['documents'][idx]
        label = batch['labels'][idx]

        doc_urls = documents.split("\n")
        doc_contents = []

        # Parallel fetching of URLs.
        global num_workers
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_url = {executor.submit(fetch_url, url): url for url in doc_urls}
            for future in as_completed(future_to_url):
                doc_contents.append(future.result())
        
        retrieved_chunks = retrieve_from_each_doc(query=question, documents=doc_contents)
        retrieved_chunks = "\n".join(retrieved_chunks)
        
        ...
  ```

- `fetch_url`: 위에서 했던 request를 multithread로 처리하기 위해 간단한 함수로 묶었다.
- `ThreadPoolExecutor(max_workers=num_workers)`: 여러 thread들이 request를 하나씩 집어 담당한다. 
- `as_completed`: 모든 thread들이 request 수행을 완료할 때까지 기다렸다가 모아서 append 한다.
- Single process 때 26시간이 걸렸던 것에 비해 대략 6~7시간으로 단축할 수 있었다! 그러나 여전히 여러 방식을 테스트하기엔 너무 길었다.

> #### 💁‍♂️ URL Request에서 Multi-threading 방식의 한계.
> - URL request는 내 리소스가 아무리 뛰어나도 서버에 따라 실행시간이 차이가 많이 난다.
> - Multi-threading은 동시에 여러 thread가 request하여 한 번에 많은 request가 가능하지만, 결국 한 thread 당 하나의 request만 가능하고, 다른 모든 thread가 수행 완료할 때까지 기다려야 하며, 기다리는 동안 메모리 등 리소스를 가진 채로 있어야 한다는 단점이 있다.
> - 반면 아래의 async에서 하나의 thread는 다른 녀석들이나 request의 response를 기다리지 않고 계속 request를 날린다. 잠깐만 상상해봐도 얼마나 빨라질지 기대되지 않는가? ~~<small>직접 겪어보면 분명 기대될 것이다</small>~~

### 5.4.4. `asyncio`.

```python
import aiohttp
import asyncio
from bs4 import BeautifulSoup

async def fetch_url_async(session, url):
    """Fetch content from a URL asynchronously."""
    try:
        async with session.get(url, timeout=10) as response:
            html = await response.text()
            soup = BeautifulSoup(html, 'html.parser')
            
            abstract_div = soup.find('div', class_='abstract-content selected')
            
            if abstract_div:
                abstract_list = abstract_div.find_all('p')
                
                # if abstract is found, return with title.
                if abstract_list:   
                    abstract = " ".join(p.get_text(strip=True) for p in abstract_list)
                    return abstract
                else:
                    return f"abstract not found."
                
            else:
                return f"abstract_div not found."
            
    except asyncio.TimeoutError:
        return f"Timeout occurred: {url}"
    
    except Exception as e:
        return f"Failed to load URL: {str(e)}"

async def fetch_all_urls_async(doc_urls):
    """Fetch all URLs concurrently using aiohttp."""
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url_async(session, url) for url in doc_urls]
        
        return await asyncio.gather(*tasks)
```

- 대망의 비동기적 request 구현이다! 실행시간이 무려 **80분**으로 줄었다!
- `aiohttp.ClientSession()`: async request를 위한 session이다. 기존의 동기 방식인 `ThreadPoolExecutor` 및 `WebBaseLoader`를 대체한다.
- `BeautifulSoup`: 기존의 하드코딩 대신 HTML parsing을 위한 bs4이다. 깔끔하게 abstract를 찾는다.

> #### 💁‍♂️ Many Requests at Once.
> 단시간에 너무 많은 request를 날리면 서버에서 당신을 차단할 수 있다!
> 이는 기술적인 문제라기보단 현실적인 문제, 그리고 양심의 문제로, 본격적으로 날리기 전에 API 문서나 테스트를 꼭 해보자!

## 5.5. Retrieval Result.

![](https://velog.velcdn.com/images/aruwad/post/76e08807-50d6-4c7a-b70b-77b036e8e160/image.JPG)

- 소중한 데이터이니만큼 저장을 잊지 말자!
- 요청한 prompt 양식으로 잘 만들어 준 것을 확인할 수 있다.

# 6. RAG Model Training.

## 6.1. Data Augmentation by Splits.

```python
def split_by_snippets_docs(ds, only_no):
    if only_no:
        if label == 0:
            # add with snippets.
            splitted_inputs.append(question + snippets + "\nAnswer:\n")
            splitted_labels.append(label)

            # add with docs.
            splitted_inputs.append(question + "\nSnippets:\n" + docs)
            splitted_labels.append(label)

        # if 'yes', just copy original sample.
        else:
            splitted_inputs.append(sample['input'])
            splitted_labels.append(label)
    else:
        # add with snippets.
        splitted_inputs.append(question + snippets + "\nAnswer:\n")
        splitted_labels.append(label)

        # add with docs.
        splitted_inputs.append(question + "\nSnippets:\n" + docs)
        splitted_labels.append(label)
```

- Uneven class distribution 및 lack of data를 handling하기 위해 data augmentation을 시도했다.
- 원래의 'question-snippets-documents' 구조를 다양한 방식으로 분할했다.
  - **'q-s'** and **'q-d'**: snippets과 documents를 나누기 (2개).
  - **'q-s_1', 'q-s_2', ..., 'q-s_n'** and **'q-d'**: 각 snippet 별로 모두 나누고, 추가로 documents 째로 나누기 (n+1개).
  - **'q-s_1', 'q-s_2', ..., 'q-s_n'** and **'q-d_1', 'q-d_2', ..., 'q-d_m'**: 각 snippets 및 documents 별로 모두 나누기 (n x m개).
  - Rare class의 경우에 가중치를 부여해 더 자주 나누기.
  - (기타 수많은 시도들...)
- 20시간 가량의 수많은 시도들 끝에, **'no' sample만 'q-s' and 'q-d'로 나누는 방식**이 합리적인 training time 내에 가장 좋은 결과를 냈다. 
~~<small>(살려주세요 Optuna님 ㅠㅠ)</small>~~

> #### 💁‍♂️ Data Augmentation and Shuffle.
> 이 방식의 Data Augmentation 후에는 꼭 Dataset을 섞어야 한다. **'q-s'**와 **'q-d'**는 당연히 높은 상관성이 존재하기 때문이다.

## 6.2. Stride.

```python
# Tokenization.
def tokenize_rag(sample):
    return tokenizer_rag(sample['input'], 
                         truncation      = True, 
                         padding         = 'max_length',
                         max_length      = hyperparams['max_length'],
#                        stride          = hyperparams['stride'],
#                        return_overflowing_tokens = True,
                         return_tensors  ='pt')

# Custom collate function to handle the shape after stride.
def collate_fn(batch):
    input_ids       = []
    attention_masks = []
    labels          = []

    for example in batch:
        input_ids.extend(example['input_ids'])                          # Extend all chunks.
        attention_masks.extend(example['attention_mask'])               # Extend attention masks.
        labels.extend([example['labels']])  # Extend labels.
#            labels.extend([example['labels']] * len(example['input_ids']))  # Duplicate labels, for stride.

    return {
        'input_ids'      : torch.tensor(input_ids),
        'attention_mask' : torch.tensor(attention_masks),
        'labels'         : torch.tensor(labels),
    }

```

- 이 task처럼 sequence의 길이가 불규칙하고 툭하면 `max_length`를 넘는 경우, `stride`가 널리 사용된다.
- `stride`를 사용하는 경우 `input`의 shape이 변하므로, 이를 적절히 처리해줘야 한다. 의외로 이걸 잘 모르시는 분들이 많아 간단히 설명해보겠다.
  
  
> #### 💁‍♂️ `tokenizer(stride)` 후 shape 처리하기.
> - 예컨대 `txt = [1,2,3,4,5,6], max_length=4, stride=2`라고 하자.
>  - Tokenize 결과는 `token_1 = [1,2,3,4], token_2 = [3,4,5,6]`이 된다.
>  - 이제 `input`의 shape는 `(n_samples, 6)`에서 `(n_samples, 4, 2)`가 된다.
>  - 근데 여기서 '2'는 txt의 길이에 따라 달라진다. 예컨대 `[1,2, ..., 8]`은 `[1,2,3,4], [3,4,5,6], [5,6,7,8]`로 '3'이 된다.
>  - 더군다나 `label`의 shape은 아직 `(n_samples, 1)`로 변하지도 않았다.
> - 이를 처리하기 위해선,
>   1. 각 token을 꺼내주고 각각 label을 달아줘서 별개의 sample로 취급하던가 (예컨대 (4, 2)를 (4,1)짜리 2개로),
>   2. 1처럼 하되 각 token 별로 logits을 계산하고 평균 등으로 취합해 sample 별로 하나의 prediction을 만들던가 ~~<small>(쁘띠 앙상블)</small>~~,
>   3. Longformer나 BigBird처럼 긴 sequence 전용 transformer를 쓰던가,
>   4. Summarization 등 무슨 수를 써서라도 length 자체를 줄이던가, 등등 많은 방법이 있다.
> - 나는 **수많은** 시도 끝에, `stride`를 하지 않고, 앞서 언급한 data augmentation 방법으로 length를 줄이고(방법 4), logits의 평균 계산(방법 2)을 결합하는 방식을 택했다.

## 6.3. Hyperparameter Tuning.

```python
# Hyperparameters to tune.
hyperparams = {
    'n_epochs': 30,
    'batch_size': trial.suggest_categorical('batch_size', [8, 16]),
    'max_length': 512,
    'weight_decay': trial.suggest_float('weight_decay', 2e-2, 3e-2),
#    'stride': trial.suggest_categorical('stride', [128, 256]),
    'warmup_ratio': trial.suggest_float('warmup_ratio', 0.08, 0.12),     # warmup_steps = total_steps * warmup_ratio.
    'learning_rate': trial.suggest_float('learning_rate', 5e-5, 5e-4),
    'lora_dropout': trial.suggest_float('lora_dropout', 0.1, 0.2),
    'class_weights_penalty': trial.suggest_float('class_weights_penalty', 1.2, 2.0),     # multiply class weights of less frequent class, i.e. 'no'.
    'early_stopping_patience': 3,
    'early_stopping_threshold': 1e-3,
    'max_grad_norm': 1.0,
#    'gradient_accumulation_steps': trial.suggest_categorical('gradient_accumulation_steps', [0, 1]),
#    'lr_scheduler_type': trial.suggest_categorical(
#        'lr_scheduler_type', ['linear', 'cosine', 'cosine_with_restarts']
#    )
}

checkpoint = trial.suggest_categorical(
    "checkpoint",
    [
        "bert-base-uncased",
#            "distilbert-base-uncased",
        "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        "dmis-lab/biobert-base-cased-v1.2",
        "kamalkraj/bioelectra-base-discriminator-pubmed",
    ]
)

```



- 사용한 hyperparameter의 값 및 suggestion이다. 범위를 줄이고 줄여서 마지막 무렵에 study한 것들이다.
- `n_epochs: 30`: 많은 경우처럼 큰 epochs를 두고 `earlystopping`으로 조절했다.
- `batch_size`: 물론 16이 더 빨랐지만, small dataset과 class imbalance, domain specific task 등을 고려해 overfitting에 대응하기 위해 많은 경우 8이 선호되었다. 특히나 long sequence가 많은 task 특성상 16만 해도 자주 커널이 죽어서 반강제로 8을 선호하게 되었다.
- `max_length: 512`: `stride`를 적용하고 128, 256, 512를 여러 조합으로 테스트해봤다. 그치만 결론은 허무하게도 `stride` 빼고 512 고정이다.
- `weight_decay`: Overfitting 가능성이 농후한 task 특성상 약간 높게 시작하여 낮춰나갔다.
- `warmup_ratio`: Overfitting 가능성이 농후한 task 특성상 (이하 동일 ㅎㅎ) 하지 않을 수 없었다. 특히 이 task에선 **높은 learning rate**의 catastrophic forgetting 등의 불안정성을 낮춰줌으로써 좋은 시너지를 보였다.
- `learning_rate`: 이 task에선 domain specific한 data들을 aggressive하게 학습해야 한다고 판단해서 통상적인 (1e-5, 5e-5) 수준보다 높게 두고 overfitting을 handling하는 방식으로 접근했다. ~~<small>(본격 병주고 약주기)</small>~~
- `lora_dropout`: 역시 통상 수준인 0.1보다 약간 높게 두었다. 처음엔 task-specific learning이라는 특징을 활용해보고자 낮추고 다양하게 adapter를 추가해보았는데, 잘 작동하지 않아 그냥 뺐다. 
- `class_weights_penalty`: Unfrequent class에 더욱 penalty를 주었다. Data augmentation을 하지 않은 경우 1.5는 줘야 겨우 극심한 'yes' 편향을 벗어나는 것 같았다. 'yes'로 다 찍어도 70%가 보장된다는게 모델들에게 참 위안이 되었나보다 ^^.
- `early_stopping_patient` 및 `threshold`: 한 epoch에 30분 이상이 걸려서 참을 수가 없었다...
- `max_grad_norm`: 높은 learning rate로 인한 gradient explosion을 제어하기 위해 clipping을 추가하였다.
- `gradient_accumulation_steps`: 큰 `batch_size`가 불가능해 시도해보았지만, 역시 과도한 overfitting으로 끄는게 나았다.
- `lr_scheduler_type`: 많이 고민했던 부분이나, 현실적으로 장기간 training이 어려웠고 기껏해야 20 epochs 정도여서 그냥 `linear`로 갔다.

## 6.4. Ensemble.

```python
# Soft voting.
# Perform ensemble (mean of logits)
ensemble_logits = np.mean(logits_list, axis=0)

# Convert logits to predictions (soft voting)
preds_soft = np.argmax(ensemble_logits, axis=1)

# Convert preds_soft_onehot to one-hot encoding.
preds_soft_onehot = np.zeros((len(preds_soft), 2))
preds_soft_onehot[np.arange(len(preds_soft)), preds_soft] = 1


# Hard voting.
preds_hard = np.stack(preds_list, axis=0)
preds_hard = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=preds_hard)

# Convert final_predictions_hard to one-hot encoding.
preds_hard_onehot = np.zeros((len(preds_hard), 2))
preds_hard_onehot[np.arange(len(preds_hard)), preds_hard] = 1

# Compute metrics.
test_results_soft = bioasq_libs.compute_metrics((preds_soft_onehot, test_ds['labels']))
test_results_hard = bioasq_libs.compute_metrics((preds_hard_onehot, test_ds['labels']))
```

- Classification 대회의 국룰인 Ensemble도 시도해보았다. 전형적인 soft/hard voting부터 시작했다.
- 현실적인 실행 시간을 고려해 training 시 계산한 prediction을 저장했다 이후 가져오는 방식을 택했다.
- 애써 구현했지만 돌리자마자 극심하게 늘어나는 실행 시간에 황급히 삭제해버렸다 ^^.

## 6.5. Results.

### 6.5.1. Metrics.

![](https://velog.velcdn.com/images/aruwad/post/2428796a-5c43-4c2b-b23e-e8ef44848248/image.JPG)

![](https://velog.velcdn.com/images/aruwad/post/f37f38d4-1238-40cb-b3df-a465544fb4db/image.JPG)


- 결과적으로 best_params에서 0.9541의 macro f1을 달성하였다.
- 전형적인 Overfitting과의 전쟁이었고, 위에 언급한 여러 방법들로 줄여나간 결과이다.
- 3일 밤낮을 눈이 빠져라 돌리며 얻은 결과다. 여러차례 연이어 training 했기 때문에 1 epoch부터 결과가 좋아보이지만, 대략 1~20 epoch 정도 앞에 추가해서 상상하면 된다. ~~<small>보고있자니 왠지 허무하다...</small>~~
- 그 놈의 메모리 때문에 자꾸 커널이 죽어서 TensorBoard 결과도 날려먹었다. 전체 그래프 보여드리지 못한 점 심심한 사과의 말씀...

### 6.5.2. Feature Importances.

![](https://velog.velcdn.com/images/aruwad/post/eb8be2e3-3293-45ec-b369-0b225eb9e74d/image.JPG)

- 사실 첫 study의 결과는 확실히 달랐다. Learning rate와 weight_decay이 훨씬 높았는데, 위 그림은 막바지 study에서 tuning할 대로 한 이후라 영향이 크지 않아 보인다.
- 당연하게도 모델의 비중이 매우 높다. 비슷한 BERT 계열의 bio-specific한 놈들 + Bert-base-uncased 하나 남겼는데도 차이가 꽤 컸다.
- Dropout은 뭐... 어떤 task에서도 항상 높고 딱히 뺄 수도 없다.
- Warmup의 경우 위에 잠깐 언급했지만, 확실히 높은 learning rate의 경우 중요도가 증가하는 경항을 보였다.

# 7. 결론 및 아쉬운 점.
- 본 프로젝트는 domain specific task로 악명 높은 medical NLP 분야의 유명한 competition에 도전해보았다.
- Industry의 small project 수준의 적당한 깊이로, RAG를 구축하고 fine-tuning하여 적당한 결과를 달성하였다. 이로서 RAG와 pretrained LLM의 fine-tuning의 기초는 확실하게 다진 것 같다.
- 시도해보고 싶은 것들이 정말 많았지만, 이미 종료된 대회라는 점, 그리고 지금 나의 상황이 portfolio를 위해 단기간에 다양한 task에서 다양한 tech stack에 대한 숙련도를 보여주어야 한다는 점에서, 아쉽지만 더 파고들지 못하고 마무리했다.
- 결과를 정리하고 포스팅하는 것이 생각보다 엄청나게 시간과 노력이 많이 든다. TensorBoard 등 visualization tool을 적극 활용하고, 블로그를 간결하게 쓰는 연습을 계속 해야겠다.
