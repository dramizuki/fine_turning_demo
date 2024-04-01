from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

japanese_model = 'cl-tohoku/bert-base-japanese-whole-word-masking'
tokenizer = AutoTokenizer.from_pretrained(japanese_model)
model_directory = './test_model'
model = model = AutoModelForSequenceClassification.from_pretrained(
    model_directory,
)

analyzer = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

label2name = {
    'LABEL_0': '京都',
    'LABEL_1': '携帯',
    'LABEL_2': 'グルメ',
    'LABEL_3': 'スポーツ',
}


texts = [
    "米を食べたい",
    "最新のスマホは何？",
    "カープの成績を教えて",
    "京都のおすすめ観光地を教えて",
]

def label_name(analyze_result: dict) -> str:
    label = analyze_result[0]['label']
    return label2name[label]

for text in texts:
    print(f"{text} -> {label_name(analyzer(text))}")