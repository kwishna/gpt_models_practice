from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

model_name = "deepset/tinyroberta-squad2"

# a) Get predictions
QAndA = pipeline('question-answering', model=model_name, tokenizer=model_name)
# _input = {
#     'question': 'Why is model conversion important?',
#     'context': 'The option to convert models between FARM and transformers gives freedom to the user and let people easily switch between frameworks.'
# }
# res = QAndA(_input)
# print(res)

# b) Load model & tokenizer
# model = AutoModelForQuestionAnswering.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)


_input = {
    'question': "What is artificial intelligence?",
    'context': open('../../ai.txt', 'r').read().replace('\\n', ' ')
}
res = QAndA(_input)
print(res['answer'])
