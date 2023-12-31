from transformers import pipeline
summarizer = pipeline("summarization", model="philschmid/bart-large-cnn-samsum", max_length=100)

conversation = '''Jeff: Can I train a 🤗 Transformers model on Amazon SageMaker? 
Philipp: Sure you can use the new Hugging Face Deep Learning Container. 
Jeff: ok.
Jeff: and how can I get started? 
Jeff: where can I find documentation? 
Philipp: ok, ok you can find everything here. https://huggingface.co/blog/the-partnership-amazon-sagemaker-and-hugging-face.
'''
print(summarizer(conversation)[0]['summary_text'])
