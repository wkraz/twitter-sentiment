# Tweet Sentiment Tool

## Usage
First, install dependencies
```
pip install -r requirements.txt
```
Then, just run the driver program once to let all the models build themselves
```
python main.py
```
After the initial run, the models won't have to load themselves again so you can just run `python main.py` and you'll be asked: 
```
Choose a mode: 'interactive' for text input or 'twitter' for Twitter analysis:
```
If you choose `interactive`, you will be asked to input a text block and will see the predicted sentiment (positive or negative), and the model's confidence
If you choose `twitter`, you will be asked for a hashtag to analyze tweets for, and you will see the sentiment of `n` (editable) amount of tweets with said hashtag.

## Editing
This program is editable in many parts.
First, there are plenty of well-trained sentiment models available. I chose the `cardiffnlp/twitter-roberta-base-sentiment-latest` model, because it is a compact model that is trained on social media data, which was perfect for looking at tweets on my personal computer. Just edit the line:
```
sentiment_model = pipeline("sentiment-analysis", model="________", device="cpu")
```
to any of the following models: \
`cardiffnlp/twitter-roberta-base-sentiment-latest` - trained on social media text \
`distilbert-base-uncased-finetuned-sst-2-english` - lighter version of BERT trained on movie reviews \
`bert-large-uncased` - large BERT trained on SST-2, useful if you have a more powerful machine on hand \
`xlnet-base-cased` - transformer based model designed for capturing nuance \
VADER (requires more than one step to use, but very simple) - lexicon-based (not neural network), also optimized for social media text \
TextBlob (also requires more than one step to use, but simple) - lexicon-based and trained on various language datasets, very user-friendly and good for simple text 

Next, edit the line:
```
response = client.search_recent_tweets(query=query, max_results=__)
```
This is only applicable in Twitter mode, but you can change the max results to any number (just know that as `n` increases, the strain on the machine increases, and my Mac cannot handle more than 10, lol). 

You can also add more modes than Twitter and Interactive like Reddit, you will just have to add the appropriate variables to the `.env` file and call them appropriately. 
