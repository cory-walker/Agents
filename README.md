# Agents

The media can have bias, their owners may have agendas, and there is just so much being sent to us that we can lose the message. We can leverage modern programming and agents to help, and this is my attempt.

This is a collection of classes and procedures meant to help search for relevant videos, download transcripts, and perform NLP and docuement classification. It can write first draft articles, interface with LLMs, and perform Named Entity Recognition (NER). This work in progress are my first attempts at building tools to enable people to monitor and analyze mass media in a more efficient way.

## Classes

### Watcher

- Over program. This can utilize instances of the sub classes to automate and combine different functions into chained routines. Used mostly to batch transcript fetching, topic modelling, entity recognition, or document classification for all untouched files in one function call.

### YouTubeEplorer

- Searches YouTube, storing results and metadata in local parquet files. Requires YouTube data API Key. Filter searches by Channel is optional, but you need to know the "channel_id"
- Fetches Closed Caption transcripts for YouTube videos.

### Scholar

- Via DSPY, interacts with LLMs for document classification, article writing, summarization, and topic identification. Saves results locally as parquet.
- Via Spacy, performs Named Entity Recognition, saving results locally as parquet.

### Key Ring

- Holds onto API or other keys in a convient way to pass them programmatically across functions.

### Librarian

- Keeps track of the different files and metadata, builds and maintains an index for easier lookup.
- Can retrieve metadata about transcripts or other items from the document stores.

### Code Corner files

- Some pieces of the project may be shared out via social media. When that takes place, a copy of the article Jupyter notebook will be added here.

## Scripts

### SearchForNewVideos.py

- Arguments: Location of your YouTube API key file
- What it does:
  - Performs a new search for each YouTube channel in the youtube_channels_index.csv
  - Attempts to download the closed captions transcript for any video it doesn't have one for.

### ClassifyTranscipts.py

- Arguments: Location of your OpenAI API key file
- What it does:
  - Builds a Named Entity file using Spacy for any transcripts that don't have one.
  - Builds a Topic list file using ChatGPT for any transcripts that don't have one.
  - "Classifies" any transcripts that doesn't already have the file for ChatGPT. This includes: Summarization, Primary Emotion, Secondary Emotion, and Sentiment.

## Notes

- To keep costs lower through testing, I have limited my list of channels to Canadian news sites. You can add any you like to your copy.
