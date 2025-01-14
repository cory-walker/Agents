# Agents

This is a collection of classes and procedures meant to help search for relevant videos, download transcripts, and perform NLP and docuement classification. It can write first draft articels, interface with LLMs, and perform Named Entity Recognition (NER). This work in progress are my first attempts at building tools to enable people to monitor and analyze mass media in a more efficient way.

## Features

Class: Watcher

- Over program. This can utilize instances of the sub classes to automate and combine different functions into chained routines. Used mostly to batch transcript fetching, topic modelling, entity recognition, or document classification for all untouched files in one function call.

Class: YouTubeEplorer

- Searches YouTube, storing results and metadata in local parquet files. Requires YouTube data API Key. Filter searches by Channel is optional, but you need to know the "channel_id"
- Fetches Closed Caption transcripts for YouTube videos.

Class: Scholar

- Via DSPY, interacts with LLMs for document classification, article writing, summarization, and topic identification. Saves results locally as parquet.
- Via Spacy, performs Named Entity Recognition, saving results locally as parquet.

Class: Key Ring

- Holds onto API or other keys in a convient way to pass them programmatically across functions.

Class: Librarian

- Keeps track of the different files and metadata, builds and maintains an index for easier lookup.
- Can retrieve metadata about transcripts or other items from the document stores.

Code Corner files

- Some pieces of the project may be shared out via social media. When that takes place, a copy of the article Jupyter notebook will be added here.
