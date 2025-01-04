'''
The Scholar program seeks to find meaning in texts, classifying it, and building reports
'''

# from openai import OpenAI
import dspy
import os
from typing import Literal
import pandas as pd
# import pyarrow as pa
# import pyarrow.parquet as pq
import spacy
import Keyring
import re


class Interaction:

    class EmotionSecondary(dspy.Signature):
        '''Classify secondary emotion'''
        text: str = dspy.InputField()
        sentiment: Literal['affection', 'lust', 'longing', 'cheerfulness', 'zest', 'contentment', 'pride', 'optimism', 'enthrallment', 'relief', 'surprise', 'irritability',
                           'exasperation', 'rage', 'disgust', 'envy', 'torment', 'suffering', 'disappointment', 'shame', 'neglect', 'sympathy', 'horror', 'nervousness'] = dspy.OutputField()

    class Emotion(dspy.Signature):
        '''Classify Emotion'''
        text: str = dspy.InputField()
        sentiment: Literal['sadness', 'joy', 'love',
                           'anger', 'fear', 'surprise'] = dspy.OutputField()

    class TopicList(dspy.Signature):
        '''Create a list of topics from a given text.'''
        text = dspy.InputField()
        topics = dspy.OutputField(desc='A comma separated list of topics')

    class ArticleOutline(dspy.Signature):
        '''Write an article outline about a given topic. Please use the provided text as the primary information source.'''
        topic = dspy.InputField()
        text = dspy.InputField()
        article_outline = dspy.OutputField(
            desc='A comma separated list of sections')

    class TopicToParagraph(dspy.Signature):
        '''Write a paragraph for a specific section of an article about a given topic. Please use the provided text as the primary information source.'''
        topic = dspy.InputField(desc='The topic of the article')
        text = dspy.InputField(desc='Primary source of information')
        section = dspy.InputField(
            desc='The section of the article being written')
        paragraph = dspy.OutputField(
            desc='The paragraph for the section of the article')

    class ProofReader(dspy.Signature):
        '''Proofread an article and output a more well written version of the original article. Please consolidate like ideas in a concise way.'''
        article = dspy.InputField()
        proofread_article = dspy.OutputField()

    class ArticleTitle(dspy.Signature):
        '''Write a title for a provided article.'''
        article = dspy.InputField()
        title = dspy.OutputField()

    class TextRefinerWithEmotion(dspy.Signature):
        '''Read the given text and output a more well written version of the original text, evoking the listed primary and secondary emotions.'''
        text = dspy.InputField()
        primary_emotion = dspy.InputField()
        secondary_emotion = dspy.InputField()
        refined_text = dspy.OutputField()


class EntityList:
    def __init__(self, entities=[]):
        self.original_list = entities
        self.entity_types = {
            'PERSON': 'People, including fictional', 'NORP': 'Nationalities, religious groups, political groups', 'FAC': 'Buildings, airports, highways, bridges, etc', 'ORG': 'Companies, agencies, institutions, etc', 'GPE': 'Countries, cities, states', 'LOC': 'Non-GPE locations, mountain ranges, bodies of water', 'PRODUCT': 'Objects, vehicles, foods, etc (Not services)', 'EVENT': 'Named hurricanes, battles, wars, sports events, etc', 'WORK_OF_ART': 'Titles of books, songs, etc', 'LAW': 'Named documents made into laws', 'LANGUAGE': 'Any named language', 'DATE': 'Absolute or relative dates or periods', 'TIME': 'Times smaller than a day', 'PERCENT': 'Percentage, including "%"', 'MONEY': 'Monetary values, including unit', 'QUANTITY': 'Measurements, as of weight or distance', 'ORDINAL': '"first", "second", etc', 'CARDINAL': 'Numerals that do not fall under another type'
        }

    def unique_entities(self):
        '''Returns a unique list of entities from the original list.'''
        ulist = []
        for d in self.original_list:
            if not d in ulist:
                ulist.append(d)
        return ulist


class Scholar:
    '''
    Studies text and provides interpretations. It can utilize NLP techniques via Spacy, and works with ChatGPT api for LLM support.
    '''

    def __init__(self, save_location='./data/', keyring=Keyring.Keyring(), openai_model='gpt-4o-mini', spacy_model='en_core_web_sm', openai_max_tokens=4096, text=''):
        self.save_location = save_location
        self.keyring = keyring
        self.openai_model = openai_model
        self.openai_max_tokens = openai_max_tokens
        self.spacy_model = spacy_model
        self.lm = None
        self.nlp = None
        self.text = text
        self.entity_list = EntityList()

        self.configure_openai_language_model()
        self.configure_spacy_nlp_model()

    def clean_text(self):
        '''Cleans the saved text with simple replacement and removal of repeated words'''
        to_remove = ['um', 'uh', '[ __ ]', '[Music]', 'Â']
        txt = self.text.replace('\xa0', ' ')
        txt = re.sub(' +', ' ', txt)
        tokens = txt.split(' ')
        keepers = []
        refined = []

        for t in tokens:
            if not t in to_remove:
                keepers.append(t)

        for i in range(0, len(keepers)):
            t = keepers[i]
            next_t = ''
            if i < len(keepers)-1:
                next_t = keepers[i+1]
            if t != next_t:
                refined.append(t)
        return ' '.join(refined)

    def render_text_with_entities(self, cleaned=False):
        if cleaned:
            doc = self.nlp(self.clean_text())
        else:
            doc = self.nlp(self.text)

        return spacy.displacy.render(doc, style='ent')

    #! Build in a way to refine names. Seek those with spaces, and then convert single word ones in reference to the full name

    def extract_entities(self):
        '''Extracts named entities from the text using spacy'''
        if self.nlp is None:
            self.configure_spacy_nlp_model()

        doc = self.nlp(self.text)
        entities = []
        for ent in doc.ents:
            entities.append({ent.label_: ent.text})
        self.entity_list.original_list = entities

    def configure_openai_language_model(self):
        '''Attempts to configure the dspy openai language model'''
        try:
            self.lm = dspy.LM(self.openai_model, api_key=self.keyring.get_key(
                'openai'), max_tokens=self.openai_max_tokens)
            dspy.configure(lm=self.lm)
        except:
            print('Error: Could not configure an openAI model. Please check your keys')

    def configure_spacy_nlp_model(self):
        '''Attempts to configure the spacy NLP model'''
        try:
            self.nlp = spacy.load(self.spacy_model)
        except:
            print('Error: Could not configure the spacy NLP model.')

    def create_summary(self):
        '''DSPY: Summarize the text and return the summary and underlying reasoning'''
        summarize = dspy.ChainOfThought('document -> summary')
        response = summarize(document=self.text)
        return response.summary, response.reasoning

    def analyze_sentiment(self):
        '''DSPY: Returns True for positive and False for negative sentiment of the text'''
        classify = dspy.Predict('sentence -> sentiment: bool')
        response = classify(sentence=self.text)
        return response.sentiment

    def analyze_emotion(self):
        '''DSPY: Returns the emotion most evoked by the text'''
        classify = dspy.Predict(Interaction.Emotion)
        response = classify(text=self.text)
        return response.sentiment

    def analyze_secondary_emotion(self):
        '''DSPY: Returns the secondary emotion most evoked by the text'''
        classify = dspy.Predict(Interaction.EmotionSecondary)
        response = classify(text=self.text)
        return response.sentiment

    def create_overall_classification(self):
        '''DSPY: runs the procedures for summary, sentiment, primary and secondary emotions'''
        summary, reasoning = self.create_summary()
        sentiment = self.analyze_sentiment()
        primary_emotion = self.analyze_emotion()
        secondary_emotion = self.analyze_secondary_emotion()

        metadata = {'summary': summary, 'summary_reasoning': reasoning, 'sentiment': sentiment,
                    'primary_emotion': primary_emotion, 'secondary_emotion': secondary_emotion}

        return metadata

    def refine_text_with_emotions(self, primary_emotion, secondary_emotion):
        '''Rewrites a text but with more [Emotion] and [Secondary emotion]'''
        current_word_ct = len(self.text.split(' '))
        needed_to_change_max_tokens = False
        if self.max_tokens < current_word_ct:
            self.lm = dspy.LM(
                self.use_model, api_key=self.openai_api_key, max_tokens=current_word_ct * 2)
            print(
                f'temporarily increasing LM max_tokens to {current_word_ct*2} from {self.max_tokens}')
            needed_to_change_max_tokens = True
        refined_text = dspy.ChainOfThought(Interaction.TextRefinerWithEmotion)(
            text=self.text, primary_emotion=primary_emotion, secondary_emotion=secondary_emotion).refined_text

        if needed_to_change_max_tokens:
            self.onfigure_language_model(self.max_tokens)

        return refined_text

    def create_topic_list(self):
        '''Returns the topics of the text in a list'''
        return dspy.ChainOfThought(Interaction.TopicList)(text=self.text).split(',')

    def create_article_outline(self, topic):
        '''returns an article outline as a list for a given topic while using the text as the main information source'''
        return dspy.ChainOfThought(Interaction.ArticleOutline)(topic=topic, text=self.text).article_outline.split(',')

    def create_article_paragraph(self, topic, section):
        '''Returns a paragraph for a specific section of an article on a particular topic while using the text as the main information source'''
        return dspy.ChainOfThought(Interaction.TopicToParagraph)(topic=topic, text=self.text, section=section).paragraph

    def create_article_first_draft(self, topic):
        '''Returns a rough draft of an article on a specified topic using the text as the main information source'''
        if self.text > '':
            outline = self.create_article_outline(topic=topic)
            paragraphs = []
            article_sectioned = []
            for s in outline:

                section = {'section_name': s}
                section['paragraph'] = self.create_article_paragraph(
                    topic=topic, section=s)

                paragraphs.append(section['paragraph'])
                article_sectioned.append(section)
                article_text = '\n'.join(paragraphs)
            return article_text, article_sectioned, outline
        else:
            print("No text found. Article creation cancelled.")

    def create_article(self, topic, proofread_passes=1):
        '''Creates a proofread article with a title, returning all drafts.
        profread_passes is the number of proofreading iterations you want the system to take.
        '''
        drafts = []
        draft, article_sectioned, outline = self.create_article_first_draft(
            topic=topic)
        dct_draft = {'draft_num': 0, 'draft': draft}
        drafts.append(dct_draft)
        for i in range(proofread_passes):
            refined_article = dspy.ChainOfThought(
                Interaction.ProofReader)(article=draft).proofread_article
            dct_draft = {'draft_num': i+1, 'draft': refined_article}
            drafts.append(dct_draft)
            draft = refined_article

        title = dspy.ChainOfThought(
            Interaction.ArticleTitle)(article=refined_article).title

        complete_article = title + '\n\n' + refined_article

        return complete_article, drafts

    def text_from_transcript_parquet(self, file_path):
        if os.path.exists(file_path):
            df = pd.read_parquet(file_path)
            if 'transcript' in df.columns:
                df.rename(columns={'transcript': 'text'})
            self.text = ' '.join(df['text'].to_list())
        else:
            print("Error: File not found, Loading cancelled.")
