
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os
import datetime
from youtube_transcript_api import YouTubeTranscriptApi
import googleapiclient.discovery
import dspy
from typing import Literal
import spacy
import re
import math


class Keyring:
    '''Holds your API keys'''

    def __init__(self, keys={}):
        self.keys = keys

    def create_key(self, system_name, key):
        self.keys[system_name] = key

    def get_key(self, system_name):
        return self.keys[system_name]

    def add_key_from_path(self, system_name, file_path):
        with open(file_path, 'r') as f:
            self.create_key(system_name=system_name, key=f.readline())

    def copy_key(self, system_name):
        '''copies the specific key and returns the key ring'''
        return {system_name: self.keys[system_name]}


class Scholar:
    '''Reads, writes, and analyzes things'''
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

        class MainTopic(dspy.Signature):
            '''Provide the main topic the text is about in only a few words.'''
            text = dspy.InputField()
            topic = dspy.OutputField()

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

        def cleaned_list(self):
            '''Returns a dataframe of the entities but attempts to refine them by removing shortforms of already listed entities, such as a person's first name when compared to their full name.'''
            df = self.to_dataframe()
            df['len'] = df['value'].astype(str).str.len()
            dfCleaned = pd.DataFrame(columns=['entity_type', 'value'])
            for et in self.entity_types.keys():
                cList = []
                eList = df[df['entity_type'] == et].sort_values(
                    'len', ascending=False)['value'].to_list()
                for e in eList:
                    has_match = False
                    for c in cList:
                        if e in c:
                            has_match = True
                            break

                    if not has_match:
                        cList.append(e)
                        dfCleaned = pd.concat(
                            [dfCleaned, pd.DataFrame([{'entity_type': et, 'value': e}])])

            return dfCleaned

        def to_dataframe(self):
            '''Returns a dataframe containing the unique entity key value pairs'''
            list = self.unique_entities()
            dct = {'entity_type': [], 'value': []}
            for t in list:
                key = [*t][0]
                val = t[key]
                dct['entity_type'].append(key)
                dct['value'].append(val)

            return pd.DataFrame(dct)

    class Scholar:
        '''
        Studies text and provides interpretations. It can utilize NLP techniques via Spacy, and works with ChatGPT api for LLM support.
        '''

        def __init__(self, save_location='./data/', keyring=Keyring(), openai_model='gpt-4o-mini', spacy_model='en_core_web_sm', openai_max_tokens=4096, text=''):
            self.save_location = save_location
            self.keyring = keyring
            self.openai_model = openai_model
            self.openai_max_tokens = openai_max_tokens
            self.spacy_model = spacy_model
            self.lm = None
            self.nlp = None
            self.text = text
            self.entity_list = Scholar.EntityList()
            self.video_id = ''

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

        def extract_entities(self, text):
            if self.nlp is None:
                self.configure_spacy_nlp_model()

            doc = self.nlp(text)
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
                print(
                    'Error: Could not configure an openAI model. Please check your keys')

        def configure_spacy_nlp_model(self, spacy_model='en_core_web_sm'):
            '''Attempts to configure the spacy NLP model'''
            self.spacy_model = spacy_model
            try:
                self.nlp = spacy.load(self.spacy_model)
            except:
                print('Error: Could not configure the spacy NLP model.')

        def create_summary(self):
            '''DSPY: Summarize the text and return the summary and underlying reasoning'''
            summarize = dspy.ChainOfThought('document -> summary')
            response = summarize(document=self.text)
            return response.summary, response.reasoning

        def analyze_main_topic(self):
            '''DSPY: Returns the main topic the text is about'''

            token_count = self.token_count()

            if token_count > 120000:
                print(f'splitting text due to {token_count} tokens')
                text_orig = self.text
                iterations = math.ceil(token_count / 120000)
                doc = self.nlp(text_orig)
                step = token_count / iterations
                topics = []

                for i in range(iterations):
                    token_start = i * step
                    token_stop = token_start + step
                    str_list = [i.text for i in doc[token_start:token_stop]]

                    self.text = ' '.join(str_list)
                    classify = dspy.Predict(Scholar.Interaction.MainTopic)
                    response = classify(text=self.text)
                    topics.append(response.topic)

                self.text = text_orig
                topicsU = list(set(topics))
                return ' and '.join(topicsU)

            else:
                classify = dspy.Predict(Scholar.Interaction.MainTopic)
                response = classify(text=self.text)
                return response.topic

        def analyze_sentiment(self):
            '''DSPY: Returns True for positive and False for negative sentiment of the text'''
            classify = dspy.Predict('sentence -> sentiment: bool')
            response = classify(sentence=self.text)
            return response.sentiment

        def analyze_emotion(self):
            '''DSPY: Returns the emotion most evoked by the text'''
            classify = dspy.Predict(Scholar.Interaction.Emotion)
            response = classify(text=self.text)
            return response.sentiment

        def analyze_secondary_emotion(self):
            '''DSPY: Returns the secondary emotion most evoked by the text'''
            classify = dspy.Predict(Scholar.Interaction.EmotionSecondary)
            response = classify(text=self.text)
            return response.sentiment

        def token_count(self):
            doc = self.nlp.make_doc(self.text)
            return len(doc)

        def fetch_overall_classification(self):
            '''DSPY: runs the procedures for summary, sentiment, primary and secondary emotions.
            If the text is larger than 120000 tokens, it will evenly divid it and run classification on all parts, returning a comma separated list for each one.
            '''

            token_ct = self.token_count()

            if token_ct < 120000:
                print('classifying: ' + self.video_id)
                summary, reasoning = self.create_summary()
                sentiment = self.analyze_sentiment()
                primary_emotion = self.analyze_emotion()
                secondary_emotion = self.analyze_secondary_emotion()
                main_topic = self.analyze_main_topic()
                metadata = {'summary': summary, 'summary_reasoning': reasoning, 'sentiment': sentiment,
                            'primary_emotion': primary_emotion, 'secondary_emotion': secondary_emotion, 'main_topic': main_topic}
                return metadata
            else:
                summaries = []
                reasonings = []
                sentiments = []
                primary_emotions = []
                secondary_emotions = []
                main_topics = []

                text_orig = self.text
                token_ct = self.token_count()
                iterations = math.ceil(token_ct / 120000)
                doc = self.nlp(text_orig)
                step = token_ct / iterations

                print('classifying: ' + self.video_id +
                      " in " + str(iterations) + 'parts.')

                for i in range(iterations):
                    token_start = i * step
                    token_stop = token_start + step
                    str_list = [i.text for i in doc[token_start:token_stop]]

                    self.text = ' '.join(str_list)
                    summary, reasoning = self.create_summary()
                    sentiment = self.analyze_sentiment()
                    primary_emotion = self.analyze_emotion()
                    secondary_emotion = self.analyze_secondary_emotion()
                    main_topic = self.analyze_main_topic()

                    summaries.append(summary)
                    reasonings.append(reasoning)
                    sentiments.append(sentiment)
                    primary_emotions.append(primary_emotion)
                    secondary_emotions.append(secondary_emotion)
                    main_topics.append(main_topic)

                self.text = text_orig
                metadata = {'summary': ','.join(summaries), 'summary_reasoning': ','.join(reasonings), 'primary_emotion': ','.join(primary_emotions), 'seconary_emotion': ','.join(secondary_emotions), 'main_topic': ','.join(main_topic)
                            }
                return metadata

        def lookup_video_info(self, video_id):
            '''returns the channel_id, channel_title, and channel category for a video_id'''
            channel_id = ''
            channel_title = ''
            channel_category = ''
            published_at = ''
            title = ''

            for file in os.listdir('./data/channels/'):
                df = pd.read_parquet('./data/channels/' + file)
                video_ids = df['video_id'].to_list()
                if video_id in video_ids:
                    channel_id = file.replace('_search.parquet', '')
                    published_at = df[df['video_id'] ==
                                      video_id]['published_at'].iloc[0]
                    title = df[df['video_id'] == video_id]['title'].iloc[0]
                    break

            if channel_id != '':
                dfc = pd.read_csv(
                    './data/youtube_channels_index.csv', encoding='utf-8')
                dfs = dfc[dfc['channel_id'] == channel_id]
                if dfs.shape[0] > 0:
                    channel_title = dfs['channel_title'].iloc[0]
                    channel_category = dfs['category'].iloc[0]

            dct = {'channel_id': channel_id, 'channel_title': channel_title,
                   'channel_category': channel_category, 'published_at': published_at, 'title': title}

            return dct

        def create_overall_classification(self):
            '''Loads or creates classifications to return metadata'''

            file_path = f'./data/classifications/{self.video_id}_classification.parquet'
            metadata = {}
            video_info = {}

            columns = ['summary', 'summary_reasoning', 'sentiment',
                       'primary_emotion', 'secondary_emotion', 'main_topic']

            if self.video_id != '':
                video_info = self.lookup_video_info(self.video_id)

                if os.path.exists(file_path):
                    dforig = pd.read_parquet(file_path)

                    df = pd.DataFrame(columns=columns)
                    df = pd.concat([df, dforig])

                    summary = df['summary'][0]
                    reasoning = df['summary_reasoning'][0]
                    sentiment = df['sentiment'][0]
                    primary_emotion = df['primary_emotion'][0]
                    secondary_emotion = df['secondary_emotion'][0]
                    main_topic = df['main_topic'][0]

                    metadata = {'summary': summary, 'summary_reasoning': reasoning, 'sentiment': sentiment,
                                'primary_emotion': primary_emotion, 'secondary_emotion': secondary_emotion, 'main_topic': main_topic}
                else:
                    metadata = self.fetch_overall_classification()
                    df = pd.DataFrame([metadata])
                    table = pa.Table.from_pandas(df)
                    pq.write_table(table, file_path,
                                   use_dictionary=True, compression='gzip')
            else:
                video_info = {'channel_id': '', 'channel_title': '',
                              'channel_category': '', 'title': '', 'published_at': ''}
                metadata = self.fetch_overall_classification()

            metadata['channel_id'] = video_info['channel_id']
            metadata['channel_title'] = video_info['channel_title']
            metadata['channel_category'] = video_info['channel_category']
            metadata['title'] = video_info['title']
            metadata['published_at'] = video_info['published_at']

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
            refined_text = dspy.ChainOfThought(Scholar.Interaction.TextRefinerWithEmotion)(
                text=self.text, primary_emotion=primary_emotion, secondary_emotion=secondary_emotion).refined_text

            if needed_to_change_max_tokens:
                self.onfigure_language_model(self.max_tokens)

            return refined_text

        def create_topic_list(self):
            '''Returns the topics of the text in a list'''
            token_ct = self.token_count()

            if token_ct < 120000:
                return dspy.ChainOfThought(Scholar.Interaction.TopicList)(text=self.text).topics.split(',')
            else:
                text_orig = self.text
                token_ct = self.token_count()
                iterations = math.ceil(token_ct / 120000)
                doc = self.nlp(text_orig)
                step = token_ct / iterations
                all_topics = []

                for i in range(iterations):
                    token_start = i * step
                    token_stop = token_start + step
                    str_list = [i.text for i in doc[token_start:token_stop]]

                    self.text = ' '.join(str_list)
                    topics = dspy.ChainOfThought(Scholar.Interaction.TopicList)(
                        text=self.text).topics.split(',')
                    all_topics.extend(topics)

                self.text = text_orig
                all_topics = list(set(all_topics))
                return all_topics

        def create_article_outline(self, topic):
            '''returns an article outline as a list for a given topic while using the text as the main information source'''
            return dspy.ChainOfThought(Scholar.Interaction.ArticleOutline)(topic=topic, text=self.text).article_outline.split(',')

        def create_article_paragraph(self, topic, section):
            '''Returns a paragraph for a specific section of an article on a particular topic while using the text as the main information source'''
            return dspy.ChainOfThought(Scholar.Interaction.TopicToParagraph)(topic=topic, text=self.text, section=section).paragraph

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
                    Scholar.Interaction.ProofReader)(article=draft).proofread_article
                dct_draft = {'draft_num': i+1, 'draft': refined_article}
                drafts.append(dct_draft)
                draft = refined_article

            title = dspy.ChainOfThought(
                Scholar.Interaction.ArticleTitle)(article=refined_article).title

            complete_article = title + '\n\n' + refined_article

            return complete_article, drafts

        def text_from_transcript_parquet(self, file_path):
            if os.path.exists(file_path):
                head, tail = os.path.split(
                    file_path)

                self.video_id = tail.replace('_transcript.parquet', '')

                df = pd.read_parquet(file_path)
                if 'transcript' in df.columns:
                    df.rename(columns={'transcript': 'text'}, inplace=True)
                self.text = ' '.join(df['text'].to_list())
            else:
                print("Error: File not found, Loading cancelled.")


class Librarian:
    '''Organizes your things and can collect things from your library'''
    class Classifer:
        def __init__(self, dimensions_folder):
            self.dim_emotions = pd.read_csv(
                dimensions_folder + 'dim_emotions.csv')

            self.document_types = {'article': 1,
                                   'transcript': 2, 'youtube channel search': 3}
            self.source_systems = {'user': 1, 'youtube': 2, 'scholar': 3}

            self.classification_columns = [
                'summary', 'summary_reasoning', 'sentiment', 'primary_emotion', 'secondary_emotion']

    class Librarian:

        def __init__(self, library_path='./', keyring=Keyring(), refresh_index=True):
            self.keyring = Keyring()

            if not library_path.endswith('/'):
                library_path += '/'

            self.library_path = library_path
            self.check_for_folder(self.library_path)
            self.check_for_folder(self.transcripts_folder())
            self.check_for_folder(self.classifications_folder())
            self.check_for_folder(self.dimensions_folder())

            if not os.path.exists(self.library_index_path()):
                self.create_document_store()

            if not os.path.exists(self.entity_index_path()):
                self.compile_entity_index()

            self.channelIndex = pd.read_csv(
                self.channels_index_path(), encoding='utf8')

            self.classifier = Librarian.Classifer(self.dimensions_folder())
            self.libraryIndex = pd.read_parquet(self.library_index_path())
            self.libraryIndex.set_index('lib_key')

            if not os.path.exists(self.transcript_skip_file_path()):
                self.create_video_skip_file()

            if refresh_index:
                self.refresh_index()

        def channels_list(self, category=''):
            dfc = pd.read_csv(self.channels_index_path())
            if category > '':
                dfc = dfc[dfc['category'].str.lower() == category.lower()]

            return dfc['channel_id'].to_list()

        def create_video_skip_file(self):
            df = pd.DataFrame({'video_id': []})
            table = pa.Table.from_pandas(df)
            pq.write_table(table, self.transcript_skip_file_path(),
                           use_dictionary=True, compression='gzip')

        def create_document_store(self):
            df = pd.DataFrame(columns=[
                'lib_key', 'source_id', 'rec_mod_dtm', 'doc_type_id',  'source_system_id', 'path', 'summary', 'summary_reasoning', 'sentiment', 'primary_emotion', 'secondary_emotion'])
            table = pa.Table.from_pandas(df)
            pq.write_table(table, self.library_path + 'library_index.parquet',
                           use_dictionary=True, compression='gzip')

        def check_for_folder(self, path):
            if not os.path.exists(path):
                os.makedirs(path)

        def library_index_path(self):
            return self.library_path + 'data/library_index.parquet'

        def transcript_skip_file_path(self):
            return self.library_path + 'data/transcript_skip.parquet'

        def channels_index_path(self):
            return self.library_path + 'data/youtube_channels_index.csv'

        def entity_index_path(self):
            return self.library_path + 'data/entity_index.parquet'

        def channels_folder(self):
            return self.library_path + 'data/channels/'

        def entities_folder(self):
            return self.library_path + 'data/entities/'

        def transcripts_folder(self):
            return self.library_path + 'data/transcripts/'

        def classifications_folder(self):
            return self.library_path + 'data/classifications/'

        def dimensions_folder(self):
            return self.library_path + 'data/dimensons/'

        def upsert_library_index(self, source_id, rec_mod_dtm, doc_type_id, source_system_id, path, summary='', summary_reasoning='', sentiment='', primary_emotion='', secondary_emotion='', main_topic=''):

            rows, columns = self.libraryIndex.shape
            new_lib_key = rows + 1
            dfn = pd.DataFrame([{'lib_key': new_lib_key, 'source_id': source_id, 'rec_mod_dtm': rec_mod_dtm, 'doc_type_id': doc_type_id, 'source_system_id': source_system_id, 'path': path,
                               'summary': summary, 'summary_reasoning': summary_reasoning, 'sentiment': sentiment, 'primary_emotion': primary_emotion, 'secondary_emotion': secondary_emotion, 'main_topic': main_topic}])

            # Does the entry for the file already exist? If not, then insert it
            if not path in self.libraryIndex['path'].to_list():
                self.libraryIndex = pd.concat([self.libraryIndex, dfn])
                return True

            # The file exists. Check record modification date and sentiment for changes
            cur_rec = self.libraryIndex[self.libraryIndex['path']
                                        == path].iloc[0]

            if (cur_rec['rec_mod_dtm'] != rec_mod_dtm) or (cur_rec['sentiment'] != sentiment) or (cur_rec['primary_emotion'] != primary_emotion) or (cur_rec['main_topic'] != main_topic):
                # There is a difference, so replace the record
                lib_key = cur_rec['lib_key']
                dfn['lib_key'] = lib_key
                self.libraryIndex = self.libraryIndex[self.libraryIndex['path'] != path]
                self.libraryIndex = pd.concat([self.libraryIndex, dfn])
                return True

            return False

        def rewrite_library_index(self):
            table = pa.Table.from_pandas(self.libraryIndex)
            pq.write_table(table, self.library_index_path(),
                           use_dictionary=True, compression='gzip')

        def fetch_classification(self, source_id):
            file_path = self.classifications_folder() + source_id + \
                '_classification.parquet'

            df = pd.DataFrame()
            if os.path.exists(file_path):
                df = pd.read_parquet(file_path)

            else:
                df = pd.DataFrame([{'summary': str(None), 'summary_reasoning': str(None),
                                  'sentiment': bool(None), 'primary_emotion': str(None), 'secondary_emotion': str(None), 'main_topic': str(None)}])

            for c in self.classifier.classification_columns:
                if not c in df.columns:
                    if c == 'sentiment':
                        df[c] = bool(None)
                    else:
                        df[c] = str(None)

            df['sentiment'] = df['sentiment'].astype(bool)

            return df

        def fetch_channel_info(self, video_id_list):
            df = pd.DataFrame({'video_id': video_id_list})
            df['channel_id'] = df['video_id'].apply(
                lambda x: self.fetch_channel(x))
            dfchan = pd.read_csv(self.channels_index_path(), encoding='utf8')
            df = df.join(dfchan.set_index('channel_id'), on='channel_id')
            return df

        def fetch_channel(self, video_id):
            for file in os.listdir(self.channels_folder()):
                dfsearch = pd.read_parquet(self.channels_folder() + file)
                channel_id = file.replace('_search.parquet', '')
                if video_id in dfsearch['video_id'].to_list():
                    return channel_id

            return ''

        def check_folder_for_new_files(self, doc_type, source_system, folder, source_id_replacement):
            any_changes = False
            doc_type_id = self.classifier.document_types[doc_type]
            source_system_id = self.classifier.source_systems[source_system]

            for file in os.listdir(folder):
                path = folder + file
                source_id = file.replace(source_id_replacement, '')

                rec_mod_dtm = datetime.datetime.fromtimestamp(
                    os.path.getctime(path)).strftime('%Y-%m-%d %H:%M:%S')

                clf = self.fetch_classification(
                    source_id=source_id)

                if 'main_topic' not in clf.columns:
                    clf['main_topic'] = ''

                summary = clf['summary'].iloc[0]
                summary_reasoning = clf['summary_reasoning'].iloc[0]
                sentiment = clf['sentiment'].iloc[0]
                primary_emotion = clf['primary_emotion'].iloc[0]
                secondary_emotion = clf['secondary_emotion'].iloc[0]
                main_topic = clf['main_topic'].iloc[0]

                new_items = self.upsert_library_index(source_id, rec_mod_dtm, doc_type_id, source_system_id,
                                                      path, summary, summary_reasoning, sentiment, primary_emotion, secondary_emotion, main_topic)

                if new_items:
                    any_changes = True

            return any_changes

        def refresh_index(self):
            any_changes = False

            if self.check_folder_for_new_files(doc_type='youtube channel search', source_system='youtube', folder=self.channels_folder(), source_id_replacement='_search.parquet'):
                any_changes = True

            if self.check_folder_for_new_files(doc_type='transcript', source_system='youtube', folder=self.transcripts_folder(), source_id_replacement='_transcript.parquet'):
                any_changes = True

            if any_changes:
                self.rewrite_library_index()

        def videos_containing_entity(self, entity_name, entity_type='any'):
            '''returns a list of video_id where the entity was found.'''
            video_ids = []
            entity_name = entity_name.lower()

            for entity_list in os.listdir(self.entities_folder()):
                dfe = pd.read_parquet(self.entities_folder() + entity_list)
                print(entity_list)
                video_id = entity_list.replace('_entities.parquet', '')

                if entity_type != 'any':
                    dfe = dfe[dfe['entity_type'] == entity_type]

                dfe['value'] = dfe['value'].astype(str).str.lower()
                l = dfe['value'].to_list()
                if entity_name in l:
                    video_ids.append(video_id)

            return video_ids

        def topic_index_path(self):
            return self.library_path + 'data/topic_index.parquet'

        def compile_entity_index(self):
            dfe = pd.DataFrame()

            for file in os.listdir(self.entities_folder()):
                source_id = file.replace('_entities.parquet', '')
                df = pd.read_parquet(self.entities_folder() + file)
                df['source_id'] = source_id
                dfe = pd.concat([dfe, df])

            table = pa.Table.from_pandas(dfe)
            pq.write_table(table, self.entity_index_path(),
                           use_dictionary=True, compression='gzip')

        def compile_topic_index(self):
            dft = pd.DataFrame()

            for file in os.listdir(self.topics_folder()):
                source_id = file.replace('_topics.parquet', '')
                df = pd.read_parquet(self.topics_folder() + file)
                dft = pd.concat([dft, df])

            table = pa.Table.from_pandas(dft)
            pq.write_table(table, self.topic_index_path(),
                           use_dictionary=True, compression='gzip')

        def collect_data_containing_entity(self, entity):
            '''Searches all documents containing the given entity and returns classification and index data'''
            entity = entity.lower()
            dfe = pd.read_parquet('./data/entity_index.parquet')
            dfe['value'] = dfe['value'].str.lower()
            dfl = pd.read_parquet('./data/library_index.parquet')

            source_ids = dfe[dfe['value'].str.contains(
                entity)]['source_id'].unique()
            dfs = dfl[dfl['source_id'].isin(source_ids)]
            dfc = self.fetch_channel_info(dfs['source_id'].unique())
            dfs = dfs.join(dfc.set_index('video_id'), on='source_id')
            dfs['entity'] = entity
            return dfs

        def collect_multi_entity_data(self, entities_list=[]):
            '''Given a list of entities to search for, this procedure returns a dataframe containing all matches with classification and index data'''
            df = pd.DataFrame()
            for entity in entities_list:
                dfe = self.collect_data_containing_entity(entity)
                df = pd.concat([dfe, df])
            return df

        def topics_folder(self):
            return self.library_path + 'data/topics/'

        def video_analysis_data_path(self):
            return self.library_path + 'data/analysis/video_analysis_data.parquet'

        def entity_analysis_data_path(self):
            return self.library_path + 'data/analysis/entity_analysis_data.parquet'

        def topic_analysis_data_path(self):
            return self.library_path + 'data/analysis/topic_analysis_data.parquet'

        def convert_to_date(self, dtm):
            return dtm.date()

        def compile_analysis_files(self):
            '''
            Combines video search, channel metadata, and video metadata into the video_analysis_data.parquet file
            Combines video search, channel metadata, video metadata, and entity data into the entity_analysis_data.parquet file
            '''
            dfChannels = pd.read_csv(
                self.channels_index_path(), encoding='utf-8')
            channel_ids = dfChannels['channel_id'].to_list()
            dfci = pd.DataFrame()
            for channel_id in channel_ids:
                file_path = self.channels_folder(
                ) + f'{channel_id}_search.parquet'
                if os.path.exists(file_path):
                    dfs = pd.read_parquet(file_path)
                    dfci = pd.concat([dfci, dfs])

            dfci = dfci.join(dfChannels.set_index(
                'channel_id'), on='channel_id')
            dfci.drop(axis=1, columns=['channel_id', 'code'], inplace=True)

            dfli = pd.read_parquet(self.library_index_path())
            dfli.drop(axis=1, columns=[
                      'lib_key', 'doc_type_id', 'source_system_id', 'path', 'summary_reasoning'], inplace=True)

            dfci = dfci.join(dfli.set_index('source_id'),
                             on='video_id', how='inner')

            dfci['published_at'] = pd.to_datetime(
                dfci['published_at'], utc=True)
            dfci['publish_dt_et'] = pd.to_datetime(
                dfci['published_at'], utc=True).dt.tz_convert('US/Eastern')

            dfci['publish_dt_et'] = dfci['publish_dt_et'].apply(
                lambda x: self.convert_to_date(x))

            table = pa.Table.from_pandas(dfci)
            pq.write_table(table, self.video_analysis_data_path(),
                           use_dictionary=True, compression='gzip')

            dfei = pd.read_parquet('./data/entity_index.parquet')

            dfei = dfei.join(dfci.set_index('video_id'),
                             on='source_id', how='inner')
            table = pa.Table.from_pandas(dfei)
            pq.write_table(table, self.entity_analysis_data_path(),
                           use_dictionary=True, compression='gzip')

            dft = pd.read_parquet(self.topic_index_path())
            dfti = dft.join(dfci.set_index('video_id'),
                            on='source_id', how='inner')

            table = pa.Table.from_pandas(dfti)
            pq.write_table(table, self.topic_analysis_data_path(),
                           use_dictionary=True, compression='gzip')


class YouTubeExplorer:
    '''Searches YouTube and fetches transcripts'''
    class SearchQuery:
        def __init__(self, search_for, part, channel_id, max_results, published_after, region_cd, relevance_language, safe_search, order, video_duration):
            self.search_for = search_for
            self.part = part
            self.channel_id = channel_id
            self.max_results = max_results
            self.published_after = published_after
            self.region_cd = region_cd
            self.relevance_language = relevance_language
            self.safe_search = safe_search
            self.order = order
            self.video_duration = video_duration

            self.search_response = None

    class SearchItem:
        '''Result item from a YouTube search'''

        def __init__(self, raw_item):
            self.kind = raw_item['kind']
            self.etag = raw_item['etag']
            self.id = raw_item['id']
            self.snippet = raw_item['snippet']

            self.is_video = False
            if self.id['kind'] == 'youtube#video':
                self.is_video = True
                self.video_id = self.id['videoId']
            else:
                self.video_id = None

            if not self.snippet is None and not self.snippet == {}:
                self.published_date = raw_item['snippet']['publishedAt']
                self.channel_id = raw_item['snippet']['channelId']
                self.title = self.clean_text(raw_item['snippet']['title'])
                self.description = self.clean_text(
                    raw_item['snippet']['description'])
                self.thumbnails = raw_item['snippet']['thumbnails']
                self.channel_title = self.clean_text(
                    raw_item['snippet']['channelTitle'])
                self.live_broadcast_content = raw_item['snippet']['liveBroadcastContent']
                self.publish_time = raw_item['snippet']['publishTime']
            else:
                self.published_date = None
                self.channel_id = None
                self.title = None
                self.description = None
                self.thumbnails = None
                self.channel_title = None
                self.live_broadcast_content = None
                self.publish_time = None

        def clean_text(self, text):
            return text.replace('&#39;', "'").replace('&amp;', '&')

    class SearchResponse:
        '''A YouTube search resonse, converted from the original JSON'''

        def __init__(self, raw_response):
            self.kind = raw_response['kind']
            self.etag = raw_response['etag']
            self.region_code = raw_response['regionCode']
            self.raw_response = raw_response
            self.items = []

            self.build_items()

        def build_items(self):
            '''Builds the items list'''
            self.items = []
            for item in self.raw_response['items']:
                yti = YouTubeExplorer.SearchItem(item)
                self.items.append(yti)

    class Explorer:
        def __init__(self, data_folder='./data/', keyring=Keyring(), api_service_name='youtube', api_version='v3'):
            self.keyring = keyring
            self.data_folder = data_folder
            self.youtube_api_client = None
            self.api_service_name = api_service_name
            self.api_version = api_version
            self.searches = []
            self.transcript_skip = []

            self.load_transcrtipt_skip_list()

            try:
                self.configure_youtube_client()
            except:
                print("No youtube API key found, client is not configured.")

            if not os.path.exists(self.transcripts_folder()):
                os.makedirs(self.transcripts_folder())

            if not os.path.exists(self.channels_folder()):
                os.makedirs(self.channels_folder())

        def load_transcrtipt_skip_list(self):
            self.transcript_skip = []
            if os.path.exists(self.data_folder + 'transcript_skip.parquet'):
                df = pd.read_parquet(self.data_folder +
                                     'transcript_skip.parquet')
                self.transcript_skip = list(set(df['video_id'].to_list()))

        def channels_folder(self):
            return self.data_folder + 'channels/'

        def transcripts_folder(self):
            return self.data_folder + 'transcripts/'

        def configure_youtube_client(self):
            '''Configures the YouTube API client for search, using the API key on the keyring'''
            self.youtube_api_client = googleapiclient.discovery.build(
                self.api_service_name, self.api_version, developerKey=self.keyring.get_key('youtube'))

        def channel_video_list_page(self, channel_id, page_token=''):
            '''Returns the page results for a channel search'''
            if page_token > '':
                r = self.youtube_api_client.search().list(part='snippet', channelId=channel_id,
                                                          maxResults=50, order='date', pageToken=page_token)
            else:
                r = self.youtube_api_client.search().list(
                    part='snippet', channelId=channel_id, maxResults=50, order='date')

            snip = r.execute()
            results = {}
            results['nextPageToken'] = snip['nextPageToken']
            results['items'] = []

            for i in snip['items']:
                if i['id']['kind'] == 'youtube#video':
                    itm = {'video_id': i['id']['videoId']}
                    itm['title'] = i['snippet']['title']
                    itm['published_at'] = i['snippet']['publishedAt']
                    results['items'].append(itm)

            return results

        def channel_video_list(self, channel_id, max_pages=1):
            '''Searches a channel in reverse chronological order, returns the entire history of videos and some metadata in a pandas dataframe'''
            print(
                f'Searching for: {channel_id} max_pages: {max_pages}', end='... ')
            # Loop through the desired number of page results to load the snippets array
            snip = None
            snippets = []
            page_token = ''
            for i in range(max_pages):
                snip = self.channel_video_list_page(
                    channel_id=channel_id, page_token=page_token)
                page_token = snip['nextPageToken']
                snippets.append(snip)

            # Combine all the results into a dataframe
            df = pd.DataFrame()
            for i in snippets:
                dfs = pd.DataFrame(i['items'])
                df = pd.concat([df, dfs])

            df['channel_id'] = channel_id

            # check to see if there is a channel file already. If so, load it and append. Otherwise create it.
            dfc = pd.DataFrame(columns=['video_id'])
            channels_file_path = self.channels_folder() + channel_id + '_search.parquet'

            if os.path.exists(channels_file_path):
                dfc = pd.read_parquet(channels_file_path)
                df = df[~df['video_id'].isin(dfc['video_id'].to_list())]

            df = pd.concat([df, dfc])
            df.drop_duplicates(inplace=True)
            table = pa.Table.from_pandas(df)
            pq.write_table(table, channels_file_path,
                           use_dictionary=True, compression='gzip')

            print('done.')
            return df

        def video_search(self, search_for='', channel_id='', max_results=25, published_after='1970-01-01T00:00:00.000Z', region_cd='ca', relevance_language='en', safe_search='none', order='date', video_duration='medium'):
            '''Searches YouTube, returning the search query and a custom object for the response'''

            part = 'snippet'

            qry = YouTubeExplorer.SearchQuery(search_for=search_for, part=part, channel_id=channel_id, max_results=max_results, published_after=published_after,
                                              region_cd=region_cd, relevance_language=relevance_language, safe_search=safe_search, order=order, video_duration=video_duration)

            if channel_id > '':
                r = self.youtube_api_client.search().list(part=part, channelId=channel_id, maxResults=max_results, publishedAfter=published_after, q=search_for, regionCode=region_cd,
                                                          relevanceLanguage=relevance_language, safeSearch=safe_search, type='video', videoCaption='closedCaption', order=order, videoDuration=video_duration)
            else:
                r = self.youtube_api_client.search().list(
                    part=part, maxResults=max_results, publishedAfter=published_after, q=search_for, regionCode=region_cd, safeSearch=safe_search, relevanceLanguage=relevance_language, type='video', videoCaption='closedCaption', order=order, videoDuration=video_duration
                )

            qry.search_response = YouTubeExplorer.SearchResponse(r.execute())
            self.searches.append(qry)

            return qry

        def fetch_transcript(self, video_id):
            '''Checks to see if a transcript has already been saved, fetching it if not. Returns a dictionary of the transcript'''
            print(f'fetching transcript for: {video_id}', end='...')

            if video_id in self.transcript_skip:
                print('in skip list. cancelled')
                return False

            file_path = self.transcripts_folder() + video_id + '_transcript.parquet'

            if os.path.exists(file_path):
                print('found.')
                return True

            try:
                captions = YouTubeTranscriptApi.get_transcript(
                    video_id=video_id)
                if captions != None:
                    df = pd.DataFrame(captions)
                    table = pa.Table.from_pandas(df)
                    pq.write_table(table, file_path,
                                   use_dictionary=True, compression='gzip')
                print('downloaded.')
                return True
            except:
                self.transcript_skip.append(video_id)
                self.save_transcript_skip_list()
                print('unable to download.')
                return False

        def save_transcript_skip_list(self):
            df = pd.DataFrame({'video_id': self.transcript_skip})
            table = pa.Table.from_pandas(df)
            pq.write_table(table, self.data_folder + 'transcript_skip.parquet',
                           use_dictionary=True, compression='gzip')

        def fetch_transcripts_for_channel(self, channel_id):
            '''Uses the channel search history file and fetches any missing transcripts'''
            channel_file_path = self.channels_folder() + channel_id + '_search.parquet'
            dfc = pd.read_parquet(channel_file_path)
            dfc['code'] = dfc['video_id'].apply(
                lambda x: self.fetch_transcript(video_id=x))
            dfc = dfc[dfc['code'] == True]
            table = pa.Table.from_pandas(dfc)
            pq.write_table(table, channel_file_path,
                           use_dictionary=True, compression='gzip')


class Watchmen:
    '''This is a team of the other classes, so you can coordinate their efforts.'''

    def __init__(self, youtube_api_key_path='', openai_api_key_path='', spacy_model='en_core_web_sm', refresh_indexes=True, include_youtube_explorer=True, include_scholar=True):
        self.master_keyring = Keyring()

        print('setting up librarian...')
        self.librarian = Librarian.Librarian(refresh_index=refresh_indexes)

        if include_youtube_explorer:
            print('setting up youtube explorer.')
            # setup keys
            if youtube_api_key_path > '':
                self.master_keyring.add_key_from_path(
                    system_name='youtube', file_path=youtube_api_key_path)
                self.yte_explorer = YouTubeExplorer.Explorer(keyring=Keyring(
                    self.master_keyring.copy_key('youtube')))

            else:
                self.yte_explorer = YouTubeExplorer.Explorer()
        else:
            print('Skipping YouTube Explorer setup')

        if include_scholar:
            print('setting up scholar...')
            if openai_api_key_path > '':
                self.master_keyring.add_key_from_path(
                    system_name='openai', file_path=openai_api_key_path)

                self.scholar = Scholar.Scholar(keyring=Keyring(
                    self.master_keyring.copy_key('openai')), spacy_model=spacy_model)

            else:
                self.scholar = Scholar.Scholar()
        else:
            print('Skipping Scholar setup.')

    def identify_topics_for_all(self):
        '''Generates topic files for all transcripts that don't have one'''
        print('Identifing topics for all transcripts.')
        new_file_ct = 0
        for file in os.listdir(self.librarian.transcripts_folder()):
            source_id = file.replace('_transcript.parquet', '')
            topics_file_path = self.librarian.topics_folder() + source_id + \
                '_topics.parquet'

            if not os.path.exists(topics_file_path):
                print(f'Building topics for {source_id}')
                new_file_ct += 1
                self.scholar.text_from_transcript_parquet(
                    self.librarian.transcripts_folder() + file)
                topics_list = self.scholar.create_topic_list()
                df = pd.DataFrame(
                    {'topic': topics_list, 'source_id': source_id})
                table = pa.Table.from_pandas(df)
                pq.write_table(table, topics_file_path,
                               use_dictionary=True, compression='gzip')
        print(f'Topics built for {new_file_ct} transcripts.')

    def update_new_classifications(self):
        '''Checks to see if any classifications are missing. You should update this procedure if you add new classifications'''
        print('Checking existing classifications')
        chan_list = self.librarian.channels_list()
        new_file_ct = 0
        columns = ['summary', 'summary_reasoning', 'sentiment',
                   'primary_emotion', 'secondary_emotion', 'main_topic']
        for chan in chan_list:
            df = pd.read_parquet(
                self.librarian.channels_folder() + chan + '_search.parquet')
            video_ids = df['video_id'].to_list()
            for video_id in video_ids:
                classification_file = self.librarian.classifications_folder() + video_id + \
                    '_classification.parquet'
                if os.path.exists(classification_file):
                    dfc = pd.read_parquet(classification_file)
                    if 'main_topic' not in dfc.columns:
                        dfc['main_topic'] = ''

                    if dfc['main_topic'][0] == '':
                        # main topic was added - make sure the file has that
                        new_file_ct += 1
                        print(f'rewriting {classification_file}')
                        self.scholar.text_from_transcript_parquet(
                            self.librarian.transcripts_folder() + video_id + "_transcript.parquet")
                        main_topic = self.scholar.analyze_main_topic()
                        dfc['main_topic'] = main_topic
                        table = pa.Table.from_pandas(dfc)
                        pq.write_table(
                            table, classification_file, use_dictionary=True, compression='gzip')

        print(f'{new_file_ct} main topics added.')

    def classify_transcripts(self, category=''):
        '''Uses AI to classify all transcripts. Providing a category is optional'''
        print('Classifying transcripts.')
        chan_list = self.librarian.channels_list(category=category)
        new_file_ct = 0
        for chan in chan_list:
            df = pd.read_parquet(
                self.librarian.channels_folder() + chan + '_search.parquet')
            video_ids = df['video_id'].to_list()
            for video_id in video_ids:
                classification_file = self.librarian.classifications_folder() + video_id + \
                    '_classification.parquet'
                if not os.path.exists(classification_file):
                    new_file_ct += 1
                    self.scholar.text_from_transcript_parquet(
                        self.librarian.transcripts_folder() + video_id + "_transcript.parquet")
                    self.scholar.create_overall_classification()
        print(f'{new_file_ct} transcripts classified.')

    def entity_file_path(self, video_id):
        return './data/entities/' + video_id + '_entities.parquet'

    def build_entities_file(self, video_id, use_summary=True, recompile_entities_index=True):
        '''Reads in a transcript and builds a parquet file containing the unique entity list'''
        print(f'building entity file for {video_id}')

        self.scholar.video_id = video_id

        if use_summary:
            classification = self.scholar.create_overall_classification()
            text = classification['summary']
        else:
            transcript_path = './data/transcripts/' + video_id + '_transcript.parquet'
            self.scholar.text_from_transcript_parquet(transcript_path)
            text = self.scholar.text

        self.scholar.extract_entities(text)
        df = self.scholar.entity_list.cleaned_list()
        table = pa.Table.from_pandas(df)
        pq.write_table(table, self.entity_file_path(video_id), use_dictionary=True,
                       compression='gzip')
        if recompile_entities_index:
            self.librarian.compile_entity_index()

    def build_entities_for_all(self):
        '''Builds entity files for all transcripts. Skips building if an entity file already exists for a particular video'''
        print('Building entities for all transcripts...')
        new_files_ct = 0
        for transcript_file in os.listdir('./data/transcripts'):
            video_id = transcript_file.replace('_transcript.parquet', '')
            if not os.path.exists(self.entity_file_path(video_id)):
                new_files_ct += 1
                self.build_entities_file(
                    video_id, recompile_entities_index=False)
        self.librarian.compile_entity_index()
        print(f'Entities built for {new_files_ct} transcripts.')

    def fetch_new_transcripts_for_channels(self, category='', max_pages=1):
        '''Fetches any new transcripts for channels matching the category provided. Blank seeks all categories.'''
        chan_list = self.librarian.channels_list(category=category)
        print(f'{len(chan_list)} channels found.')
        for chan in chan_list:
            print(f'Updating: {chan}')
            self.yte_explorer.channel_video_list(
                channel_id=chan, max_pages=max_pages)
            self.yte_explorer.fetch_transcripts_for_channel(channel_id=chan)
