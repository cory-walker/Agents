from youtube_transcript_api import YouTubeTranscriptApi
import googleapiclient.discovery
import Keyring
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os


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
            yti = SearchItem(item)
            self.items.append(yti)


class Explorer:
    def __init__(self, data_folder='./data/', keyring=Keyring.Keyring(), api_service_name='youtube', api_version='v3'):
        self.keyring = keyring
        self.data_folder = data_folder
        self.youtube_api_client = None
        self.api_service_name = api_service_name
        self.api_version = api_version
        self.searches = []

        try:
            self.configure_youtube_client()
        except:
            print("No youtube API key found, client is not configured.")

        if not os.path.exists(self.transcripts_folder()):
            os.makedirs(self.transcripts_folder())

        if not os.path.exists(self.channels_folder()):
            os.makedirs(self.channels_folder())

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

        return df

    def video_search(self, search_for='', channel_id='', max_results=25, published_after='1970-01-01T00:00:00.000Z', region_cd='ca', relevance_language='en', safe_search='none', order='date', video_duration='medium'):
        '''Searches YouTube, returning the search query and a custom object for the response'''

        part = 'snippet'

        qry = SearchQuery(search_for=search_for, part=part, channel_id=channel_id, max_results=max_results, published_after=published_after,
                          region_cd=region_cd, relevance_language=relevance_language, safe_search=safe_search, order=order, video_duration=video_duration)

        if channel_id > '':
            r = self.youtube_api_client.search().list(part=part, channelId=channel_id, maxResults=max_results, publishedAfter=published_after, q=search_for, regionCode=region_cd,
                                                      relevanceLanguage=relevance_language, safeSearch=safe_search, type='video', videoCaption='closedCaption', order=order, videoDuration=video_duration)
        else:
            r = self.youtube_api_client.search().list(
                part=part, maxResults=max_results, publishedAfter=published_after, q=search_for, regionCode=region_cd, safeSearch=safe_search, relevanceLanguage=relevance_language, type='video', videoCaption='closedCaption', order=order, videoDuration=video_duration
            )

        qry.search_response = SearchResponse(r.execute())
        self.searches.append(qry)

        return qry

    def fetch_transcript(self, video_id):
        '''Checks to see if a transcript has already been saved, fetching it if not. Returns a dictionary of the transcript'''
        file_path = self.transcripts_folder() + video_id + '_transcript.parquet'

        if os.path.exists(file_path):
            return True

        try:
            captions = YouTubeTranscriptApi.get_transcript(
                video_id=video_id)
            if captions != None:
                df = pd.DataFrame(captions)
                table = pa.Table.from_pandas(df)
                pq.write_table(table, file_path,
                               use_dictionary=True, compression='gzip')
            return True
        except:
            return False

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
