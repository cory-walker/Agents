import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os
import Keyring
from datetime import datetime


class Classifer:
    def __init__(self, dimensions_folder):
        self.dim_emotions = pd.read_csv(dimensions_folder + 'dim_emotions.csv')

        self.document_types = {'article': 1,
                               'transcript': 2, 'youtube channel search': 3}
        self.source_systems = {'user': 1, 'youtube': 2, 'scholar': 3}


class Librarian:
    def __init__(self, library_path='./', keyring=Keyring.Keyring(), refresh_index=True):
        self.keyring = Keyring.Keyring()

        if not library_path.endswith('/'):
            library_path += '/'

        self.library_path = library_path
        self.check_for_folder(self.library_path)
        self.check_for_folder(self.transcripts_folder())
        self.check_for_folder(self.summaries_folder())
        self.check_for_folder(self.dimensions_folder())

        if not os.path.exists(self.library_index_path()):
            self.create_document_store()

        self.classifier = Classifer(self.dimensions_folder())
        self.libraryIndex = pd.read_parquet(self.library_index_path())
        self.libraryIndex.set_index('lib_key')

        if refresh_index:
            self.refresh_index()

    def channels_list(self, category=''):
        dfc = pd.read_csv(self.channels_index_path())
        if category > '':
            dfc = dfc[dfc['category'] == category]

        return dfc['channel_id'].to_list()

    def create_document_store(self):
        df = pd.DataFrame(columns=[
                          'lib_key', 'source_id', 'rec_mod_dtm', 'doc_type_id',  'source_system_id', 'path'])
        table = pa.Table.from_pandas(df)
        pq.write_table(table, self.library_path + 'library_index.parquet',
                       use_dictionary=True, compression='gzip')

    def check_for_folder(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def library_index_path(self):
        return self.library_path + 'data/library_index.parquet'

    def channels_index_path(self):
        return self.library_path + 'data/youtube_channels_index.csv'

    def channels_folder(self):
        return self.library_path + 'data/channels/'

    def transcripts_folder(self):
        return self.library_path + 'data/transcripts/'

    def summaries_folder(self):
        return self.library_path + 'data/summaries/'

    def dimensions_folder(self):
        return self.library_path + 'data/dimensons/'

    def upsert_library_index(self, source_id, rec_mod_dtm, doc_type_id, source_system_id, path):

        rows, columns = self.libraryIndex.shape

        new_lib_key = rows + 1
        dfn = pd.DataFrame([{'lib_key': new_lib_key, 'source_id': source_id, 'rec_mod_dtm': rec_mod_dtm,
                             'doc_type_id': doc_type_id, 'source_system_id': source_system_id, 'path': path}])

        # Does the entry for the file already exist? If not, then insert it
        if not path in self.libraryIndex['path'].to_list():
            self.libraryIndex = pd.concat([self.libraryIndex, dfn])
            return True
        else:
            cur_rec_mod_dtm = self.libraryIndex[self.libraryIndex['path']
                                                == path]['rec_mod_dtm'][0]
            # If it does have an entry, is the file modification date different?
            if rec_mod_dtm != cur_rec_mod_dtm:
                new_lib_key = self.libraryIndex[self.libraryIndex['path']
                                                == path]['lib_key']

                self.libraryIndex = self.libraryIndex[~self.library_path['path'] == path]
                self.libraryIndex = pd.concat([self.libraryIndex, dfn])
                return True

        return False

    def rewrite_library_index(self):
        table = pa.Table.from_pandas(self.libraryIndex)
        pq.write_table(table, self.library_index_path(),
                       use_dictionary=True, compression='gzip')

    def check_folder_for_new_files(self, doc_type, source_system, folder, source_id_replacement):
        any_changes = False
        doc_type_id = self.classifier.document_types[doc_type]
        source_system_id = self.classifier.source_systems[source_system]

        for file in os.listdir(folder):
            path = folder + file
            source_id = file.replace(source_id_replacement, '')
            rec_mod_dtm = datetime.fromtimestamp(
                os.path.getctime(path)).strftime('%Y-%m-%d %H:%M:%S')

            new_items = self.upsert_library_index(
                source_id, rec_mod_dtm, doc_type_id, source_system_id, path)

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
