class Keyring:
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
