from alphaslime.store.config import Config
import pickle
import copy


class Constants(Config):
    """Custom config class

        This class has a modifed save method so that
        the env class can be pickled
    Args:
        Config (Config): Base class
    """
    def __init__(self) -> None:
        '''
            Constructor
        '''
        super().__init__()

    def save(self, path):
        """Save config data to disk using python
        Pickle.

        Env vairable is saved as a string

        Args:
            path (str): Path to save file
        """
        config = copy.deepcopy(self._config)
        config.set('env') = config.get('env')
        with open(path, 'wb') as f:
            pickle.dump(config, f)