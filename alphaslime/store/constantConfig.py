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
    def __init__(self, init_dict=None) -> None:
        '''
            Constructor
        '''
        super().__init__(init_dict)

    def save(self, path):
        """Save config data to disk using python
        Pickle.

        Env vairable is saved as a string

        Args:
            path (str): Path to save file
        """
        # config = copy.deepcopy(self.__config)
        cfg = self.load(self._config)
        cfg.set('env', str(cfg.get('env')))
        with open(path, 'wb') as f:
            pickle.dump(cfg._config, f)