from typing import Any
import pickle

class Config:

    def __init__(self) -> None:
        self._config = {}

    def set_prop(self, name:str, data:Any) -> None:
        """Set a configuration property.
            Note that this method also creates a new
            property

        Args:
            name (str): name of property to set
            data (Any): data stored for property
        """
        self._config[name] = data

    def get_prop(self, name:str) -> Any:
        """Get the data assoicated with a property



        Args:
            name (str): name of property

        Returns:
            Any: data associated with property 
                Raises an error if property does not exist 
        """
        data = None
        try:
            data = self._config[name]
        except Exception as e:
            print(e)
            raise Exception('Propetry: {}, does not exist'.format(name))
        return data

    def save(self, path):
        """Save config to disk

            Python Pickle is used to save data
        Args:
            path (str): Path to save config
        """
        # path = self.BASE_PATH  + model_info + '_hyper' + '.pkl'
        # filenames.append(path)
        with open(path, 'wb') as f:
            pickle.dump(self., f)
        