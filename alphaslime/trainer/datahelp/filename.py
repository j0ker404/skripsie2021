from alphaslime.store.config import Config
class FileName:
    """Base Class for FileName generators

    Returns:
        [type]: [description]
    """

    def __init__(self) -> None:
        return        

    def gen_name(self, agent, prefix=''):
        """Generate a file base name used for saving
        

        Args:
            agent (Agent): Trained agent

            prefix: Prefix to add to base file name

        Return:
            file_name (str): Base file name
        """
        pass
