from alphaslime.store.config import Config
from alphaslime.store.constantConfig import Constants
from alphaslime.trainer.datahelp.filename import FileName
from alphaslime.trainer.trainer import Trainer


class SelfplayPPOTrainer(Trainer):

    def __init__(self, CONSTANTS: Constants) -> None:
        super().__init__(CONSTANTS)

    def train(self, training_config: Config, agent_config: Config, fileNamer: FileName, prefix=''):
        # return super().train(training_config, agent_config, fileNamer, prefix=prefix)
        # load champion list

        # load champion
        pass