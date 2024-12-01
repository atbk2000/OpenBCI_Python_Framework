import abc
from typing import Final, Any, Tuple, List

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.ensemble._forest import RandomForestClassifier

from models.exception.invalid_parameter_value import InvalidParameterValue
from models.exception.missing_parameter import MissingParameterError
from models.framework_data import FrameworkData
from models.node.processing.trainable.classifier.sklearn_classifier import SKLearnClassifier

class RandomForest(SKLearnClassifier):
    """ This node is a wrapper for the ``RandomForestClassifier`` classifier from sklearn. It is a subclass of
    ``SKLearnClassifier`` and implements the abstract methods from it. The ``RandomForestClassifier`` classifier
    can be used to classify data into two or more classes.

    Attributes:
        _MODULE_NAME (str): The name of the module (in this case ``node.processing.trainable.classifier``)
    """
    _MODULE_NAME: Final[str] = 'node.processing.trainable.classifier.randomforest'

    def _initialize_parameter_fields(self, parameters: dict):
        """ Initializes the parameters of this node. In this case it initializes the parameters from its superclass and
        adds the parameters specific to this node.

        :param parameters: The parameters to initialize.
        :type parameters: dict
        """
        self.n_estimators = parameters['n_estimators']
        self.random_state = parameters['random_state']
        super()._initialize_parameter_fields(parameters)

    @abc.abstractmethod
    def _validate_parameters(self, parameters: dict):
        """ Validates the parameters passed to this node. In this case it checks if the parameters are present and if they
        are of the correct type, extending from its superclass.

        :param parameters: The parameters to validate.
        :type parameters: dict
        """
        if 'random_state' not in parameters:
            raise MissingParameterError(module=self._MODULE_NAME, name=self.name,
                                        parameter='random_state')
        
        if 'n_estimators' not in parameters:
            raise MissingParameterError(module=self._MODULE_NAME, name=self.name,
                                        parameter='n_estimators')
        
        if type(parameters['random_state']) is not int:
            raise InvalidParameterValue(module=self._MODULE_NAME,name=self.name,
                                        parameter='random_state',
                                        cause='must_be_int')

        if parameters['random_state'] <= 0:
            raise InvalidParameterValue(module=self._MODULE_NAME, name=self.name, parameter='random_state', cause='must_be_greater_than_0')
        
        if type(parameters['n_estimators']) is not int:
            raise InvalidParameterValue(module=self._MODULE_NAME, name=self.name,
                                        parameter='n_estimators',
                                        cause='must_be_int')
         
        if parameters['n_estimators'] <= 0:
            raise InvalidParameterValue(module=self._MODULE_NAME, name=self.name, parameter='n_estimators', cause='must_be_greater_than_0')

    def _initialize_trainable_processor(self) -> Tuple[TransformerMixin, BaseEstimator]:
        """ Initializes the trainable processor. In this case it initializes the ``RandomForestClassifier`` classifier
        from sklearn.

        :return: The initialized ``RandomForestClassifier`` classifier.
        :rtype: (TransformerMixin, BaseEstimator)
        """

        rfClassifier = RandomForestClassifier(n_estimators=self.n_estimators, random_state=self.random_state, verbose=100)

        return rfClassifier

    @abc.abstractmethod
    def _should_retrain(self) -> bool:
        """ Checks if the processor should be retrained. In this case it always returns False, so the processor will
        never be retrained.
        """
        return False

    @abc.abstractmethod
    def _is_next_node_call_enabled(self) -> bool:
        """ Checks if the next node call is enabled. In this case it checks if the processor is trained and if the
        output buffer has data.
        """
        return self._is_trained

    def _format_processed_data(self, processed_data: Any, sampling_frequency: float) -> FrameworkData:
        """ Formats the processed data. In this case it creates a ``FrameworkData`` object and adds the processed data
        to it.

        :param processed_data: The processed data.
        :type processed_data: Any
        :param sampling_frequency: The sampling frequency of the processed data.
        :type sampling_frequency: float

        :return: The formatted data.
        :rtype: FrameworkData
        """
        formatted_data = FrameworkData(sampling_frequency_hz=sampling_frequency)
        formatted_data.input_data_on_channel(processed_data)
        return formatted_data