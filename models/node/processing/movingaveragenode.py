import numpy as np
from typing import List, Dict, Final
from models.framework_data import FrameworkData
from models.node.processing.processing_node import ProcessingNode
from models.exception.missing_parameter import MissingParameterError
from models.exception.invalid_parameter_value import InvalidParameterValue

class MovingAverageNode(ProcessingNode):
    """ This node computes the moving average of the input signal with the specified window size.

    Attributes:
        _MODULE_NAME (str): The name of the module (in this case ``node.processing.movingaverage``)
    """
    _MODULE_NAME: Final[str] = 'node.processing.movingaveragenode'

    INPUT_MAIN: Final[str] = 'main'
    OUTPUT_MAIN: Final[str] = 'main'

    def _validate_parameters(self, parameters: dict):
        """ Validates the parameters passed to this node. """
        if 'window_size' not in parameters:
            raise MissingParameterError(module=self._MODULE_NAME, name=self.name, parameter='window_size')
        if type(parameters['window_size']) is not int or parameters['window_size'] < 1:
            raise InvalidParameterValue(module=self._MODULE_NAME, name=self.name, parameter='window_size', cause='must_be_positive_int')
        super()._validate_parameters(parameters)
        
    def _initialize_parameter_fields(self, parameters: dict):
        """ Initializes the parameter fields of this node. """
        self._window_size = parameters['window_size']
        super()._initialize_parameter_fields(parameters)

    def _is_next_node_call_enabled(self) -> bool:
        """ Determines if the next node call is enabled. """
        return self._output_buffer[self.OUTPUT_MAIN].get_data_count() > 0

    def _is_processing_condition_satisfied(self) -> bool:
        """ Checks if the processing condition is satisfied. """
        return self._input_buffer[self.INPUT_MAIN].get_data_count() > 0
        
    def _process(self, data: Dict[str, FrameworkData]) -> Dict[str, FrameworkData]:
        """ This method processes the data that was inputted to the node. It computes the moving average of the input signal.

        :param data: The data that was inputted to the node.
        :type data: Dict[str, FrameworkData]

        :return: The smoothed data.
        :rtype: Dict[str, FrameworkData]
        """
        input_data = data[self.INPUT_MAIN]
        smoothed_data = FrameworkData(sampling_frequency_hz=input_data.sampling_frequency,
                                      channels=input_data.channels)

        for channel in input_data.channels:
            signal = input_data.get_data_on_channel(channel)
            smoothed_signal = np.convolve(signal, np.ones(self.window_size) / self.window_size, mode='same')
            smoothed_data.input_data_on_channel(smoothed_signal, channel)

        return smoothed_data
    
    def _get_inputs(self) -> List[str]:
        """ Returns the list of input keys. """
        return [self.INPUT_MAIN]

    def _get_outputs(self) -> List[str]:
        """ Returns the list of output keys. """
        return [self.OUTPUT_MAIN]