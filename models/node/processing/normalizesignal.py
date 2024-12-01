import numpy as np
from typing import List, Dict, Final
from models.framework_data import FrameworkData
from models.node.processing.processing_node import ProcessingNode
from models.exception.invalid_parameter_value import InvalidParameterValue

class NormalizeSignal(ProcessingNode):
    """ This node normalizes the input signal by subtracting the mean and dividing by the standard deviation.

    Attributes:
        _MODULE_NAME (str): The name of the module (in this case ``node.processing.normalizesignal``)
    """
    _MODULE_NAME: Final[str] = 'node.processing.normalizesignal'

    INPUT_MAIN: Final[str] = 'main'
    OUTPUT_MAIN: Final[str] = 'main'
    
    def _is_next_node_call_enabled(self) -> bool:
        """ This method will return ``True`` if the next node call is enabled. This method will always return ``True``
        because the next node call is always enabled.
        """
        return True

    def _is_processing_condition_satisfied(self) -> bool:
        """ This method will return ``True`` if the processing condition is satisfied. This method will return ``True`` if the input buffer has data.

        :return: ``True`` if the input buffer has data, ``False`` otherwise.
        :rtype: bool
        """
        return self._input_buffer[self.INPUT_MAIN].get_data_count() > 0

    def _validate_parameters(self, parameters: dict):
        """ Validates the parameters passed to this node. In this case, there are no specific parameters to validate.

        :param parameters: The parameters to validate.
        :type parameters: dict
        """
        pass  # No specific parameters to validate for normalization

    def _process(self, data: Dict[str, FrameworkData]) -> Dict[str, FrameworkData]:
        """ This method processes the data that was inputted to the node. It normalizes the input signal.

        :param data: The data that was inputted to the node.
        :type data: Dict[str, FrameworkData]

        :return: The normalized data.
        :rtype: Dict[str, FrameworkData]
        """
        input_data = data[self.INPUT_MAIN]
        normalized_data = FrameworkData(sampling_frequency_hz=input_data.sampling_frequency,
                                        channels=input_data.channels)

        for channel in input_data.channels:
            signal = input_data.get_data_on_channel(channel)
            normalized_signal = (signal - np.mean(signal)) / np.std(signal)
            normalized_data.input_data_on_channel(normalized_signal, channel)

        return {
            self.OUTPUT_MAIN: normalized_data
        }
    
    def _get_inputs(self) -> List[str]:
        """ This method will return the inputs of the node.
        
        :return: The inputs of the node.
        :rtype: list
        """
        return [
            self.INPUT_MAIN
        ]

    def _get_outputs(self) -> List[str]:
        """ This method will return the outputs of the node.
        
        :return: The outputs of the node.
        :rtype: list
        """
        return [
            self.OUTPUT_MAIN
        ]