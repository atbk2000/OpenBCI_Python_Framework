import numpy as np
from typing import List, Dict, Final
from models.framework_data import FrameworkData
from models.node.processing.processing_node import ProcessingNode
from models.exception.invalid_parameter_value import InvalidParameterValue

class RectifySignal(ProcessingNode):
    """ This node rectifies the input signal by taking the absolute value of each element.

    Attributes:
        _MODULE_NAME (str): The name of the module (in this case ``node.processing.rectifysignal``)
    """
    _MODULE_NAME: str = 'node.processing.rectifysignal'

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

    def _process(self, data: Dict[str, FrameworkData]) -> Dict[str, FrameworkData]:
        """ This method processes the data that was inputted to the node. It rectifies the input signal.

        :param data: The data that was inputted to the node.
        :type data: Dict[str, FrameworkData]

        :return: The rectified data.
        :rtype: Dict[str, FrameworkData]
        """
        input_data = data[self.INPUT_MAIN]
        rectified_data = FrameworkData(sampling_frequency_hz=input_data.sampling_frequency,
                                       channels=input_data.channels)

        for channel in input_data.channels:
            signal = input_data.get_data_on_channel(channel)
            rectified_signal = np.abs(signal)
            rectified_data.input_data_on_channel(rectified_signal, channel)

        return {
            self.OUTPUT_MAIN: rectified_data
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