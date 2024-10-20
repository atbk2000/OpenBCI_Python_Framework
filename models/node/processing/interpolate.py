from typing import Dict, Final, List

from models.exception.invalid_parameter_value import InvalidParameterValue
from models.exception.missing_parameter import MissingParameterError
from models.framework_data import FrameworkData
from models.node.processing.processing_node import ProcessingNode


class Interpolate(ProcessingNode):
    """ This node is used to interpolate the data in the input buffer. Data that was segmented is interpolated using this node and placed in the output buffer.

    Attributes:
        _MODULE_NAME (str): The name of the module (in this case ``models.node.
        processing.interpolate``)
        INPUT_MAIN (str): The name of the main input (in this case ``main``) 
        OUTPUT_MAIN (str): The name of the main output (in this case ``main``)

    configuration.json usage:
        **module** (*str*): The name of the module (``models.node.processing``)\n
        **type** (*str*): The name of the class (``Interpolate``)\n
        **window_size** (*int*): The size of the window (epoch) in samples.\n
        **sliding_window** (*bool*): tells whether sliding window is used or not.\n
        **step_size** (*int*): The value of the step. It must be positive and smaller than the window. Only required if ``sliding_window`` is set to True.\n
    """

    _MODULE_NAME: Final[str] = 'node.processing.interpolate'

    INPUT_MAIN: Final[str] = 'main'
    OUTPUT_MAIN: Final[str] = 'main'

    def _validate_parameters(self, parameters: dict):
        """ Validate the parameters passed to the node. This method will raise an exception if the parameters are not valid or when they don't exist.

        :param parameters: The parameters passed to the node.
        :type parameters: dict

        :raises MissingParameterError: The ``window_size`` and ``sliding_window`` parameters are required. The ``step_size`` parameter is required if ``sliding_window`` is True.
        :raises InvalidParameterValue: The ``window_size`` and ``step_size`` parameters must be integers. The ``sliding_window`` parameter must be bool. 
        :raises InvalidParameterValue: The ``window_size`` parameter must be greater than 0. The ``step_size`` parameter must greater than 0 and smaller than ``window_size``.
        """

        if 'window_size' not in parameters:
            raise MissingParameterError(module=self._MODULE_NAME, name=self.name, parameter='window_size')
        
        if type(parameters['window_size']) is not int:
            raise InvalidParameterValue(module=self._MODULE_NAME, name=self.name, parameter='window_size', cause='must_be_int')
        
        if parameters['window_size'] <= 1:
            raise InvalidParameterValue(module=self._MODULE_NAME, name=self.name, parameter='window_size', cause='must_be_greater_than_0')
        
        if 'sliding_window' not in parameters:
            raise MissingParameterError(module=self._MODULE_NAME, name=self.name, parameter='sliding_window')
        
        if type(parameters['sliding_window']) is not bool:
            raise InvalidParameterValue(module=self._MODULE_NAME, name=self.name, parameter='sliding_window', cause='must_be_bool')
        
        if parameters['sliding_window'] == True:
            if 'step_size' not in parameters:
                raise MissingParameterError(module=self._MODULE_NAME, name=self.name, parameter='step_size')
            
            if type(parameters['step_size']) is not int:
                raise InvalidParameterValue(module=self._MODULE_NAME, name=self.name, parameter='step_size', cause='must_be_int')
            
            if (parameters['step_size'] <= 1) or (parameters['step_size'] >= parameters['window_size']):
                raise InvalidParameterValue(module=self._MODULE_NAME, name=self.name, parameter='step_size', cause='invalid_value')
        
    def _initialize_parameter_fields(self, parameters: dict):
        """ Initialize the parameter fields of the node. This method will set the ``window_size`` and ``sliding_window`` attributes and all the parent attributes as well. It sets also ``step_size`` if ``sliding_window`` is True.

        :param parameters: The parameters passed to the node.
        :type parameters: dict
        """
        super()._initialize_parameter_fields(parameters)
        self.window_size = parameters['window_size']
        self.sliding_window = parameters['sliding_window']
        if self.sliding_window == True:
            self.step_size = parameters['step_size']


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

    def _process(self, data: Dict[str, FrameworkData]) ->  Dict[str, FrameworkData]:
        """ This method will process the data in the input buffer and return the result in the output buffer. This method will receive the segmented data and interpolate it.
        
        :param data: The data to process.
        :type data: dict

        :return: The processed data.
        :rtype: dict
        """
        channels = data[self.INPUT_MAIN].get_channels_as_set()
        
        processed_data = FrameworkData.from_multi_channel(1, channels, [])

        if self.sliding_window == True:

            windows_count = data[self.INPUT_MAIN].get_data_count()
            intersection_lenght = self.window_size - self.step_size

            for channel in channels:
                window = 0
                merged_data = []

                for index in range(self.window_size):
                    merged_data.append([data[self.INPUT_MAIN]._data[channel][window]])

                window += 1
                window_end_index = self.window_size

                while window < windows_count:

                    for index in range(window_end_index - intersection_lenght, window_end_index):
                        merged_data[index].append(data[self.INPUT_MAIN]._data[channel][window])

                    for index in range(self.step_size):
                        merged_data.append([data[self.INPUT_MAIN]._data[channel][window]])

                    window += 1
                    window_end_index += self.step_size

                for index in range(len(merged_data)):
                    if len(merged_data[index]) > 1:
                        processed_data.input_data_on_channel([max(merged_data[index])], channel)
                    else:
                        processed_data.input_data_on_channel(merged_data[index], channel)
        else:

            for channel in channels:
                for window in range(data[self.INPUT_MAIN].get_data_count()):

                    processed_data.input_data_on_channel([data[self.INPUT_MAIN]._data[channel][window] for _ in range(0, self.window_size)], channel)

        return {
            self.OUTPUT_MAIN:    processed_data
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