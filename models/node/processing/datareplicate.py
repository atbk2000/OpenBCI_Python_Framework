from typing import Dict, Final, List

from models.exception.invalid_parameter_value import InvalidParameterValue
from models.exception.missing_parameter import MissingParameterError
from models.framework_data import FrameworkData
from models.node.processing.processing_node import ProcessingNode


class DataReplicate(ProcessingNode):
    """ This node is used to replicate the data in the input buffer by a certain number.

    Attributes:
        _MODULE_NAME (str): The name of the module (in this case ``models.node.
        processing.data_replicate``)
        INPUT_MAIN (str): The name of the main input (in this case ``main``) 
        OUTPUT_MAIN (str): The name of the main output (in this case ``main``)

    configuration.json usage:
        **module** (*str*): The name of the module (``node.processing.data_replicate``)\n
        **type** (*str*): The name of the class (``DataReplicate``)\n
        **data_repetition_count** (*int*): Number of times that the data in the input buffer must be repeated.\n
        **use_extra_data** (*bool*): Whether to use extra data when replicating.\n
        **extra_data** (*int*): Additional data. Simply multiplies the last data in the input by a certain number. Must be smaller than ``data_repetition_count``. Only mandatory if ``use_extra_data`` is True.\n 
    """

    _MODULE_NAME: Final[str] = 'node.processing.dataReplicate'

    INPUT_MAIN: Final[str] = 'main'
    OUTPUT_MAIN: Final[str] = 'main'

    def _validate_parameters(self, parameters: dict):
        """ Validate the parameters passed to the node. This method will raise an exception if the parameters are not valid or when they don't exist.

        :param parameters: The parameters passed to the node.
        :type parameters: dict

        :raises MissingParameterError: The ``data_repetition_count`` parameter is required. The ``extra_data`` parameter is required if ``use_extra_data`` is True.
        :raises InvalidParameterValue: The ``data_repetition_count`` and ``extra_data`` parameters must be integers. The ``use_extra_data`` parameter must be bool. 
        :raises InvalidParameterValue: The ``data_repetition_count`` parameter must be bigger than 1. The ``extra_data`` parameter must be bigger than 1 and smaller than ``data_repetition_count``.
        """

        if 'data_repetition_count' not in parameters:
            raise MissingParameterError(module=self._MODULE_NAME, name=self.name, parameter='data_repetition_count')
        
        if type(parameters['data_repetition_count']) is not int:
            raise InvalidParameterValue(module=self._MODULE_NAME, name=self.name, parameter='data_repetition_count', cause='must_be_int')
        
        if parameters['data_repetition_count'] <= 1:
            raise InvalidParameterValue(module=self._MODULE_NAME, name=self.name, parameter='data_repetition_count', cause='must_be_bigger_than_1')
        
        if 'use_extra_data' not in parameters:
            raise MissingParameterError(module=self._MODULE_NAME, name=self.name, parameter='use_extra_data')
        
        if type(parameters['use_extra_data']) is not bool:
            raise InvalidParameterValue(module=self._MODULE_NAME, name=self.name, parameter='use_extra_data', cause='must_be_bool')
        
        if parameters['use_extra_data'] == True:
            if 'extra_data' not in parameters:
                raise MissingParameterError(module=self._MODULE_NAME, name=self.name, parameter='extra_data')
            
            if type(parameters['extra_data']) is not int:
                raise InvalidParameterValue(module=self._MODULE_NAME, name=self.name, parameter='extra_data', cause='must_be_int')
            
            if (parameters['extra_data'] <= 1) or (parameters['extra_data'] >= parameters['data_repetition_count']):
                raise InvalidParameterValue(module=self._MODULE_NAME, name=self.name, parameter='extra_data', cause='invalid_value')
        
    def _initialize_parameter_fields(self, parameters: dict):
        """ Initialize the parameter fields of the node. This method will set the ``data_repetition_count`` and ``use_extra_data`` attributes and all the parent attributes as well. It sets also ``extra_data`` if ``use_extra_data`` is True.

        :param parameters: The parameters passed to the node.
        :type parameters: dict
        """
        super()._initialize_parameter_fields(parameters)
        self.data_repetition_count = parameters['data_repetition_count']
        self.use_extra_data = parameters['use_extra_data']
        if self.use_extra_data == True:
            self.extra_data = parameters['extra_data']


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
        """ This method will process the data in the input buffer and return the result in the output buffer. This method will fill the input buffer with a certain amount of samples.
        
        :param data: The data to process.
        :type data: dict

        :return: The processed data.
        :rtype: dict
        """
        channels = data['main'].channels
        
        processed_data = FrameworkData.from_multi_channel(1, channels, [])

        for channel in channels:
            channel_elements = data[self.INPUT_MAIN].get_data_on_channel(channel)

            for element in channel_elements:
                processed_data.input_data_on_channel([element for _ in range(0,self.data_repetition_count)], channel)

            if self.use_extra_data == True:
                processed_data.input_data_on_channel([channel_elements[-1] for _ in range(0, self.extra_data)], channel)

        return {
            self.OUTPUT_MAIN: processed_data
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