import math

from typing import Final

from models.exception.invalid_parameter_value import InvalidParameterValue
from models.exception.missing_parameter import MissingParameterError
from models.framework_data import FrameworkData
from models.node.processing.segmenter.segmenter import Segmenter


class SlidingWindowSegmenter(Segmenter):
    """ Segments the input data into fixed size windows. Each window is a list of samples, normally called epoch. The windows are obtained sliding over the input data. For example, if step size is 50 and the window size is 100, the first window is obtained from sample 1 up to 100, the second window from sample 50 to 150, and so on.


    This is important because the other nodes in the pipeline expect the data to be segmented in this way. For exemple,
    the trainable feature extractor node expects the input data to be segmented in epochs, so if you segment the data in epochs of 
    200 samples and configure the feature extractor training_set_size to be 10, the feature extractor will use 2000 samples
    to train the model. If you segment the data in epochs of 100 samples and configure the feature extractor training_set_size
    to be 10, the feature extractor will use 1000 samples to train the model. 

    Attributes:
        _MODULE_NAME (str): The name of the module (in his case ``node.processing.segmenter.slidingwindowsegmenter``)
    
    configuration.json usage:
        **module** (*str*): The name of the module (``node.processing.segmenter``)\n
        **type** (*str*): The type of the node (``SlidingWindowSegmenter``)\n
        **window_size** (*int*): The size of the window (epoch) in samples.\n
        **filling_value** (*str*): The value to fill the last window if it's not complete. Can be ``zero`` or ``latest``.\n
        **step_size** (*int*): The value of the step. It must be positive and smaller than the window.\n
        **buffer_options** (*dict*): Buffer options.\n
            **clear_output_buffer_on_data_input** (*bool*): Whether to clear the output buffer when new data is inserted in the input buffer.\n
            **clear_input_buffer_after_process** (*bool*): Whether to clear the input buffer after processing.\n
            **clear_output_buffer_after_process** (*bool*): Whether to clear the output buffer after processing.\n
        

    """
    _MODULE_NAME: Final[str] = 'node.processing.segmenter.slidingwindowsegmenter'

    def __init__(self, parameters: dict):
        super().__init__(parameters)
        self.window_size = parameters['window_size']
        self.filling_value = parameters['filling_value']
        self.step_size = parameters['step_size']

    def _is_processing_condition_satisfied(self) -> bool:
        """ Returns whether the processing condition is satisfied. In this case it returns True if there is data in the
        input buffer.
        """

        return self._input_buffer[self.INPUT_MAIN].get_data_count() >= self.window_size

    def _validate_parameters(self, parameters: dict):

        """ Validates the parameters passed to this node. In this case it checks if the parameters are present and if they
        are of the correct type.

        :param parameters: The parameters passed to this node.
        :type parameters: dict

        :raises MissingParameterError: the ``window_size`` parameter is required.
        :raises InvalidParameterValue: the ``window_size`` parameter must be an int.
        :raises InvalidParameterValue: the ``window_size`` parameter must be greater than 0.
        :raises MissingParameterError: the ``filling_value`` parameter is required.
        :raises InvalidParameterValue: the ``filling_value`` parameter must be a str.
        :raises InvalidParameterValue: the ``filling_value`` parameter must be in [``zero``, ``latest``].
        :raises MissingParameterError: the ``step_size`` parameter is required.
        :raises InvalidParameterValue: the ``step_size`` parameter must be an int.
        :raises InvalidParameterValue: the ``step_size`` parameter must be greater than 0 and smaller than ''window_size''.

        """
        
        super()._validate_parameters(parameters)
        
        if 'window_size' not in parameters:
            raise MissingParameterError(module=self._MODULE_NAME, name=self.name,
                                        parameter='window_size')
        if 'filling_value' not in parameters:
            raise MissingParameterError(module=self._MODULE_NAME, name=self.name,
                                        parameter='filling_value')
        if 'step_size' not in parameters:
            raise MissingParameterError(module=self._MODULE_NAME, name=self.name, 
                                        parameter='step_size')
        if type(parameters['window_size']) is not int:
            raise InvalidParameterValue(module=self._MODULE_NAME, name=self.name,
                                        parameter='window_size',
                                        cause='must_be_int')
        if parameters['window_size'] < 1:
            raise InvalidParameterValue(module=self._MODULE_NAME, name=self.name,
                                        parameter='window_size',
                                        cause='must_be_greater_than_0')
        if type(parameters['filling_value']) is not str:
            raise InvalidParameterValue(module=self._MODULE_NAME, name=self.name,
                                        parameter='filling_value',
                                        cause='must_be_str')
        if parameters['filling_value'] not in ['zero', 'latest']:
            raise InvalidParameterValue(module=self._MODULE_NAME, name=self.name,
                                        parameter='filling_value',
                                        cause='must_be_in.[zero, latest]')
        if type(parameters['step_size']) is not int:
            raise InvalidParameterValue(module=self._MODULE_NAME, name=self.name,                       
                                        parameter='step_size',
                                        cause='must_be_int')
        if parameters['step_size'] < 1 or parameters['step_size'] >= parameters['window_size']:
            raise InvalidParameterValue(module=self._MODULE_NAME, name=self.name, 
                                        parameter='step_size', cause='must_be_greater_than_0_and_smaller_than_window_size')

    def segment_data(self, data: FrameworkData) -> FrameworkData:
        """Method that segments the data into fixed size windows, with sliding. It just segments the data on the main input channel, 
        and if the last window is not complete, it fills it with zeros or with the last sample of the window, depending on
        the ``filling_value`` parameter.

        :param data: The data to segment.
        :type data: FrameworkData

        :return: The segmented data.
        :rtype: FrameworkData
        """
        segmented_data: FrameworkData = FrameworkData(sampling_frequency_hz=data.sampling_frequency)

        index = 0
        data_count = data.get_data_count()
        channels = data.get_channels_as_set()

        while (index + self.window_size) < data_count:
            for channel in channels:
                segmented_data.input_data_on_channel([data._data[channel][index:(index + self.window_size)]], channel)
            index += self.step_size

        remaining_data = data_count - index

        if remaining_data != 0:
            for channel in channels:
                filling_vector = []
                
                if self.filling_value == 'zero':
                    filling_vector = [0] * (self.window_size - remaining_data)
                elif self.filling_value == 'latest':
                    filling_vector = [data._data[channel][data_count - 1]] * (self.window_size - remaining_data)

                segmented_data.input_data_on_channel([data._data[channel][index:data_count] + filling_vector], channel)

        return segmented_data
