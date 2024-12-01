import abc
from typing import Final, Tuple
from scipy.signal import butter, iirnotch
from models.framework_data import FrameworkData

from models.exception.invalid_parameter_value import InvalidParameterValue
from models.exception.missing_parameter import MissingParameterError
from models.node.processing.filter.filter import Filter


class Notch(Filter):
    """ This is a notch filter. It's a node that filters the input data with a notch filter. A notch filter
    is a filter that passes frequencies above a certain cutoff value and rejects (attenuates) frequencies below that range.\n
    This class does that by creating a Butterworth scipy filter with the given parameters and btype='high'. The filtering 
    itself is done by the parent class Filter in the _process method.

    Attributes:
        _MODULE_NAME (`str`): The name of the module (in his case ``node.processing.filter.notch``)
    
    configuration.json usage:
        **module** (*str*): The name of the module (``node.processing.filter``)\n
        **type** (*str*): The type of the node (``notch``)\n
        **cut_frequency_hz** (*float*): The cut frequency in Hz.\n

        **buffer_options** (*dict*): Buffer options.\n
            **clear_output_buffer_on_data_input** (*bool*): Whether to clear the output buffer when new data is inserted in the input buffer.\n
            **clear_input_buffer_after_process** (*bool*): Whether to clear the input buffer after processing.\n
            **clear_output_buffer_after_process** (*bool*): Whether to clear the output buffer after processing.\n
    
    """
    _MODULE_NAME: Final[str] = 'node.processing.filter.notch'

    def _validate_parameters(self, parameters: dict):
        """ Validates the parameters passed to this node. In this case it checks if the parameters are present and if they
        are of the correct type.

        :param parameters: The parameters passed to this node.
        :type parameters: dict

        :raises MissingParameterError: the ``cut_frequency_hz`` parameter is required.
        :raises MissingParameterError: the ``order`` parameter is required.
        :raises InvalidParameterValue: the ``cut_frequency_hz`` parameter must be a number.
        :raises InvalidParameterValue: the ``order`` parameter must be an int.
        """
        if 'cut_frequency_hz' not in parameters:
            raise MissingParameterError(module=self._MODULE_NAME,name=self.name,
                                        parameter='cut_frequency_hz')
        if 'Q' not in parameters:
            raise MissingParameterError(module=self._MODULE_NAME,name=self.name,
                                        parameter='Q')

        if type(parameters['cut_frequency_hz']) is not float and type(
                parameters['cut_frequency_hz']) is not int:
            raise InvalidParameterValue(module=self._MODULE_NAME,name=self.name,
                                        parameter='cut_frequency_hz',
                                        cause='must_be_number')

        if type(parameters['Q']) is not int:
            raise InvalidParameterValue(module=self._MODULE_NAME,name=self.name,
                                        parameter='Q',
                                        cause='must_be_int')

    @abc.abstractmethod
    def _initialize_parameter_fields(self, parameters: dict):
        """ Initializes the parameter fields of this node.
        """
        super()._initialize_parameter_fields(parameters)

    def _get_filter_coefficients(self, parameters: dict, sampling_frequency_hz: float) -> Tuple[list, list]:
        """ Returns the filter coefficients for the given parameters and sampling frequency. In this case it returns the
        coefficients of a Butterworth filter with the given parameters and btype='high'.

        :param parameters: The parameters passed to this node.
        :type parameters: dict
        :param sampling_frequency_hz: The sampling frequency in Hz.
        :type sampling_frequency_hz: float

        :return: a scipy Butterworth filter with the given parameters and btype='high'.
        :rtype: Tuple[list, list]
        """
        
        return iirnotch(
            parameters['cut_frequency_hz'],
            parameters['Q'],
            fs=sampling_frequency_hz
        )