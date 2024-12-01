import numpy as np
from typing import Dict, List, Final
from models.framework_data import FrameworkData
from models.node.processing.processing_node import ProcessingNode
from models.exception.missing_parameter import MissingParameterError
from models.exception.invalid_parameter_value import InvalidParameterValue
from collections import Counter

class FeatureExtractionNode(ProcessingNode):
    """ This node extracts features from the input signal and associates them with event labels.

    Attributes:
        _MODULE_NAME (str): The name of the module (in this case ``node.processing.feature_extraction_node``)
    """
    _MODULE_NAME: Final[str] = 'node.processing.feature_extraction_node'

    INPUT_FEATURES: Final[str] = 'features'
    INPUT_EVENTS: Final[str] = 'events'

    OUTPUT_FEATURES: Final[str] = 'features'
    OUTPUT_EVENTS: Final[str] = 'events'

    def _validate_parameters(self, parameters: dict):
        """ Validates the parameters passed to this node.

        :param parameters: The parameters to validate.
        :type parameters: dict
        """
        super()._validate_parameters(parameters)

    def _is_next_node_call_enabled(self) -> bool:
        """ This method will return ``True`` if the next node call is enabled. This method will always return ``True``
        because the next node call is always enabled.
        """
        return self._output_buffer[self.OUTPUT_EVENTS].get_data_count() > 0 and self._output_buffer[self.OUTPUT_FEATURES].get_data_count() > 0

    def _is_processing_condition_satisfied(self) -> bool:
        """ This method will return ``True`` if the processing condition is satisfied. This method will return ``True`` if the input buffer has data.

        :return: ``True`` if the input buffer has data, ``False`` otherwise.
        :rtype: bool
        """
        return self._input_buffer[self.INPUT_EVENTS].get_data_count() > 0 \
               and self._input_buffer[self.INPUT_EVENTS].get_data_count() >= self._input_buffer[
                   self.INPUT_FEATURES].get_data_count() \
               and self._input_buffer[self.INPUT_FEATURES].get_data_count() > 0

    def _process(self, data: Dict[str, FrameworkData]) -> Dict[str, FrameworkData]:
        """ This method processes the data that was inputted to the node. It extracts features and associates them with event labels.

        :param data: The data that was inputted to the node.
        :type data: Dict[str, FrameworkData]

        :return: The extracted features and associated labels.
        :rtype: Dict[str, FrameworkData]
        """
        signal_data = data[self.INPUT_FEATURES]
        event_data = data[self.INPUT_EVENTS]

        features_data = FrameworkData(sampling_frequency_hz=signal_data.sampling_frequency, channels=signal_data.channels)
        labels_data = FrameworkData(sampling_frequency_hz=event_data.sampling_frequency, channels=event_data.channels)

        for channel in signal_data.channels:

            features = []
            signal_channel_data = signal_data.get_data_on_channel(channel)

            for i in range(0, len(signal_channel_data), 5*signal_data.sampling_frequency):
                window = signal_data.get_data_on_channel(channel)[i:i+5*signal_data.sampling_frequency]

                if len(window) == signal_data.sampling_frequency * 5:
                    features.append(self.extract_features(window))
                    
            features_data.input_data_on_channel(features, channel)

        label_channel_data = event_data.get_data_on_channel('marker')
        labels = []

        for i in range(0, len(label_channel_data), 5*labels_data.sampling_frequency):
    
            window_label = event_data.get_data_on_channel('marker')[i:i+5*labels_data.sampling_frequency]

            if len(window_label) == labels_data.sampling_frequency * 5:
                counter = Counter(window_label)
                label,aux = counter.most_common(1)[0]#round(np.average(window_label))
                labels.append(label)
                    
        labels_data.input_data_on_channel(labels, 'marker')

        return {
            self.OUTPUT_FEATURES: features_data,
            self.OUTPUT_EVENTS: labels_data
        }

    def extract_features(self, signal):
        mav = np.mean(np.abs(signal))
        zc = len(np.where(np.diff(np.sign(signal)))[0])
        ssc = sum(np.sign(signal[i] - signal[i-1]) != np.sign(signal[i+1] - signal[i]) for i in range(1, len(signal)-1))
        wl = np.sum(np.abs(np.diff(signal)))
        return [mav, zc, ssc, wl]

    def _get_inputs(self) -> List[str]:
        """ Returns the input fields of this node.
        """
        return [
            self.INPUT_FEATURES,
            self.INPUT_EVENTS
        ]

    def _get_outputs(self) -> List[str]:
        """ Returns the output fields of this node.
        """
        return [
            self.OUTPUT_FEATURES,
            self.OUTPUT_EVENTS
        ]