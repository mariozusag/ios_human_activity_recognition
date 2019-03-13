# Mario Zusag mariozusag@gmail.com
# 12.03.19
# Purpose:
# Description:
import coremltools
import os


def transform_h5_to_coreml(h5_model_path='../../data/output/experimental_results/'):
    output_path = "/".join(h5_model_path.split("/")[:-1])
    coreml_model = coremltools.converters.keras.convert(h5_model_path,
                                                        input_names=['sensor-data'],
                                                        output_names=['output'],
                                                        class_labels=['walking', 'jogging'])

    coreml_model.author = "Mario Zusag"
    coreml_model.short_description = 'Human activity recognition based on motionSense dataset'
    coreml_model.save(os.path.join(output_path, 'coreml_model.mlmodel'))
    print("Saved coreml model to {}".format(os.path.join(output_path, 'coreml_model.mlmodel')))


if __name__ == '__main__':
    transform_h5_to_coreml('../../data/output/experimental_results/LSTM_2019-03-13-14:15:20/model.h5')
