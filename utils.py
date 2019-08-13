from keras.models import Model
from keras.utils import multi_gpu_model


class ModelMGPU(Model):
    def __init__(self, ser_model, gpus):
        pmodel = multi_gpu_model(ser_model, gpus)
        self.__dict__.update(pmodel.__dict__)
        self._smodel = ser_model
        super().__init__()

    def __getattribute__(self, attrname):
        '''Override load and save methods to be used from the serial-model. The
           serial-model holds references to the weights in the multi-gpu model.
           '''
        if 'load' in attrname or 'save' in attrname:
            return getattr(self._smodel, attrname)
        else:
           return super(ModelMGPU, self).__getattribute__(attrname)