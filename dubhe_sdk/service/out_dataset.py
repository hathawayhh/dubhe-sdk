import typing


class OutDataset(object):
    def Add(self, Id: str):
        pass

    def setOutFile(self, Id: str, func: typing.Callable[[typing.IO], None], key: str = "default") -> bool:
        """
        Set output to dst dataset file
        For example in a preprocess predict method, will use this method to crop input image and save them to
        another dataset.
        If no dst dataset is assigned, the func will not be called and all annotations will be saved into input
        dataset.
        Only available in predict method.

        :return: is_saved: true if func is called and new file is saved to dst dataset or false
        """
        pass

    def setClassification(self, Id: str, clazz: str, key: str = "default"):
        pass

    def setDetection(self, Id: str, detection: typing.List, key: str = "default"):
        pass

    def setMetadata(self, Id: str, value: str, key: str = "default"):
        pass

    def flush(self, force: bool = False) -> bool:
        """
        flush and upload all output to cloud, so that we do not need to cache too much data here
        if run in local, only save in local
        """
        pass
