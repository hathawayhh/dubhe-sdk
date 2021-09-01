import typing
from dubhe_sdk.dataset import dataset

class Dataset(object):
    def __init__(self, ds: dataset.Dataset):
        self._org_dataset = ds

    def getCount(self):
        return self._org_dataset.getCount()

    def getId(self, index: int) -> str:
        return self._org_dataset.getId(index)

    def openFile(self, index: int, key: str = "default") -> typing.IO:
        """
        Return file of a dataset if has

        Args:
            index: index of item
            key: Key of the file, if there is only one image in each item, use 'default'
            or you must assign them different key in dubhe cloud

        :return: file: the file object of the item
        """
        return self._org_dataset.openFile(index, key)

    def getClassification(self, index: int, key: str = "default") -> str:
        return self._org_dataset.getClassification(index, key)

    def getDetection(self, index: int, key: str = "default") -> typing.List:
        return self._org_dataset.getDetection(index, key)

    def getMetadata(self, index: int, key: str = "default") -> str:
        return self._org_dataset.getMetadata(index, key)