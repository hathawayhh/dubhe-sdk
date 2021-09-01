from dubhe_sdk.service import (
    train_decorator as train,
    predict_decorator as predict,
    inference_decorator as inference,
    pipeline_decorator as pipeline
)

from dubhe_sdk.service.context import (
    Context,
    ContextKey
)



import dubhe_sdk.service.run as service

service.image_run()
