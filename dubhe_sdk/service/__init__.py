def train_decorator():
    def do_train(f):
        setattr(f, "_is_train_func", True)
        return f

    return do_train


def predict_decorator():
    def do_predict(f):
        setattr(f, "_is_predict_func", True)
        return f

    return do_predict

def inference_decorator():
    def do_inference(f):
        setattr(f, "_is_inference_func", True)
        return f

    return do_inference

def pipeline_decorator():
    def do_pipeline(f):
        setattr(f, "_is_pipeline_func", True)

        return f

    return do_pipeline