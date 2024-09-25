import tensorflow as tf
from models.thumos_wo_mil import AdvCrossModalSimple, ModelParams
#from models.i2v_crossmodal_triplet_thumos14 import AdvCrossModalSimple, ModelParams
#from models.wiki_shallow import AdvCrossModalSimple, ModelParams
def main(_):
    graph = tf.Graph()
    model_params = ModelParams()
    model_params.update()
    config = tf.ConfigProto() #gpu_options=tf.GPUOptions(allow_growth=True)
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    with graph.as_default():
        model = AdvCrossModalSimple(model_params)
    with tf.Session(graph=graph,config=config) as sess:
        model.train(sess)
        # model.eval_random_rank()
        model.eval(sess)


if __name__ == '__main__':
    tf.app.run()