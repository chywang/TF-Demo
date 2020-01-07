import numpy as np
import tensorflow as tf



def main(_):
    # train model
    print('Training model...')

    # define model
    emb_dim=100
    class_dim=10

    x = tf.placeholder(tf.float32, shape=[None, emb_dim])
    y = tf.placeholder(tf.int32, shape=[None, 10])

    w = tf.Variable(tf.zeros([emb_dim, class_dim]))
    b = tf.Variable(tf.zeros([class_dim]))

    logits = tf.matmul(x, w) + b

    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
    solver = tf.train.AdamOptimizer().minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for _ in range(1000):
        # generate random data
        batch_size=100
        x_batch = np.random.randn(batch_size, emb_dim)
        y_batch_labeled = np.random.randint(0, class_dim, size=batch_size)
        n_values = np.max(y_batch_labeled) + 1
        y_batch = np.eye(n_values)[y_batch_labeled]
        sess.run([loss, solver], feed_dict={x: x_batch, y: y_batch})

    print('Training complete...')

    # export model

    # export_path_base = sys.argv[-1]
    export_path = 'saved_model_temp'
    print('Exporting trained model to', export_path)
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)

    # Build the signature_def_map.
    classification_inputs = tf.saved_model.utils.build_tensor_info(x)
    classification_outputs_classes = tf.saved_model.utils.build_tensor_info(y)

    # 分类签名：算法类型+输入+输出（概率和名字）
    classification_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs={
                tf.saved_model.signature_constants.CLASSIFY_INPUTS:
                    classification_inputs
            },
            outputs={
                tf.saved_model.signature_constants.CLASSIFY_OUTPUT_CLASSES:
                    classification_outputs_classes,
            },
            method_name=tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME))

    tensor_info_x = tf.saved_model.utils.build_tensor_info(x)
    tensor_info_y = tf.saved_model.utils.build_tensor_info(y)

    # 预测签名：输入的x和输出的y
    prediction_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs={'input': tensor_info_x},
            outputs={'output': tensor_info_y},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

    # predict_images就是服务调用的方法
    # serving_default是没有输入签名时，使用的方法
    builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            'input_x':
                prediction_signature,
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                classification_signature,
        },
        main_op=tf.tables_initializer(),
        strip_default_attrs=True)

    # 保存
    builder.save()

    print('Done exporting!')


if __name__ == '__main__':
    tf.app.run()

