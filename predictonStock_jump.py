import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import pandas as pd
import time
tf.disable_v2_behavior()


# 获取训练集
def get_trainSet(data, TIME_STEP=20):
    normalized_data = data
    train_x, train_y = [], []
    for i in range(len(normalized_data) - TIME_STEP - 3):
        if i % 3 == 0:
            # 进行标准化
            mean = np.mean(normalized_data[i:i + TIME_STEP, :6], axis=0)
            std = np.std(normalized_data[i:i + TIME_STEP, :6], axis=0)
            x = (normalized_data[i:i + TIME_STEP, :6] - mean) / std
            y_open = (normalized_data[i + TIME_STEP:i + TIME_STEP + 3, [0]] - mean[0]) / std[0]
            y_close = (normalized_data[i + TIME_STEP:i + TIME_STEP + 3, [3]] - mean[3]) / std[3]
            y = np.concatenate((y_open, y_close), axis=1)
            train_x.append(x.tolist())
            train_y.append(y.flatten())

    return train_x, train_y


# basicLstm单元
def lstmCell():
    basicLstm = tf.nn.rnn_cell.BasicLSTMCell(rnn_units)
    # dropout
    drop = tf.compat.v1.nn.rnn_cell.DropoutWrapper(basicLstm, output_keep_prob=keep_prob)
    return drop


#  LSTM网络
def lstm(input_X):
    with tf.name_scope("LSTM"):
        cell = tf.nn.rnn_cell.MultiRNNCell([lstmCell() for i in range(layers_num)])
        with tf.name_scope("h0_state"):
            h0 = cell.zero_state(batch_size=tf.shape(input_X)[0], dtype=tf.float32)
        with tf.name_scope("RNN"):
            output, final_states = tf.nn.dynamic_rnn(cell, inputs=input_X, initial_state=h0)
        with tf.name_scope("lstm_outputs"):
            outputs = tf.reshape(output, shape=[tf.shape(input_X)[0], -1])
        with tf.name_scope("Weights"):
            weights = tf.Variable(tf.random.truncated_normal(shape=[TIME_STEP * rnn_units, OUTPUT_SIZE], stddev=0.1),
                                  dtype=tf.float32, name="weights")
        with tf.name_scope("biases"):
            bias = tf.Variable(tf.constant(0, shape=[OUTPUT_SIZE], dtype=tf.float32), name="bias")
        with tf.name_scope("pred_outputs"):
            pred = tf.matmul(outputs, weights) + bias
    return pred, final_states


#  训练模型
def trainSet(data):
    train_x, train_y = get_trainSet(data)
    x = tf.compat.v1.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, TIME_STEP, DIM], name='input-x')
    y = tf.compat.v1.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, OUTPUT_SIZE], name='input-y')
    pred, _ = lstm(x)
    with tf.name_scope("loss"):
        loss = tf.reduce_mean(tf.square(pred - y))
    train_step = tf.compat.v1.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

    saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(), max_to_keep=1)
    with tf.name_scope("Average_Error"):
        error_value = tf.reduce_mean(abs(pred - y))

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        loss_meanlist = []
        for j in range(10):
            losslist = []
            for i in range(len(train_x) - BATCH_SIZE):
                _, loss_val = sess.run([train_step, loss],
                                       feed_dict={x: train_x[i:i + BATCH_SIZE], y: train_y[i:i + BATCH_SIZE]})

                losslist.append(loss_val)
            loss_mean = np.mean(losslist)
            loss_meanlist.append(loss_mean)

        print("model_save: ", saver.save(sess, 'model_save/prediction.ckpt'))
        fig = plt.figure()
        print(loss_meanlist)
        plt.scatter(np.arange(len(loss_meanlist)), loss_meanlist)
        plt.show()


# 获取测试集
def get_testSet(data, TIME_STEP=20):
    normalized_data = data
    test_x, test_y, mean_list, std_list = [], [], [], []
    for i in range(len(normalized_data) - TIME_STEP - 3):
        mean, std = [], []
        if i % 3 == 0:
            mean = np.mean(normalized_data[i:i + TIME_STEP, :6], axis=0)
            std = np.std(normalized_data[i:i + TIME_STEP, :6], axis=0)
            x = (normalized_data[i:i + TIME_STEP, :6] - mean) / std
            y_open = (normalized_data[i + TIME_STEP:i + TIME_STEP + 3, [0]] - mean[0]) / std[0]
            y_close = (normalized_data[i + TIME_STEP:i + TIME_STEP + 3, [3]] - mean[3]) / std[3]
            y = np.concatenate((y_open, y_close), axis=1)
            test_x.append(x.tolist())
            test_y.append(y.flatten())
            mean_list.append(mean)
            std_list.append(std)
    return mean_list, std_list, test_x, test_y


# 测试模型
def testSet(data):
    mean, std, test_x, test_y = get_testSet(data)
    X = tf.compat.v1.placeholder(tf.float32, shape=[None, TIME_STEP, DIM])
    pred, _ = lstm(X)
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        model_file = tf.train.latest_checkpoint('model_save')
        saver.restore(sess, model_file)
        predict_open = []
        predict_close = []
        y_open = []
        y_close = []
        for i in range(len(test_y)):
            prob = sess.run(pred, feed_dict={X: [test_x[i]]})
            prediction = prob.flatten().reshape(3, 2)
            prediction_open = prediction[:, 0] * std[i][0] + mean[i][0]
            prediction_close = prediction[:, 1] * std[i][3] + mean[i][3]
            predict_open.extend(prediction_open)
            predict_close.extend(prediction_close)
            y_open.append(test_y[i][0] * std[i][0] + mean[i][0])
            y_open.append(test_y[i][2] * std[i][0] + mean[i][0])
            y_open.append(test_y[i][4] * std[i][0] + mean[i][0])
            y_close.append(test_y[i][1] * std[i][3] + mean[i][3])
            y_close.append(test_y[i][3] * std[i][3] + mean[i][3])
            y_close.append(test_y[i][5] * std[i][3] + mean[i][3])
    open_acc_rate, close_acc_rate = plotloss(predict_open, predict_close, y_open, y_close)
    print("预测后{}分钟".format(len(predict_open)))
    print("开盘价测试准确率为：{}，收盘价测试准确率为：{}".format(open_acc_rate, close_acc_rate))

    return predict_open, predict_close, y_open, y_close


# 画图
def plotloss(pred_open, pred_close, y_open, y_close):
    open_accuracy = []
    close_accuracy = []
    for i in range(len(pred_open) - 1):
        predopen_acc = pred_open[i + 1] - pred_open[i]
        open_acc = y_open[i + 1] - y_open[i]
        predclose_acc = pred_close[i + 1] - pred_close[i]
        close_acc = y_close[i + 1] - y_close[i]
        if predopen_acc >= 0 and open_acc >= 0 or predopen_acc < 0 and open_acc < 0:
            open_accuracy.append(i + 1)
        if predclose_acc >= 0 and close_acc >= 0 or predclose_acc < 0 and close_acc < 0:
            close_accuracy.append(i + 1)
    fig = plt.figure()
    # ax1 = plt.subplot(211)
    plt.plot(np.arange(len(pred_open)), pred_open, label='prediction_open', )
    plt.plot(np.arange(len(y_open)), y_open, label='open')
    plt.plot(np.arange(len(pred_close)), pred_close, label='prediction_close')
    plt.plot(np.arange(len(y_close)), y_close, label='close')
    plt.legend()
    # plt.xticks(np.arange(0, len(pred_close), 1))
    plt.grid()
    plt.show()
    print("预测开盘价为：", pred_open)
    print("真实开盘价为：", y_open)
    print("预测收盘价为：", pred_close)
    print("真实收盘价为：", y_close)
    open_acc_rate, close_acc_rate = len(open_accuracy) / len(y_open), len(close_accuracy) / len(y_close)
    return open_acc_rate, close_acc_rate


# 预测函数
def prediction(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    x_data = (data - mean) / std
    input_X = tf.placeholder(tf.float32, shape=[None, TIME_STEP, DIM])
    pred, _ = lstm(input_X)
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        model_file = tf.train.latest_checkpoint('model_save')
        saver.restore(sess, model_file)
        predict_open = []
        predict_close = []
        prob = sess.run(pred, feed_dict={input_X: [x_data]})

        prediction = prob.flatten().reshape(3, 2)
        prediction_open = prediction[:, 0] * std[0] + mean[0]
        prediction_close = prediction[:, 1] * std[3] + mean[3]

        predict_open.extend(prediction_open)
        predict_close.extend(prediction_close)

    return predict_open, predict_close


if __name__ == "__main__":
    # 定义常量
    OUTPUT_SIZE = 6
    BATCH_SIZE = 200
    TIME_STEP = 20
    DIM = 6
    layers_num = 2
    rnn_units = 10
    keep_prob = 1
    LEARNING_RATE = 0.001

    # 读取数据
    df = pd.read_csv('../data/2020-2-19-20_10to2020-3-14-10_33.json.csv')
    data = df.iloc[:30000, 1:7].values
    data_test = df.iloc[31600:31652, 1:7].values
    data_pre = df.iloc[31600:31620, 1:7].values
    # 测试程序运行时间
    start = time.perf_counter()
    # 训练模型
    # trainSet(data)

    # 测试模型
    # predict_open, predict_close, y_open, y_close = testSet(data_test)

    # 预测模型
    predict_open1, predict_close1 = prediction(data_pre)
    print("预测开盘价为：", predict_open1)
    print("预测收盘价为：", predict_close1)

    end = time.perf_counter()
    print('Running time: %s Seconds' % (end - start))
