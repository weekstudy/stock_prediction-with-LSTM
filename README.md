# stock_prediction-with-LSTM
第三方库:  
tensorflow 2.1  
tensorflow-cpu 2.1  
numpy 1.18.1 
matplotlib 3.2.0  
pandas 1.0.1  
使用教程：  
1、如果想重新训练模型，则只需要以下几步：  
&nbsp; &nbsp;&nbsp;&nbsp;a、读取CSV文件，（文件内容与文件格式一样）  
&nbsp; &nbsp;&nbsp;&nbsp;b、将数据传入trainSet函数，注释掉主函数中的其他函数，此时trainSet函数就会训练模型，保存模型参数,并在本文件目录下生成一个文件夹model_save。  
2、如果想测试模型，则只需要以下几步：  
&nbsp; &nbsp;&nbsp;&nbsp;a、读取CSV文件，（文件内容与文件格式一样）  
&nbsp; &nbsp;&nbsp;&nbsp;b、调用testSet(),注释掉主函数中的其他函数，  
3、如果想预测，则调用prediction()函数，注释掉主函数中的其他函数。

