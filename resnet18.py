import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import  layers, Sequential, datasets, optimizers
class BasicBlock(layers.Layer):
    # 残差模块类, H(x)
    def __init__(self, filter_num, stride=1): 
        super(BasicBlock, self).__init__() # 继承Layer
        # f(x)包含了2个普通卷积层，创建卷积层1
        self.conv1 = layers.Conv2D(filter_num, (3, 3), strides=stride, padding='same')  # 减小stride倍！！！
        self.bn1 = layers.BatchNormalization()        
        self.relu = layers.Activation('relu')
        # 创建卷积层2
        self.conv2 = layers.Conv2D(filter_num, (3, 3), strides=1, padding='same') 		
        # shape没有减小
        self.bn2 = layers.BatchNormalization()


        if stride != 1:            
            self.downsample = Sequential()                    
            self.downsample.add(layers.Conv2D(filter_num, (1, 1), strides=stride)) 
        else: # 否则，直接连接
            self.downsample = lambda x:x


    def call(self, inputs, training=None):
        # 前向传播函数
        out = self.conv1(inputs) # 通过第一个卷积层        
        out = self.bn1(out, training=training)        
        out = self.relu(out)        
        out = self.conv2(out) 
        # 通过第二个卷积层     
        out = self.bn2(out, training=training)
        # 输入通过 identity()转换
        identity = self.downsample(inputs)
        # f(x)+x 运算
        output = layers.add([out, identity])
        # 再通过激活函数并返回
        output = tf.nn.relu(output)        
        return output
class ResNet(keras.Model):
    # 通用的ResNet 实现类
    def __init__(self, layer_dims, num_classes=10): # [2, 2, 2, 2]
        super(ResNet, self).__init__()
        # 根网络，预处理，[b, 32, 32, 3]
        self.stem = Sequential([layers.Conv2D(64, (3, 3), strides=(1, 1) ,padding='same'),
        # => [b, 32, 32, 64]
        layers.BatchNormalization(),
        layers.Activation('relu'),
        # => [b, 32, 32, 64]
        layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1),padding='same')
        ])

        # [b, 32, 32, 64] => [b, 32, 32, 64]
        self.layer1 = self.build_resblock(64, layer_dims[0])
        # [b, 32, 32, 64] => [b, 16, 16, 128]
        self.layer2 = self.build_resblock(128, layer_dims[1], stride=2)
        # [b, 16, 16, 128] => [b, 8, 8, 256]
        self.layer3 = self.build_resblock(256, layer_dims[2], stride=2)
        #  [b, 8, 8, 256] => [b, 4, 4, 512]
        self.layer4 = self.build_resblock(512, layer_dims[3], stride=2)

        # 通过Pooling 层将高宽降低为1x1，=> [b, 512]
        self.avgpool = layers.GlobalAveragePooling2D()
        # 最后连接一个全连接层分类，[b, 512] => [b,10]
        self.fc = layers.Dense(num_classes)

    @tf.function # 优化
    def call(self, inputs, training=None):
        # 通过根网络
        x = self.stem(inputs, training= training)

        # 一次通过4 个模块
        x = self.layer1(x, training= training)
        x = self.layer2(x, training= training)
        x = self.layer3(x, training= training)
        x = self.layer4(x, training= training)
        # 通过池化层
        x = self.avgpool(x)
        # 通过全连接层
        x = self.fc(x)
        return x



    def build_resblock(self, filter_num, blocks, stride=1): # 这个是在ResNet里的方法

        # 辅助函数，堆叠blocks 个BasicBlock，一个BasicBlock有两个卷积层
        res_blocks = Sequential() # 一个大类包含小类，使用时使用call方法

        # 只有第一个BasicBlock 的步长可能不为1，实现下采样
        res_blocks.add(BasicBlock(filter_num, stride))
        for _ in range(1, blocks):#其他BasicBlock 步长都为1
            res_blocks.add(BasicBlock(filter_num, stride=1))

        return res_blocks

def resnet18():
    # 通过调整模块内部BasicBlock 的数量和配置实现不同的ResNet
    return ResNet([2, 2, 2, 2])

@tf.function # 优化
def train_loss(x,y):
    with tf.GradientTape() as tape:
        # [b, 32, 32, 3] => [b, 10],前向传播
        logits = model(x, training=True)  # 参数传进去，用不用是另一回事，类的__call__()方法
        # [b] => [b, 10],one-hot 编码，y属于0~9
        y_onehot = tf.one_hot(y, depth=10)
        # 计算交叉熵
        loss = tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True)
        loss = tf.reduce_mean(loss)  # 我们需要标量
    # 计算梯度信息
    grads = tape.gradient(loss, model.trainable_variables)
    # 更新网络参数
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

@tf.function # 优化
def test_acc():
    total_num = 0
    total_correct = 0

    for x,y in test_db:

        logits = model(x, training=False)
        prob = tf.nn.softmax(logits, axis=1)
        pred = tf.argmax(prob, axis=1)
        pred = tf.cast(pred, dtype=tf.int32)

        correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
        correct = tf.reduce_sum(correct)

        total_num += x.shape[0]
        total_correct += int(correct)

    acc = total_correct / total_num
    
    return acc

def preprocess(x, y):
    # 将数据映射到-1~1
    x = 2*tf.cast(x, dtype=tf.float32) / 255. - 1
    y = tf.cast(y, dtype=tf.int32) # 类型转换
    return x,y

(x,y), (x_test, y_test) = datasets.cifar10.load_data() # 加载数据集

y = tf.squeeze(y, axis=1) # 删除不必要的维度
y_test = tf.squeeze(y_test, axis=1) 

train_db = tf.data.Dataset.from_tensor_slices((x,y)) # 构建训练集
train_db = train_db.shuffle(1000).map(preprocess).batch(512)

test_db = tf.data.Dataset.from_tensor_slices((x_test,y_test)) #构建测试集
test_db = test_db.map(preprocess).batch(512)

sample = next(iter(train_db))
print('sample:', sample[0].shape, sample[1].shape,tf.reduce_min(sample[0]), tf.reduce_max(sample[0]) ,sample[1].dtype)

#[b, 32, 32, 3] => [b, 1, 1, 512]
model = resnet18() # 调用函数resnet18，返回一个类
model.build(input_shape=(None, 32, 32, 3))
model.summary()
summary_writer = tf.summary.create_file_writer('log')

optimizer = optimizers.Adam(lr=1e-3)

def main():

    

    total_step = list(enumerate(train_db))[-1][0] + 1

    for epoch in range(60): # 训练epoch

        if epoch < 15:
            optimizer.learning_rate = 1e-3
        elif epoch >= 15 and epoch <= 25:
            optimizer.learning_rate = 1e-4
        elif epoch >= 26 and epoch <= 35:
            optimizer.learning_rate = 1e-5
        elif epoch >= 36 and epoch <= 45:
            optimizer.learning_rate = 1e-6
        else:
            optimizer.learning_rate = 1e-7
        

        for step, (x,y) in enumerate(train_db):
            loss=train_loss(x,y)

            with summary_writer.as_default(): 
                tf.summary.scalar('train-loss', float(loss), step=step+epoch*total_step)

            if step % 50 == 0:
                print(epoch, step, 'loss:', float(loss))

        acc = test_acc()
        print(epoch, 'acc:', acc)

        with summary_writer.as_default(): 
            tf.summary.scalar('test_acc', float(acc), step=epoch)

        tf.saved_model.save(model, 'model_savedmode')
        print('saving savedmodel.')

if __name__ == '__main__':
    main()
    
    
