from kenn.AdvancedCNN import AdvancedCNN
from kenn.utils import load_data


def main():
    train_x, train_y, test_x, test_y = load_data('weak_two')
    model = AdvancedCNN(train_x, train_y, test_x, test_y,
                        seg_len=650, num_channels=3, num_labels=9,
                        num_conv=3, filters=16, k_size=5, conv_strides=1, pool_size=4, pool_strides=4,
                        batch_size=100, learning_rate=0.00025, num_epochs=1000,
                        print_val_each_epoch=2, print_test_each_epoch=10, print_cm=False,
                        padding='same', cnn_type='1d')
    model.train()


if __name__ == '__main__':
    main()
