import argparse
import helper

parser = argparse.ArgumentParser(description='Training neural network')
parser.add_argument('data_directory', help='Path to dataset')
parser.add_argument('--save_dir', help='Path to the checkpoint')
parser.add_argument('--arch', help='Model architecture')
parser.add_argument('--learning_rate', help='Learning rate')
parser.add_argument('--hidden_units', help='Number of hidden units')
parser.add_argument('--epochs', help='Number of epochs')


args = parser.parse_args()

#data_directory = 'flowers' if args.data_directory is None else args.data_directory
save_dir = '' if args.save_dir is None else args.save_dir
arch = 'vgg16' if args.arch is None else args.arch
learning_rate = 0.001 if args.learning_rate is None else int(args.learning_rate)
hidden_units = 1024 if args.hidden_units is None else float(args.hidden_units)
epochs = 10 if args.epochs is None else int(args.epochs)

train_data, train_dl, val_dl, test_dl = helper.load_data(args.data_directory,64)
model = helper.build_model(arch,hidden_units)
model.class_to_idx = train_data.class_to_idx

model, criterion = helper.train_nn(model,epochs,learning_rate,train_dl,val_dl)

helper.evaluate_model(model,test_dl,criterion)

helper.save_model(model, arch, hidden_units,epochs,learning_rate,save_dir)
