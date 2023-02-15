import argparse

import json

import helper

parser = argparse.ArgumentParser(description='Predicting flower name from an image along with the probability of that name.')
parser.add_argument('image_path', help='Path to image directory')
parser.add_argument('checkpoint', help='Given checkpoint of a saved model')
parser.add_argument('--top_k', help='Return top k  classes')
parser.add_argument('--category_names', help='Use a mapping of categories to names')


args = parser.parse_args()
top_k = 1 if args.top_k is None else int(args.top_k)  #default values
category_names = "cat_to_name.json" if args.category_names is None else args.category_names

model = helper.load_model(args.checkpoint)
print(model)


probs, predict_classes = helper.predict(helper.process_image(args.image_path), model, top_k)


with open(category_names, 'r') as f:
    cat_to_name = json.load(f)

classes = []
    
for predict_class in predict_classes:
    classes.append(cat_to_name[predict_class])

print(probs)
print(classes)
