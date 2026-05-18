'''
Test model on TestData
'''

# System imports
import glob
import os
import pickle
import argparse
# 3rd party imports

# local imports


from .asd import ClassifyVVAD

# end file header
__author__      = 'Adrian Auer'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "path", help="path to a folder holding pickle files", type=str)

    parser.add_argument("-f", "--feature_type", help="Type for usage of a pretrained model.",
                        choices=["LipShape", "FaceShape"], type=str, default="LipShape")

    args = parser.parse_args()
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    # load pickle data
    for i, samplePath in enumerate(glob.glob(os.path.join(os.path.join(args.path, '**'), '*.pickle'))):
        with open(samplePath, 'rb') as file:
            sample = pickle.load(file)
            #  load pipeline (preprocess + model)
            pipeline = ClassifyVVAD(architecture=args.feature_type, averaging_window_size=1)
            # crank data in pipeline
            for j, frame in enumerate(sample['data']):
                pred = pipeline(frame) # TODO: ideally this is not necessary - we should use the same pipeline - but somehow pipeline.reset() is not working propperly
                if pred['scores'] != 0:
                    print(f"Prediction for sample {samplePath} after frame {j} is {pred['scores']}")
                    if sample['label'] and pred['scores'] >= 0.5:
                        tp+=1
                    elif sample['label'] and pred['scores'] < 0.5:
                        fn+=1
                    elif not sample['label'] and pred['scores'] >= 0.5:
                        fp+=1
                    elif not sample['label'] and pred['scores'] < 0.5:
                        tn+=1
                    else:
                        print(f'{sample=}, {pred=}')
                        raise ValueError("This should not happen")
        # TODO: clear buffer not working propperly
        # pipeline.reset()               
    # calc accuracy
    acc = (tp+tn) / (tp+tn+fp+fn)
    print(f"Accuracy for {args.feature_type} is {acc}")
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    # also calculate precision, recall, f1 score, fp rate, fn rate
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
    print(f"Precision: {precision}, Recall: {recall}, F1-Score: {f1_score}")
    print(f"FP Rate: {fp_rate}, FN Rate: {fn_rate}")