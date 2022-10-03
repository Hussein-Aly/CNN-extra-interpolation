import pandas as pd


def pkl_reader(file):
    obj = pd.read_pickle(file)
    print(obj[0])


if __name__ == "__main__":
    inputs_file = './test/inputs.pkl'
    outputs_file = './predictions.pkl'
    outputs_file2 = './test/outputs.pkl'
    example = 'supplements/example_submission_random.pkl'

    pkl_reader(outputs_file2)