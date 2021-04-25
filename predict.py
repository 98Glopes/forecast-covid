import argparse
import warnings

from predict_covid.model import CovidModel

warnings.simplefilter('ignore')

def print_results(forecast):
    for day, cases in forecast.items():
        print(F"{day} ->   {cases}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Provide a CLI to CovidModel'
        )
    parser.add_argument(
        'method', metavar='M', type=str,
        help='Choice between predict or update model',
        choices=['predict', 'update']                
        )
    parser.add_argument(
        '--days', type=int, default=1,
        help='Number of days to predict',
        required=False
        )

    args = parser.parse_args()

    model = CovidModel()

    if args.method == 'predict':
        model.load_model()
        forecast = model.predict(days=args.days)
        print_results(forecast)

    elif args.method == 'update':
        model.get_new_model(force_remote_dataset=True)
        print("Updated Model")